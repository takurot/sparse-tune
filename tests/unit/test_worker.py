from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import diags

from sparsetune import canonicalize_matrix
from sparsetune._backends import NativeSolveResult
from sparsetune._backends import UnsupportedBackendError
from sparsetune._runner import canonicalize_rhs
from sparsetune._runner import _SUPPORTED_DTYPES
from sparsetune._solve_worker import _read_config as _read_solve_config
from sparsetune._types import SolveStatus
from sparsetune._worker import (
    _read_config as _read_benchmark_config,
    _end_to_end_samples,
    _steady_state_samples,
    run_worker,
    status_for_exception,
)


class OutOfMemoryError(RuntimeError):
    pass


class RecordingBackend:
    id = "cupy:cuda:0"
    solver_impl = "mock.cg"

    def __init__(self) -> None:
        self.warmups = 0

    def prepare(
        self,
        matrix: object,
        rhs: np.ndarray,
        *,
        dtype: str,
    ) -> tuple[object, np.ndarray]:
        return matrix, rhs.astype(dtype)

    def warmup(
        self,
        prepared: tuple[object, np.ndarray],
        *,
        rtol: float,
        atol: float,
        max_iter: int,
    ) -> None:
        self.warmups += 1

    def solve_prepared(
        self,
        prepared: tuple[object, np.ndarray],
        *,
        rtol: float,
        atol: float,
        max_iter: int,
    ) -> NativeSolveResult:
        return NativeSolveResult(
            solution=np.ones(2),
            info=0,
            iterations=1,
        )

    def synchronize(self) -> None:
        return None

    def fetch_solution(self, native: NativeSolveResult) -> np.ndarray:
        return native.solution

    def release(self, prepared: tuple[object, np.ndarray]) -> None:
        return None


def test_workers_share_the_parent_supported_dtype_set(tmp_path: Path) -> None:
    future_dtype = "float16"
    benchmark_config = tmp_path / "benchmark.json"
    solve_config = tmp_path / "solve.json"
    benchmark_config.write_text(
        json.dumps(
            {
                "dtype": future_dtype,
                "rtol": 1.0e-6,
                "atol": 0.0,
                "max_iter": 50,
                "runs": 1,
                "measure": ["end-to-end"],
            }
        ),
        encoding="utf-8",
    )
    solve_config.write_text(
        json.dumps(
            {
                "dtype": future_dtype,
                "rtol": 1.0e-6,
                "atol": 0.0,
                "max_iter": 50,
            }
        ),
        encoding="utf-8",
    )

    _SUPPORTED_DTYPES.add(future_dtype)
    try:
        assert _read_benchmark_config(benchmark_config)["dtype"] == future_dtype
        assert _read_solve_config(solve_config)["dtype"] == future_dtype
    finally:
        _SUPPORTED_DTYPES.remove(future_dtype)


def test_worker_exception_statuses_are_structured() -> None:
    assert status_for_exception(MemoryError()) is SolveStatus.OOM
    assert status_for_exception(OutOfMemoryError()) is SolveStatus.OOM
    assert (
        status_for_exception(UnsupportedBackendError("missing"))
        is SolveStatus.UNSUPPORTED
    )
    assert (
        status_for_exception(RuntimeError("unexpected")) is SolveStatus.INTERNAL_ERROR
    )


def test_run_worker_executes_requested_measurement_modes(tmp_path: Path) -> None:
    matrix = diags(
        [-np.ones(4), 2.0 * np.ones(5), -np.ones(4)],
        offsets=[-1, 0, 1],
        format="csr",
    )
    matrix_path, canonical = canonicalize_matrix(matrix, "float64", tmp_path)
    rhs_path, _ = canonicalize_rhs(
        None,
        canonical,
        dtype_str="float64",
        work_dir=tmp_path,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "dtype": "float64",
                "rtol": 1.0e-6,
                "atol": 0.0,
                "max_iter": 50,
                "runs": 2,
                "measure": ["end-to-end", "steady-state"],
            }
        ),
        encoding="utf-8",
    )

    result = run_worker(
        "scipy:cpu",
        matrix_path,
        rhs_path,
        config_path,
    )

    assert result.status is SolveStatus.CONVERGED
    assert len(result.samples) == 4
    assert result.setup_seconds > 0.0
    assert result.solve_seconds > 0.0
    for sample in result.samples:
        assert sample.total_seconds >= (
            sample.setup_seconds + sample.solve_seconds + sample.transfer_seconds
        )
    assert all(
        sample.setup_seconds > 0.0
        for sample in result.samples
        if sample.measure == "end-to-end"
    )
    assert all(
        sample.setup_seconds == 0.0
        for sample in result.samples
        if sample.measure == "steady-state"
    )


@pytest.mark.parametrize(
    ("measures", "expected_warmups"),
    [
        (["end-to-end"], 1),
        (["steady-state"], 1),
        (["end-to-end", "steady-state"], 2),
    ],
)
def test_each_measurement_mode_has_one_untimed_warmup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    measures: list[str],
    expected_warmups: int,
) -> None:
    backend = RecordingBackend()
    monkeypatch.setattr("sparsetune._worker.get_backend", lambda _backend: backend)
    matrix = diags([np.ones(2)], offsets=[0], format="csr")
    matrix_path, canonical = canonicalize_matrix(matrix, "float64", tmp_path)
    rhs_path, _ = canonicalize_rhs(
        None,
        canonical,
        dtype_str="float64",
        work_dir=tmp_path,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "dtype": "float64",
                "rtol": 1.0e-6,
                "atol": 0.0,
                "max_iter": 50,
                "runs": 1,
                "measure": measures,
            }
        ),
        encoding="utf-8",
    )

    run_worker(backend.id, matrix_path, rhs_path, config_path)

    assert backend.warmups == expected_warmups


def test_timing_fields_match_the_documented_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = RecordingBackend()
    matrix = diags([np.ones(2)], offsets=[0], format="csr")
    rhs = np.ones(2)
    config = {
        "dtype": "float64",
        "rtol": 1.0e-6,
        "atol": 0.0,
        "max_iter": 50,
        "runs": 1,
    }
    clock = iter([0.0, 2.0, 2.0, 5.0, 6.0, 10.0, 13.0, 14.0])
    monkeypatch.setattr("sparsetune._worker.time.perf_counter", lambda: next(clock))

    end_to_end = _end_to_end_samples(backend, matrix, rhs, config)[0]
    steady_state = _steady_state_samples(backend, matrix, rhs, config)[0]

    assert end_to_end.setup_seconds == 2.0
    assert end_to_end.solve_seconds == 3.0
    assert end_to_end.transfer_seconds == 1.0
    assert end_to_end.total_seconds == 6.0
    assert steady_state.setup_seconds == 0.0
    assert steady_state.solve_seconds == 3.0
    assert steady_state.transfer_seconds == 1.0
    assert steady_state.total_seconds == 4.0

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
from scipy.io import mmwrite
from scipy.sparse import csr_matrix

from sparsetune import canonicalize_matrix
from sparsetune._benchmark import recommend
from sparsetune._runner import (
    canonicalize_rhs,
    classify_solve,
    run_backend_in_subprocess,
    run_solve_in_subprocess,
)
from sparsetune._types import SolveStatus
from sparsetune._types import RunSample
from sparsetune._types import SolverResult


def test_canonicalize_rhs_inputs_are_equivalent(tmp_path: Path) -> None:
    matrix = csr_matrix([[4.0, 1.0], [1.0, 3.0]])
    matrix_path, canonical = canonicalize_matrix(matrix, "float32", tmp_path)
    expected = np.asarray(matrix @ np.ones(2), dtype=np.float32)
    rhs_mtx = tmp_path / "rhs.mtx"
    mmwrite(rhs_mtx, expected.reshape(-1, 1))

    omitted_path, omitted = canonicalize_rhs(
        None,
        canonical,
        dtype_str="float32",
        work_dir=tmp_path / "omitted",
    )
    file_path, from_file = canonicalize_rhs(
        rhs_mtx,
        canonical,
        dtype_str="float32",
        work_dir=tmp_path / "file",
    )
    array_path, from_array = canonicalize_rhs(
        expected,
        canonical,
        dtype_str="float32",
        work_dir=tmp_path / "array",
    )

    assert matrix_path.is_file()
    for path, rhs in (
        (omitted_path, omitted),
        (file_path, from_file),
        (array_path, from_array),
    ):
        assert path.is_file()
        assert rhs.dtype == np.float32
        np.testing.assert_array_equal(rhs, expected)
        np.testing.assert_array_equal(np.load(path, allow_pickle=False), expected)


@pytest.mark.parametrize(
    ("rhs", "message"),
    [
        (np.ones((2, 2)), "single vector"),
        (np.ones(3), "matrix size"),
        (np.array([1.0, np.nan]), "finite"),
    ],
)
def test_canonicalize_rhs_rejects_invalid_values(
    tmp_path: Path,
    rhs: np.ndarray,
    message: str,
) -> None:
    _, canonical = canonicalize_matrix(
        csr_matrix(np.eye(2)),
        "float64",
        tmp_path,
    )

    with pytest.raises(ValueError, match=message):
        canonicalize_rhs(
            rhs,
            canonical,
            dtype_str="float64",
            work_dir=tmp_path,
        )


@pytest.mark.parametrize(
    "values",
    [
        np.asarray([1.0 + 2.0j, 3.0 + 4.0j]),
        np.asarray([1.0 + 0.0j, 3.0 + 0.0j]),
    ],
)
def test_canonicalize_rhs_rejects_complex_arrays_before_conversion(
    tmp_path: Path,
    values: np.ndarray,
) -> None:
    _, canonical = canonicalize_matrix(
        csr_matrix(np.eye(2)),
        "float64",
        tmp_path,
    )

    with pytest.raises(ValueError, match="complex"):
        canonicalize_rhs(
            values,
            canonical,
            dtype_str="float64",
            work_dir=tmp_path,
        )


@pytest.mark.parametrize("imaginary", [2.0, 0.0])
def test_canonicalize_rhs_rejects_complex_matrix_market_before_conversion(
    tmp_path: Path,
    imaginary: float,
) -> None:
    _, canonical = canonicalize_matrix(
        csr_matrix(np.eye(2)),
        "float64",
        tmp_path,
    )
    rhs_path = tmp_path / "complex_rhs.mtx"
    mmwrite(
        rhs_path,
        np.asarray([1.0 + imaginary * 1j, 3.0 + 0.0j]).reshape(-1, 1),
    )

    with pytest.raises(ValueError, match="complex"):
        canonicalize_rhs(
            rhs_path,
            canonical,
            dtype_str="float64",
            work_dir=tmp_path,
        )


@pytest.mark.parametrize(
    ("info", "solution", "residual", "threshold", "expected"),
    [
        (0, [np.nan], 0.0, 1.0, SolveStatus.NAN_INF),
        (0, [1.0], 0.1, 0.2, SolveStatus.CONVERGED),
        (0, [1.0], 0.3, 0.2, SolveStatus.ACCURACY_FAILED),
        (4, [1.0], 0.0, 0.2, SolveStatus.MAX_ITER),
        (-1, [1.0], 0.0, 0.2, SolveStatus.BREAKDOWN),
    ],
)
def test_classify_solve_checks_finite_values_before_native_info(
    info: int,
    solution: list[float],
    residual: float,
    threshold: float,
    expected: SolveStatus,
) -> None:
    assert classify_solve(info, np.asarray(solution), residual, threshold) is expected


def test_timeout_becomes_structured_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def time_out(*_args: object, **_kwargs: object) -> object:
        raise subprocess.TimeoutExpired(["python"], 0.01)

    monkeypatch.setattr(subprocess, "run", time_out)

    result = run_backend_in_subprocess(
        "scipy:cpu",
        tmp_path / "matrix.npz",
        tmp_path / "rhs.npy",
        {},
        timeout=0.01,
    )

    assert result.status is SolveStatus.TIMEOUT
    assert result.error == "Timed out after 0.01 seconds"


def test_crash_diagnostic_is_bounded_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    secret = "do-not-leak-this-token"

    def crash(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            ["python"],
            2,
            stdout=f"API_TOKEN={secret}",
            stderr=f'API_TOKEN="{secret}" ' + ("x" * 1000),
        )

    monkeypatch.setattr(subprocess, "run", crash)

    result = run_backend_in_subprocess(
        "scipy:cpu",
        tmp_path / "matrix.npz",
        tmp_path / "rhs.npy",
        {},
        timeout=1.0,
    )

    assert result.status is SolveStatus.PROCESS_CRASH
    assert result.error is not None
    assert secret not in result.error
    assert "[REDACTED]" in result.error
    assert len(result.error) <= 500


def test_non_finite_worker_result_is_a_process_crash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def malformed(
        command: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        result_path = Path(command[command.index("--result") + 1])
        payload = SolverResult(
            backend="scipy:cpu",
            solver_impl="scipy.sparse.linalg.cg",
            dtype="float64",
            transfer_seconds=0.0,
            setup_seconds=0.0,
            solve_seconds=0.0,
            total_seconds=float("nan"),
            iterations=0,
            residual_norm=0.0,
            relative_residual=0.0,
            convergence_threshold=0.0,
            pool_used_gb=None,
            status=SolveStatus.CONVERGED,
            error=None,
        )
        result_path.write_text(payload.to_json(), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", malformed)

    result = run_backend_in_subprocess(
        "scipy:cpu",
        tmp_path / "matrix.npz",
        tmp_path / "rhs.npy",
        {},
        timeout=1.0,
    )

    assert result.status is SolveStatus.PROCESS_CRASH
    assert result.error == "Worker returned a malformed result"


def test_malformed_worker_result_is_a_process_crash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def malformed(
        command: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        result_path = Path(command[command.index("--result") + 1])
        result_path.write_text(json.dumps({"status": "converged"}), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", malformed)

    result = run_backend_in_subprocess(
        "scipy:cpu",
        tmp_path / "matrix.npz",
        tmp_path / "rhs.npy",
        {},
        timeout=1.0,
    )

    assert result.status is SolveStatus.PROCESS_CRASH
    assert result.error == "Worker returned a malformed result"


def _valid_worker_payload() -> dict[str, object]:
    samples = [
        RunSample(
            measure=measure,
            transfer_seconds=0.1,
            setup_seconds=0.2 if measure == "end-to-end" else 0.0,
            solve_seconds=0.3,
            total_seconds=0.6 if measure == "end-to-end" else 0.4,
            iterations=2,
            residual_norm=1.0e-8,
            relative_residual=1.0e-9,
            convergence_threshold=1.0e-6,
            status=SolveStatus.CONVERGED,
        )
        for measure in ("end-to-end", "steady-state")
    ]
    return SolverResult(
        backend="scipy:cpu",
        solver_impl="scipy.sparse.linalg.cg",
        dtype="float64",
        transfer_seconds=0.1,
        setup_seconds=0.2,
        solve_seconds=0.3,
        total_seconds=0.6,
        iterations=2,
        residual_norm=1.0e-8,
        relative_residual=1.0e-9,
        convergence_threshold=1.0e-6,
        pool_used_gb=None,
        status=SolveStatus.CONVERGED,
        error=None,
        samples=samples,
    ).to_dict()


def _run_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    payload: dict[str, object],
) -> SolverResult:
    def complete(
        command: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        result_path = Path(command[command.index("--result") + 1])
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", complete)
    return run_backend_in_subprocess(
        "scipy:cpu",
        tmp_path / "matrix.npz",
        tmp_path / "rhs.npy",
        {
            "dtype": "float64",
            "rtol": 1.0e-6,
            "atol": 0.0,
            "max_iter": 50,
            "runs": 1,
            "measure": ["end-to-end", "steady-state"],
        },
        timeout=1.0,
    )


@pytest.mark.parametrize(
    ("location", "field"),
    [
        ("result", "transfer_seconds"),
        ("result", "setup_seconds"),
        ("result", "solve_seconds"),
        ("result", "total_seconds"),
        ("result", "iterations"),
        ("result", "residual_norm"),
        ("result", "relative_residual"),
        ("result", "convergence_threshold"),
        ("result", "pool_used_gb"),
        ("sample", "transfer_seconds"),
        ("sample", "setup_seconds"),
        ("sample", "solve_seconds"),
        ("sample", "total_seconds"),
        ("sample", "iterations"),
        ("sample", "residual_norm"),
        ("sample", "relative_residual"),
        ("sample", "convergence_threshold"),
        ("sample", "pool_used_gb"),
    ],
)
def test_negative_worker_metrics_are_a_process_crash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    location: str,
    field: str,
) -> None:
    payload = _valid_worker_payload()
    target = (
        payload if location == "result" else payload["samples"][0]  # type: ignore[index]
    )
    target[field] = -1  # type: ignore[index]

    result = _run_payload(monkeypatch, tmp_path, payload)

    assert result.status is SolveStatus.PROCESS_CRASH
    assert result.error == "Worker returned a malformed result"


@pytest.mark.parametrize(
    ("location", "field", "value"),
    [
        ("result", "dtype", "garbage"),
        ("result", "dtype", "float32"),
        ("result", "status", "converged"),
        ("result", "error", "unexpected"),
        ("sample", "measure", "unknown"),
        ("sample", "status", "accuracy_failed"),
        ("sample", "total_seconds", 0.1),
    ],
)
def test_inconsistent_worker_semantics_are_a_process_crash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    location: str,
    field: str,
    value: object,
) -> None:
    payload = _valid_worker_payload()
    if location == "result" and field == "status":
        payload["samples"][0]["status"] = "accuracy_failed"  # type: ignore[index]
    target = (
        payload if location == "result" else payload["samples"][0]  # type: ignore[index]
    )
    target[field] = value  # type: ignore[index]

    result = _run_payload(monkeypatch, tmp_path, payload)

    assert result.status is SolveStatus.PROCESS_CRASH
    assert result.error == "Worker returned a malformed result"
    assert recommend([result])["end_to_end"].backend is None


def test_missing_requested_samples_are_a_process_crash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _valid_worker_payload()
    payload["samples"] = payload["samples"][:1]  # type: ignore[index]

    result = _run_payload(monkeypatch, tmp_path, payload)

    assert result.status is SolveStatus.PROCESS_CRASH


@pytest.mark.parametrize(
    ("status", "write_solution"),
    [
        (SolveStatus.CONVERGED, False),
        (SolveStatus.OOM, True),
    ],
)
def test_solution_presence_must_match_worker_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status: SolveStatus,
    write_solution: bool,
) -> None:
    payload = _valid_worker_payload()
    payload["samples"] = payload["samples"][:1]  # type: ignore[index]
    if status is SolveStatus.OOM:
        payload.update(
            {
                "solver_impl": "",
                "transfer_seconds": 0.0,
                "setup_seconds": 0.0,
                "solve_seconds": 0.0,
                "total_seconds": 0.0,
                "iterations": 0,
                "residual_norm": 0.0,
                "relative_residual": None,
                "convergence_threshold": 0.0,
                "pool_used_gb": None,
                "status": status.value,
                "error": "Backend ran out of memory",
                "samples": [],
            }
        )

    def complete(
        command: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        result_path = Path(command[command.index("--result") + 1])
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        if write_solution:
            solution_path = Path(command[command.index("--solution") + 1])
            np.save(solution_path, np.ones(2), allow_pickle=False)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", complete)
    result, solution = run_solve_in_subprocess(
        "scipy:cpu",
        tmp_path / "matrix.npz",
        tmp_path / "rhs.npy",
        {
            "dtype": "float64",
            "rtol": 1.0e-6,
            "atol": 0.0,
            "max_iter": 50,
        },
        timeout=1.0,
        expected_size=2,
    )

    assert result.status is SolveStatus.PROCESS_CRASH
    assert result.error == "Worker returned a malformed solve result"
    assert solution is None

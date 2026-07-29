from __future__ import annotations

from pathlib import Path
import importlib
import json
import subprocess

import pytest
from scipy.sparse import csr_matrix

from sparsetune import RunSample, SolveStatus, SolverResult, benchmark
from sparsetune._benchmark import _probe_backend, recommend


def _result(
    backend: str,
    *,
    total: float,
    solve: float,
    status: SolveStatus = SolveStatus.CONVERGED,
) -> SolverResult:
    samples = [
        RunSample(
            measure="end-to-end",
            transfer_seconds=max(total - solve, 0.0),
            setup_seconds=0.0,
            solve_seconds=solve,
            total_seconds=total,
            iterations=2,
            residual_norm=0.0,
            relative_residual=0.0,
            convergence_threshold=1.0e-6,
            status=status,
        ),
        RunSample(
            measure="steady-state",
            transfer_seconds=0.0,
            setup_seconds=0.0,
            solve_seconds=solve,
            total_seconds=solve,
            iterations=2,
            residual_norm=0.0,
            relative_residual=0.0,
            convergence_threshold=1.0e-6,
            status=status,
        ),
    ]
    return SolverResult(
        backend=backend,
        solver_impl="cg",
        dtype="float64",
        transfer_seconds=max(total - solve, 0.0),
        setup_seconds=0.0,
        solve_seconds=solve,
        total_seconds=total,
        iterations=2,
        residual_norm=0.0,
        relative_residual=0.0,
        convergence_threshold=1.0e-6,
        pool_used_gb=None,
        status=status,
        error=None,
        samples=samples,
    )


def test_recommend_selects_fastest_converged_result_per_mode() -> None:
    cpu = _result("scipy:cpu", total=1.0, solve=0.8)
    gpu = _result("cupy:cuda:0", total=1.2, solve=0.2)

    recommendations = recommend([gpu, cpu])

    assert recommendations["end_to_end"].backend == "scipy:cpu"
    assert recommendations["steady_state"].backend == "cupy:cuda:0"
    assert recommendations["end_to_end"].speedup == pytest.approx(1.2)
    assert recommendations["steady_state"].speedup == pytest.approx(4.0)
    assert recommendations["steady_state"].break_even_solves == 2


def test_recommend_excludes_failed_and_inaccurate_results() -> None:
    good = _result("scipy:cpu", total=3.0, solve=3.0)
    inaccurate = _result(
        "cupy:cuda:0",
        total=0.1,
        solve=0.1,
        status=SolveStatus.ACCURACY_FAILED,
    )

    recommendations = recommend([inaccurate, good])

    assert recommendations["end_to_end"].backend == "scipy:cpu"
    assert recommendations["steady_state"].backend == "scipy:cpu"


def test_recommend_is_deterministic_and_handles_all_failed() -> None:
    first = _result("z:backend", total=1.0, solve=1.0)
    second = _result("a:backend", total=1.0, solve=1.0)
    failed = _result(
        "failed",
        total=0.0,
        solve=0.0,
        status=SolveStatus.INTERNAL_ERROR,
    )

    assert recommend([first, second])["end_to_end"].backend == "a:backend"
    assert recommend([failed])["end_to_end"].backend is None
    assert recommend([failed])["steady_state"].backend is None


def test_break_even_handles_no_saving_and_zero_overhead() -> None:
    cpu = _result("scipy:cpu", total=1.0, solve=1.0)
    slower_gpu = _result("cupy:cuda:0", total=2.0, solve=1.1)
    zero_overhead_gpu = _result("cupy:cuda:1", total=0.5, solve=0.5)

    assert recommend([cpu, slower_gpu])["end_to_end"].break_even_solves is None
    assert recommend([cpu, zero_overhead_gpu])["steady_state"].break_even_solves == 1


def test_recommend_isolates_failures_by_measurement_mode() -> None:
    cpu = _result("scipy:cpu", total=2.0, solve=1.0)
    gpu = _result("cupy:cuda:0", total=1.0, solve=0.5)
    gpu.samples[0].status = SolveStatus.ACCURACY_FAILED
    gpu.status = SolveStatus.ACCURACY_FAILED

    recommendations = recommend([gpu, cpu])

    assert recommendations["end_to_end"].backend == "scipy:cpu"
    assert recommendations["steady_state"].backend == "cupy:cuda:0"


def test_recommend_requires_samples_for_the_requested_mode() -> None:
    result = _result("scipy:cpu", total=1.0, solve=0.5)
    result.samples = [
        sample for sample in result.samples if sample.measure == "end-to-end"
    ]

    recommendations = recommend([result])

    assert recommendations["end_to_end"].backend == "scipy:cpu"
    assert recommendations["steady_state"].backend is None


def test_benchmark_orchestrates_inputs_and_retains_unsupported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    successful = _result("scipy:cpu", total=1.0, solve=0.8)

    def fake_probe(
        backend_id: str,
        _dtype: str,
    ) -> tuple[str | None, dict[str, object] | None]:
        if backend_id == "scipy:cpu":
            return None, {
                "backend": backend_id,
                "kind": "cpu",
                "scipy_version": "1.14",
            }
        return "unavailable", None

    def fake_run(
        backend_id: str,
        _matrix_path: Path,
        _rhs_path: Path,
        _config: dict[str, object],
        *,
        timeout: float,
    ) -> SolverResult:
        calls.append(backend_id)
        assert timeout == 4.0
        return successful

    monkeypatch.setattr("sparsetune._benchmark._probe_backend", fake_probe)
    monkeypatch.setattr(
        "sparsetune._benchmark.run_backend_in_subprocess",
        fake_run,
    )

    report = benchmark(
        csr_matrix([[4.0, 1.0], [1.0, 3.0]]),
        backends=["scipy:cpu", "cupy:cuda:0"],
        runs=2,
        timeout=4.0,
    )

    assert calls == ["scipy:cpu"]
    assert report.matrix.spd_status == "screen_passed"
    assert [result.status for result in report.results] == [
        SolveStatus.CONVERGED,
        SolveStatus.UNSUPPORTED,
    ]
    assert report.results[1].error == "unavailable"
    assert report.environment["python"]
    assert report.environment["numpy"]
    assert report.environment["scipy"]


@pytest.mark.parametrize("backends", [[], ()])
def test_benchmark_rejects_explicit_empty_backends_before_processing(
    monkeypatch: pytest.MonkeyPatch,
    backends: list[str] | tuple[()],
) -> None:
    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("empty backends must fail before matrix or worker processing")

    monkeypatch.setattr("sparsetune._benchmark.diagnose_matrix", unexpected)
    monkeypatch.setattr("sparsetune._benchmark.canonicalize_matrix", unexpected)
    monkeypatch.setattr("sparsetune._benchmark._probe_backend", unexpected)
    monkeypatch.setattr(
        "sparsetune._benchmark.run_backend_in_subprocess",
        unexpected,
    )

    with pytest.raises(ValueError, match="at least one backend is required"):
        benchmark(object(), backends=backends)  # type: ignore[arg-type]


def test_benchmark_none_uses_default_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probed: list[str] = []

    def unavailable(
        backend: str,
        _dtype: str,
    ) -> tuple[str, None]:
        probed.append(backend)
        return "unavailable", None

    monkeypatch.setattr("sparsetune._benchmark._probe_backend", unavailable)

    benchmark(
        csr_matrix([[4.0, 1.0], [1.0, 3.0]]),
        backends=None,
        runs=1,
    )

    assert probed == ["scipy:cpu", "cupy:cuda:0"]


def test_benchmark_does_not_import_cupy_in_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []
    real_import = importlib.import_module

    def poison_cupy(name: str) -> object:
        if name == "cupy":
            imported.append(name)
            raise AssertionError("CuPy must not be imported in the parent")
        return real_import(name)

    monkeypatch.setattr(
        "sparsetune._benchmark.importlib.import_module",
        poison_cupy,
    )

    report = benchmark(
        csr_matrix([[4.0, 1.0], [1.0, 3.0]]),
        backends=["scipy:cpu"],
        runs=1,
    )

    assert report.results[0].status is SolveStatus.CONVERGED
    assert imported == []


def test_probe_accepts_validated_cuda_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = {
        "backend": "cupy:cuda:0",
        "kind": "cuda",
        "gpu_uuid": "gpu-uuid",
        "gpu_model": "Example GPU",
        "cuda_driver": 12080,
        "cuda_runtime": 12060,
        "cupy_version": "14.0.0",
    }
    monkeypatch.setattr(
        "sparsetune._benchmark.subprocess.run",
        lambda *_a, **_k: subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(identity),
            stderr="",
        ),
    )

    error, parsed = _probe_backend("cupy:cuda:0", "float64")

    assert error is None
    assert parsed == identity


@pytest.mark.parametrize(
    "identity",
    [
        {"backend": "cupy:cuda:0", "kind": "cuda"},
        {
            "backend": "cupy:cuda:1",
            "kind": "cuda",
            "gpu_uuid": "gpu-uuid",
            "gpu_model": "Example GPU",
            "cuda_driver": 12080,
            "cuda_runtime": 12060,
            "cupy_version": "14.0.0",
        },
    ],
)
def test_probe_rejects_malformed_or_mismatched_identity(
    monkeypatch: pytest.MonkeyPatch,
    identity: dict[str, object],
) -> None:
    monkeypatch.setattr(
        "sparsetune._benchmark.subprocess.run",
        lambda *_a, **_k: subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(identity),
            stderr="",
        ),
    )

    error, parsed = _probe_backend("cupy:cuda:0", "float64")

    assert error == "Backend capability probe returned malformed identity"
    assert parsed is None


def test_benchmark_runs_scipy_worker_and_preserves_raw_samples() -> None:
    report = benchmark(
        csr_matrix([[4.0, 1.0], [1.0, 3.0]]),
        backends=["scipy:cpu"],
        runs=1,
    )

    result = report.results[0]
    assert result.status is SolveStatus.CONVERGED
    assert [sample.measure for sample in result.samples] == [
        "end-to-end",
        "steady-state",
    ]
    assert report.best("end-to-end") is result
    assert report.best("steady-state") is result


def test_benchmark_enforces_spd_rules_before_probing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probes: list[str] = []
    monkeypatch.setattr(
        "sparsetune._benchmark._probe_backend",
        lambda backend, _dtype: probes.append(backend),
    )

    with pytest.raises(ValueError, match="not eligible"):
        benchmark(csr_matrix([[1.0, 2.0], [0.0, 1.0]]), assume_spd=True)

    with pytest.raises(ValueError, match="assume_spd"):
        benchmark(csr_matrix([[0.0, 0.0], [0.0, 0.0]]))

    assert probes == []

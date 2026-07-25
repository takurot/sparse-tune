from __future__ import annotations

from pathlib import Path

import pytest
from scipy.sparse import csr_matrix

from sparsetune import RunSample, SolveStatus, SolverResult, benchmark
from sparsetune._benchmark import recommend


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
    assert recommend([cpu, zero_overhead_gpu])["steady_state"].break_even_solves == 0


def test_benchmark_orchestrates_inputs_and_retains_unsupported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    successful = _result("scipy:cpu", total=1.0, solve=0.8)

    def fake_probe(backend_id: str, _dtype: str) -> str | None:
        return None if backend_id == "scipy:cpu" else "unavailable"

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

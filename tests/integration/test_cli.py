from __future__ import annotations

import json
from pathlib import Path

import pytest

from sparsetune import (
    BenchmarkResult,
    MatrixInfo,
    Recommendation,
    SolveResult,
    SolveStatus,
    SolverResult,
)
from sparsetune._cli import main


def _solver_result(status: SolveStatus = SolveStatus.CONVERGED) -> SolverResult:
    return SolverResult(
        backend="scipy:cpu",
        solver_impl="scipy.sparse.linalg.cg",
        dtype="float64",
        transfer_seconds=0.0,
        setup_seconds=0.1,
        solve_seconds=0.2,
        total_seconds=0.3,
        iterations=2,
        residual_norm=0.0,
        relative_residual=0.0,
        convergence_threshold=1.0e-6,
        pool_used_gb=None,
        status=status,
        error=None if status is SolveStatus.CONVERGED else "failed",
    )


def _report() -> BenchmarkResult:
    result = _solver_result()
    return BenchmarkResult(
        matrix=MatrixInfo(
            path="matrix.mtx",
            shape=(2, 2),
            nnz=4,
            density=1.0,
            is_square=True,
            symmetry_ratio=1.0,
            diagonal_sign="all_positive",
            spd_status="screen_passed",
            fingerprint="sha256:example",
        ),
        environment={"python": "3.11"},
        results=[result],
        recommendations={
            "end_to_end": Recommendation(
                mode="end-to-end",
                backend="scipy:cpu",
                reason="fastest",
            ),
            "steady_state": Recommendation(
                mode="steady-state",
                backend="scipy:cpu",
                reason="fastest",
            ),
        },
    )


def test_inspect_json_stdout_is_clean(
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = Path("tests/fixtures/small_symmetric.mtx")

    exit_code = main(["inspect", str(fixture), "--format", "json"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out)["shape"] == [3, 3]
    assert "Inspecting" in captured.err


def test_inspect_accepts_nonsquare_matrix_market_path(
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = Path("tests/fixtures/small_nonsquare.mtx")

    assert main(["inspect", str(fixture), "--quiet"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["shape"] == [2, 3]
    assert payload["is_square"] is False
    assert payload["spd_status"] == "failed"


@pytest.mark.parametrize("output_format", ["json", "csv", "table"])
def test_bench_formats_stay_on_stdout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    output_format: str,
) -> None:
    monkeypatch.setattr("sparsetune._cli.benchmark", lambda *_a, **_k: _report())

    exit_code = main(
        [
            "bench",
            "matrix.mtx",
            "--backends",
            "scipy",
            "--runs",
            "1",
            "--format",
            output_format,
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    if output_format == "json":
        assert json.loads(captured.out)["results"][0]["backend"] == "scipy:cpu"
    elif output_format == "csv":
        assert captured.out.startswith("backend,status")
    else:
        assert "scipy:cpu" in captured.out
    assert "Benchmarking" in captured.err


def test_bench_rejects_empty_backend_input_before_calling_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sparsetune._cli.benchmark",
        lambda *_args, **_kwargs: pytest.fail("benchmark must not run"),
    )

    with pytest.raises(SystemExit) as error:
        main(["bench", "matrix.mtx", "--backends", " , "])

    assert error.value.code == 2


def test_quiet_suppresses_progress_only(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sparsetune._cli.benchmark", lambda *_a, **_k: _report())

    assert main(["bench", "matrix.mtx", "--backends", "scipy", "--quiet"]) == 0

    captured = capsys.readouterr()
    assert json.loads(captured.out)["schema_version"] == "1.0"
    assert captured.err == ""


def test_tune_writes_profile_and_prints_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output = tmp_path / "profile.json"

    def fake_tune(
        _matrix: str,
        *,
        output: Path,
        **_kwargs: object,
    ) -> dict[str, object]:
        profile = {"schema_version": "1.0"}
        output.write_text(json.dumps(profile), encoding="utf-8")
        return profile

    monkeypatch.setattr("sparsetune._cli.tune", fake_tune)

    assert main(["tune", "matrix.mtx", "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["schema_version"] == "1.0"
    assert output.is_file()


def test_solve_selection_is_exclusive_and_failures_have_stable_exit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    failed = SolveResult(
        x=None,
        backend="scipy:cpu",
        dtype="float64",
        status=SolveStatus.MAX_ITER,
        iterations=10,
        residual_norm=1.0,
        relative_residual=1.0,
        convergence_threshold=1.0e-6,
        setup_seconds=0.0,
        solve_seconds=0.1,
        total_seconds=0.1,
        error="failed",
    )
    monkeypatch.setattr("sparsetune._cli.solve", lambda *_a, **_k: failed)

    with pytest.raises(SystemExit) as missing:
        main(["solve", "matrix.mtx"])
    with pytest.raises(SystemExit) as both:
        main(
            [
                "solve",
                "matrix.mtx",
                "--profile",
                "profile.json",
                "--backend",
                "scipy:cpu",
            ]
        )
    assert missing.value.code == 2
    assert both.value.code == 2

    assert main(["solve", "matrix.mtx", "--backend", "scipy:cpu"]) == 3
    assert json.loads(capsys.readouterr().out)["status"] == "max_iter"


def test_doctor_and_version_are_available(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "sparsetune._cli.environment_info",
        lambda: {"python": "3.11", "backends": ["scipy:cpu"]},
    )

    assert main(["doctor", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["backends"] == ["scipy:cpu"]

    with pytest.raises(SystemExit) as version:
        main(["--version"])
    assert version.value.code == 0
    assert capsys.readouterr().out == "0.1.0\n"


def test_runtime_input_error_goes_only_to_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "sparsetune._cli.diagnose_matrix",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad matrix")),
    )

    assert main(["inspect", "bad.mtx"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "bad matrix" in captured.err

import json
from pathlib import Path
from typing import get_args

import numpy as np
from scipy.sparse import spmatrix

from sparsetune import (
    BenchmarkResult,
    CanonicalMatrix,
    MatrixInfo,
    MatrixInput,
    Recommendation,
    RunSample,
    SolveResult,
    SolveStatus,
    SolverResult,
)


def test_solve_status_values_match_the_public_contract() -> None:
    assert [status.value for status in SolveStatus] == [
        "converged",
        "accuracy_failed",
        "max_iter",
        "breakdown",
        "nan_inf",
        "oom",
        "timeout",
        "process_crash",
        "unsupported",
        "internal_error",
    ]
    assert json.dumps(SolveStatus.CONVERGED) == '"converged"'


def test_run_sample_and_solver_result_are_json_serializable() -> None:
    sample = RunSample(
        measure="end-to-end",
        transfer_seconds=0.1,
        setup_seconds=0.2,
        solve_seconds=0.3,
        total_seconds=0.6,
        iterations=7,
        residual_norm=1.0e-8,
        relative_residual=1.0e-9,
        convergence_threshold=1.0e-6,
        status=SolveStatus.CONVERGED,
    )
    result = SolverResult(
        backend="scipy:cpu",
        solver_impl="scipy.sparse.linalg.cg",
        dtype="float64",
        transfer_seconds=0.1,
        setup_seconds=0.2,
        solve_seconds=0.3,
        total_seconds=0.6,
        iterations=7,
        residual_norm=1.0e-8,
        relative_residual=1.0e-9,
        convergence_threshold=1.0e-6,
        pool_used_gb=None,
        status=SolveStatus.CONVERGED,
        error=None,
        samples=[sample],
    )

    payload = json.loads(result.to_json())

    assert payload["status"] == "converged"
    assert payload["samples"][0]["measure"] == "end-to-end"
    assert payload["samples"][0]["status"] == "converged"


def test_solve_metrics_serialization_does_not_materialize_solution() -> None:
    class SolutionWithoutList(np.ndarray):
        def tolist(self) -> list[float]:
            raise AssertionError("metrics serialization must not materialize x")

    solution = np.arange(100_000, dtype=np.float64).view(SolutionWithoutList)
    result = SolveResult(
        x=solution,
        backend="scipy:cpu",
        dtype="float64",
        status=SolveStatus.CONVERGED,
        iterations=2,
        residual_norm=0.0,
        relative_residual=0.0,
        convergence_threshold=1.0e-6,
        setup_seconds=0.01,
        solve_seconds=0.02,
        total_seconds=0.03,
        error=None,
    )

    payload = result.to_metrics_dict()

    assert "x" not in payload
    assert payload["status"] == "converged"


def test_benchmark_result_serializes_and_finds_recommended_result() -> None:
    sample = RunSample(
        measure="end-to-end",
        transfer_seconds=0.0,
        setup_seconds=0.1,
        solve_seconds=0.2,
        total_seconds=0.3,
        iterations=3,
        residual_norm=0.0,
        relative_residual=0.0,
        convergence_threshold=1.0e-6,
        status=SolveStatus.CONVERGED,
    )
    result = SolverResult(
        backend="scipy:cpu",
        solver_impl="scipy.sparse.linalg.cg",
        dtype="float64",
        transfer_seconds=0.0,
        setup_seconds=0.1,
        solve_seconds=0.2,
        total_seconds=0.3,
        iterations=3,
        residual_norm=0.0,
        relative_residual=0.0,
        convergence_threshold=1.0e-6,
        pool_used_gb=None,
        status=SolveStatus.CONVERGED,
        error=None,
        samples=[sample],
    )
    report = BenchmarkResult(
        matrix=MatrixInfo(
            path="matrix.mtx",
            shape=(2, 2),
            nnz=2,
            density=0.5,
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
                reason="Only converged backend",
            )
        },
    )

    assert report.best() is result
    assert report.best("steady-state") is None
    payload = json.loads(report.to_json())
    assert payload == report.to_dict()
    assert set(payload) == {
        "schema_version",
        "matrix",
        "environment",
        "results",
        "recommendations",
    }
    assert payload["schema_version"] == "1.0"
    assert payload["matrix"]["dtype"] == "float64"
    assert payload["results"][0]["samples"] == [sample.to_dict()]
    assert payload["recommendations"]["end_to_end"] == {
        "mode": "end-to-end",
        "backend": "scipy:cpu",
        "reason": "Only converged backend",
        "speedup": None,
        "break_even_solves": None,
    }


def test_canonical_matrix_exposes_csr_metadata() -> None:
    matrix = CanonicalMatrix(
        data=np.array([1.0, 2.0], dtype=np.float32),
        indices=np.array([0, 1], dtype=np.int32),
        indptr=np.array([0, 1, 2], dtype=np.int32),
        shape=(2, 2),
        fingerprint="sha256:example",
    )

    assert matrix.dtype == "float32"
    assert matrix.nnz == 2
    assert {str, Path, CanonicalMatrix, spmatrix}.issubset(set(get_args(MatrixInput)))

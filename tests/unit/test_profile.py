from __future__ import annotations

import json
from pathlib import Path
import importlib
import warnings

import numpy as np
import pytest
from scipy.io import mmread
from scipy.sparse import csr_matrix

from sparsetune import (
    BenchmarkResult,
    ProfileMismatchError,
    Recommendation,
    SolveStatus,
    SolverResult,
    benchmark,
    load_profile,
    solve,
    tune,
)
from sparsetune._inspect import diagnose_matrix
from sparsetune._profile import validate_profile


def _solver_result(backend: str = "scipy:cpu") -> SolverResult:
    return SolverResult(
        backend=backend,
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
        status=SolveStatus.CONVERGED,
        error=None,
    )


def _report(matrix: csr_matrix) -> BenchmarkResult:
    info = diagnose_matrix(matrix)
    result = _solver_result()
    identity = {
        "backend": "scipy:cpu",
        "kind": "cpu",
        "scipy_version": "1.14",
        "cpu_model": "example",
        "cpu_cores_physical": 4,
        "blas_implementation": "OpenBLAS",
    }
    return BenchmarkResult(
        matrix=info,
        environment={
            "python": "3.11",
            "numpy": "2.0",
            "scipy": "1.14",
            "cpu_model": "example",
            "cpu_cores_physical": 4,
            "blas_implementation": "OpenBLAS",
            "backend_identity": {"scipy:cpu": identity},
        },
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


def _profile(matrix: csr_matrix) -> dict[str, object]:
    report = _report(matrix)
    return {
        "schema_version": "1.0",
        "matrix": {
            **report.matrix.to_dict(),
            "dtype": "float64",
        },
        "environment": report.environment,
        "config": {
            "dtype": "float64",
            "solver": "cg",
            "rtol": 1.0e-6,
            "atol": 0.0,
            "max_iter": 10_000,
        },
        "results": [result.to_dict() for result in report.results],
        "recommendations": {
            key: value.to_dict() for key, value in report.recommendations.items()
        },
        "backend_identity": {
            "scipy:cpu": {
                "backend": "scipy:cpu",
                "kind": "cpu",
                "scipy_version": "1.14",
                "cpu_model": "example",
                "cpu_cores_physical": 4,
                "blas_implementation": "OpenBLAS",
            }
        },
    }


def test_tune_save_load_round_trip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    matrix = csr_matrix([[4.0, 1.0], [1.0, 3.0]])
    monkeypatch.setattr(
        "sparsetune._profile.benchmark",
        lambda *_args, **_kwargs: _report(matrix),
    )
    output = tmp_path / "profile.json"

    profile = tune(matrix, output=output, backends=["scipy:cpu"], runs=2)

    assert profile == load_profile(output)
    assert profile["schema_version"] == "1.0"
    assert profile["config"]["runs"] == 2
    report = _report(matrix)
    for key, value in report.to_dict().items():
        assert profile[key] == value
    assert json.loads(output.read_text(encoding="utf-8")) == profile


def test_tune_uses_worker_identity_without_importing_cupy_in_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = csr_matrix([[4.0, 1.0], [1.0, 3.0]])
    report = _report(matrix)
    report.results = [_solver_result("cupy:cuda:0")]
    report.environment["backend_identity"] = {
        "cupy:cuda:0": {
            "kind": "cuda",
            "gpu_uuid": "gpu-uuid",
            "gpu_model": "GPU",
            "cuda_runtime": 12000,
            "cupy_version": "14.0",
        }
    }
    imported: list[str] = []
    real_import = importlib.import_module

    def poison_cupy(name: str) -> object:
        if name == "cupy":
            imported.append(name)
            raise AssertionError("CuPy must not be imported in the parent")
        return real_import(name)

    monkeypatch.setattr("sparsetune._profile.benchmark", lambda *_a, **_k: report)
    monkeypatch.setattr(
        "sparsetune._benchmark.importlib.import_module",
        poison_cupy,
    )

    profile = tune(matrix, backends=["cupy:cuda:0"], runs=1)

    assert profile["backend_identity"] == report.environment["backend_identity"]
    assert imported == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "2.0", "schema"),
        ("fingerprint", "sha256:wrong", "fingerprint"),
        ("dtype", "float32", "dtype"),
        ("solver", "gmres", "solver"),
        ("rtol", 1.0e-4, "rtol"),
    ],
)
def test_validate_profile_rejects_identity_and_config_mismatches(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    matrix = csr_matrix([[4.0, 1.0], [1.0, 3.0]])
    profile = _profile(matrix)
    if field == "schema_version":
        profile[field] = value
    elif field == "fingerprint":
        profile["matrix"][field] = value
    elif field == "dtype":
        profile["matrix"][field] = value
    else:
        profile["config"][field] = value
    monkeypatch.setattr(
        "sparsetune._profile._current_environment",
        lambda: profile["environment"],
    )

    with pytest.raises(ProfileMismatchError, match=message):
        validate_profile(
            profile,
            diagnose_matrix(matrix),
            dtype="float64",
            solver="cg",
            rtol=1.0e-6,
            atol=0.0,
            max_iter=10_000,
        )


def test_validate_profile_requires_opt_in_for_stale_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = csr_matrix([[4.0, 1.0], [1.0, 3.0]])
    profile = _profile(matrix)
    monkeypatch.setattr(
        "sparsetune._profile._current_environment",
        lambda: {**profile["environment"], "scipy": "9.9"},
    )

    with pytest.raises(ProfileMismatchError, match="stale"):
        validate_profile(profile, diagnose_matrix(matrix))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_profile(
            profile,
            diagnose_matrix(matrix),
            allow_stale_profile=True,
        )
    assert "scipy" in str(caught[0].message)


def test_validate_profile_rejects_gpu_uuid_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = csr_matrix([[4.0, 1.0], [1.0, 3.0]])
    profile = _profile(matrix)
    profile["backend_identity"] = {"cupy:cuda:0": {"gpu_uuid": "old"}}
    monkeypatch.setattr(
        "sparsetune._profile._runtime_identity",
        lambda _backend: {"gpu_uuid": "new"},
    )
    monkeypatch.setattr(
        "sparsetune._profile._current_environment",
        lambda: profile["environment"],
    )

    with pytest.raises(ProfileMismatchError, match="GPU UUID"):
        validate_profile(
            profile,
            diagnose_matrix(matrix),
            backend="cupy:cuda:0",
        )


def test_validate_profile_rejects_critical_backend_identity_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = csr_matrix([[4.0, 1.0], [1.0, 3.0]])
    profile = _profile(matrix)
    monkeypatch.setattr(
        "sparsetune._profile._runtime_identity",
        lambda _backend: {
            **profile["backend_identity"]["scipy:cpu"],
            "blas_implementation": "MKL",
        },
    )
    monkeypatch.setattr(
        "sparsetune._profile._current_environment",
        lambda: profile["environment"],
    )

    with pytest.raises(ProfileMismatchError, match="backend identity"):
        validate_profile(
            profile,
            diagnose_matrix(matrix),
            backend="scipy:cpu",
        )


def test_validate_profile_treats_missing_stale_field_deterministically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = csr_matrix([[4.0, 1.0], [1.0, 3.0]])
    profile = _profile(matrix)
    del profile["environment"]["blas_implementation"]
    monkeypatch.setattr(
        "sparsetune._profile._current_environment",
        lambda: {
            **profile["environment"],
            "blas_implementation": "OpenBLAS",
        },
    )

    with pytest.raises(ProfileMismatchError, match="stale"):
        validate_profile(profile, diagnose_matrix(matrix))


def test_solve_profile_selects_requested_mode_without_benchmarking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = csr_matrix([[4.0, 1.0], [1.0, 3.0]])
    profile = _profile(matrix)
    selected: list[str] = []

    monkeypatch.setattr(
        "sparsetune._profile._current_environment",
        lambda: profile["environment"],
    )
    monkeypatch.setattr(
        "sparsetune._profile._runtime_identity",
        lambda backend: profile["backend_identity"][backend],
    )

    def fake_run(
        backend: str,
        *_args: object,
        **_kwargs: object,
    ) -> tuple[SolverResult, np.ndarray]:
        selected.append(backend)
        return _solver_result(backend), np.asarray([1.0, 1.0])

    monkeypatch.setattr(
        "sparsetune._profile.run_solve_in_subprocess",
        fake_run,
    )
    monkeypatch.setattr(
        "sparsetune._profile.benchmark",
        lambda *_args, **_kwargs: pytest.fail("benchmark must not run"),
    )

    result = solve(matrix, profile=profile, selection_mode="steady-state")

    assert selected == ["scipy:cpu"]
    assert result.status is SolveStatus.CONVERGED
    np.testing.assert_array_equal(result.x, [1.0, 1.0])


def test_scipy_profile_solve_does_not_probe_cupy_in_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = csr_matrix([[4.0, 1.0], [1.0, 3.0]])
    profile = _profile(matrix)
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
    monkeypatch.setattr(
        "sparsetune._profile.run_solve_in_subprocess",
        lambda *_a, **_k: (
            _solver_result("scipy:cpu"),
            np.asarray([1.0, 1.0]),
        ),
    )
    monkeypatch.setattr(
        "sparsetune._profile._runtime_identity",
        lambda backend: profile["backend_identity"][backend],
    )

    result = solve(matrix, profile=profile, allow_stale_profile=True)

    assert result.status is SolveStatus.CONVERGED
    assert imported == []


def test_solve_requires_exactly_one_backend_selection() -> None:
    matrix = csr_matrix([[4.0, 1.0], [1.0, 3.0]])

    with pytest.raises(ValueError, match="exactly one"):
        solve(matrix)
    with pytest.raises(ValueError, match="exactly one"):
        solve(matrix, profile=_profile(matrix), backend="scipy:cpu")


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"dtype": "float16"}, "dtype"),
        ({"rtol": -1.0}, "rtol"),
        ({"rtol": np.nan}, "rtol"),
        ({"atol": -1.0}, "atol"),
        ({"atol": np.inf}, "atol"),
        ({"max_iter": 0}, "max_iter"),
        ({"max_iter": 1.5}, "max_iter"),
        ({"max_iter": True}, "max_iter"),
        ({"timeout": 0.0}, "timeout"),
        ({"timeout": np.inf}, "timeout"),
    ],
)
def test_solve_rejects_invalid_options_before_processing(
    monkeypatch: pytest.MonkeyPatch,
    options: dict[str, object],
    message: str,
) -> None:
    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail(
            "invalid solve options must fail before matrix or worker processing"
        )

    monkeypatch.setattr("sparsetune._profile.diagnose_matrix", unexpected)
    monkeypatch.setattr("sparsetune._profile.run_solve_in_subprocess", unexpected)

    with pytest.raises(ValueError, match=message):
        solve(
            object(),  # type: ignore[arg-type]
            backend="scipy:cpu",
            **options,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dtype", None),
        ("dtype", "float16"),
        ("rtol", None),
        ("rtol", "invalid"),
        ("rtol", -1.0),
        ("rtol", np.nan),
        ("atol", []),
        ("atol", np.inf),
        ("max_iter", 0),
        ("max_iter", 1.5),
        ("max_iter", True),
    ],
)
def test_solve_rejects_malformed_profile_config_before_processing(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    matrix = csr_matrix([[4.0, 1.0], [1.0, 3.0]])
    profile = _profile(matrix)
    profile["config"][field] = value
    monkeypatch.setattr(
        "sparsetune._profile.diagnose_matrix",
        lambda *_args, **_kwargs: pytest.fail(
            "malformed profile must fail before matrix processing"
        ),
    )

    with pytest.raises(ProfileMismatchError, match="configuration"):
        solve(object(), profile=profile)  # type: ignore[arg-type]


def test_solve_requires_all_profile_config_fields() -> None:
    matrix = csr_matrix([[4.0, 1.0], [1.0, 3.0]])
    profile = _profile(matrix)
    del profile["config"]["rtol"]

    with pytest.raises(ProfileMismatchError, match="configuration"):
        solve(matrix, profile=profile)


@pytest.mark.parametrize(
    "options",
    [
        {"dtype": "float16"},
        {"rtol": -1.0},
        {"atol": np.inf},
        {"max_iter": 0},
        {"max_iter": 1.5},
        {"timeout": 0.0},
    ],
)
def test_benchmark_and_solve_option_validation_match(
    options: dict[str, object],
) -> None:
    with pytest.raises(ValueError) as benchmark_error:
        benchmark(
            object(),  # type: ignore[arg-type]
            backends=["scipy:cpu"],
            **options,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError) as solve_error:
        solve(
            object(),  # type: ignore[arg-type]
            backend="scipy:cpu",
            **options,  # type: ignore[arg-type]
        )

    assert str(solve_error.value) == str(benchmark_error.value)


def test_explicit_scipy_solve_writes_matrix_market_solution(tmp_path: Path) -> None:
    matrix = csr_matrix([[4.0, 1.0], [1.0, 3.0]])
    output = tmp_path / "solution.mtx"

    result = solve(matrix, backend="scipy:cpu", output=output)

    assert result.status is SolveStatus.CONVERGED
    np.testing.assert_allclose(result.x, [1.0, 1.0])
    np.testing.assert_allclose(np.asarray(mmread(output)).reshape(-1), [1.0, 1.0])


def test_explicit_scipy_solve_accepts_zero_rhs() -> None:
    matrix = csr_matrix([[4.0, 1.0], [1.0, 3.0]])

    result = solve(
        matrix,
        rhs=np.zeros(2),
        backend="scipy:cpu",
        atol=0.0,
    )

    assert result.status is SolveStatus.CONVERGED
    assert result.relative_residual is None
    assert result.convergence_threshold == 0.0
    np.testing.assert_array_equal(result.x, [0.0, 0.0])


def test_real_tune_save_load_solve_round_trip(tmp_path: Path) -> None:
    matrix = csr_matrix([[4.0, 1.0], [1.0, 3.0]])
    profile_path = tmp_path / "profile.json"

    tune(
        matrix,
        output=profile_path,
        backends=["scipy:cpu"],
        runs=1,
    )
    result = solve(matrix, profile=profile_path)

    assert result.backend == "scipy:cpu"
    assert result.status is SolveStatus.CONVERGED
    np.testing.assert_allclose(result.x, [1.0, 1.0])

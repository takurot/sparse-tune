"""Profile persistence, compatibility validation, and selected solves."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any, Mapping, cast
import warnings

import numpy as np
from scipy.io import mmwrite  # type: ignore[import-untyped]

from ._benchmark import _environment, _probe_backend, benchmark
from ._inspect import diagnose_matrix, is_cg_eligible
from ._matrix import canonicalize_matrix
from ._runner import canonicalize_rhs, run_solve_in_subprocess
from ._types import (
    MatrixInfo,
    MatrixInput,
    Profile,
    SolveResult,
    SolverResult,
)


_SCHEMA_VERSION = "1.0"
_STALE_ENVIRONMENT_FIELDS = (
    "python",
    "numpy",
    "scipy",
    "cpu_model",
    "blas_implementation",
    "cuda_driver",
    "cupy_version",
)


class ProfileMismatchError(ValueError):
    """Raised when a cached profile cannot safely select a backend."""


def _current_environment() -> dict[str, Any]:
    return _environment()


def _runtime_identity(backend_id: str) -> dict[str, Any]:
    if backend_id == "scipy:cpu":
        return {"kind": "cpu"}
    if not backend_id.startswith("cupy:cuda:"):
        raise ProfileMismatchError(f"Backend is unavailable: {backend_id}")
    error, identity = _probe_backend(backend_id, "float64")
    if error is not None or identity is None:
        raise ProfileMismatchError(f"Backend is unavailable: {backend_id}")
    return identity


def load_profile(profile: str | Path | Mapping[str, Any]) -> Profile:
    """Load a profile mapping or JSON file without applying it."""

    if isinstance(profile, (str, Path)):
        try:
            payload = json.loads(Path(profile).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ProfileMismatchError("Profile is not valid JSON") from error
    else:
        payload = dict(profile)
    if not isinstance(payload, dict):
        raise ProfileMismatchError("Profile root must be an object")
    return payload


def _profile_backend(profile: Profile, selection_mode: str) -> str:
    key = selection_mode.replace("-", "_")
    if key not in {"end_to_end", "steady_state"}:
        raise ValueError("selection_mode must be end-to-end or steady-state")
    recommendations = profile.get("recommendations")
    if not isinstance(recommendations, dict):
        raise ProfileMismatchError("Profile recommendations are missing")
    recommendation = recommendations.get(key)
    if not isinstance(recommendation, dict):
        raise ProfileMismatchError(f"Profile has no {selection_mode} recommendation")
    backend = recommendation.get("backend")
    if not isinstance(backend, str) or not backend:
        raise ProfileMismatchError(f"Profile has no {selection_mode} backend")
    return backend


def validate_profile(
    profile: Profile,
    matrix_info: MatrixInfo,
    *,
    backend: str | None = None,
    dtype: str = "float64",
    solver: str = "cg",
    rtol: float = 1.0e-6,
    atol: float = 0.0,
    max_iter: int = 10_000,
    allow_stale_profile: bool = False,
) -> None:
    """Validate hard identity/configuration and stale environment fields."""

    if profile.get("schema_version") != _SCHEMA_VERSION:
        raise ProfileMismatchError("Profile schema version is not supported")
    matrix_payload = profile.get("matrix")
    config = profile.get("config")
    if not isinstance(matrix_payload, dict) or not isinstance(config, dict):
        raise ProfileMismatchError("Profile matrix or configuration is missing")
    if matrix_payload.get("fingerprint") != matrix_info.fingerprint:
        raise ProfileMismatchError("Matrix fingerprint mismatch")
    if matrix_payload.get("dtype") != dtype or config.get("dtype") != dtype:
        raise ProfileMismatchError("Profile dtype mismatch")
    expected_config = {
        "solver": solver,
        "rtol": rtol,
        "atol": atol,
        "max_iter": max_iter,
    }
    for name, expected in expected_config.items():
        if config.get(name) != expected:
            raise ProfileMismatchError(f"Profile {name} mismatch")

    if backend is not None:
        identities = profile.get("backend_identity")
        if not isinstance(identities, dict) or backend not in identities:
            raise ProfileMismatchError(f"Profile backend identity missing: {backend}")
        stored_identity = identities[backend]
        if not isinstance(stored_identity, dict):
            raise ProfileMismatchError(f"Profile backend identity invalid: {backend}")
        current_identity = _runtime_identity(backend)
        if backend.startswith("cupy:") and stored_identity.get(
            "gpu_uuid"
        ) != current_identity.get("gpu_uuid"):
            raise ProfileMismatchError("Profile GPU UUID mismatch")

    stored_environment = profile.get("environment")
    if not isinstance(stored_environment, dict):
        raise ProfileMismatchError("Profile environment is missing")
    current_environment = _current_environment()
    changes = [
        name
        for name in _STALE_ENVIRONMENT_FIELDS
        if name in stored_environment
        and name in current_environment
        and stored_environment[name] != current_environment[name]
    ]
    if changes and not allow_stale_profile:
        raise ProfileMismatchError(
            "Profile is stale; pass allow_stale_profile=True to use it"
        )
    if changes:
        warnings.warn(
            "Using stale profile after environment changes: " + ", ".join(changes),
            UserWarning,
            stacklevel=2,
        )


def tune(
    matrix: MatrixInput,
    *,
    output: str | Path | None = None,
    **benchmark_options: Any,
) -> Profile:
    """Benchmark a matrix and optionally persist a reusable profile."""

    report = benchmark(matrix, **benchmark_options)
    dtype = str(benchmark_options.get("dtype", "float64"))
    config = {
        "dtype": dtype,
        "solver": "cg",
        "rtol": float(benchmark_options.get("rtol", 1.0e-6)),
        "atol": float(benchmark_options.get("atol", 0.0)),
        "max_iter": int(benchmark_options.get("max_iter", 10_000)),
        "runs": int(benchmark_options.get("runs", 5)),
        "measure": list(
            benchmark_options.get(
                "measure",
                ("end-to-end", "steady-state"),
            )
        ),
        "timeout": float(benchmark_options.get("timeout", 300.0)),
    }
    reported_identity = report.environment.get("backend_identity", {})
    if not isinstance(reported_identity, dict):
        reported_identity = {}
    backend_identity = {}
    for result in report.results:
        if result.error is not None:
            continue
        identity = reported_identity.get(result.backend)
        if isinstance(identity, dict):
            backend_identity[result.backend] = identity
        elif result.backend == "scipy:cpu":
            backend_identity[result.backend] = {"kind": "cpu"}
        else:
            raise ProfileMismatchError(f"Backend identity is missing: {result.backend}")
    profile: Profile = {
        "schema_version": _SCHEMA_VERSION,
        "matrix": {**report.matrix.to_dict(), "dtype": dtype},
        "environment": report.environment,
        "config": config,
        "results": [result.to_dict() for result in report.results],
        "recommendations": {
            key: recommendation.to_dict()
            for key, recommendation in report.recommendations.items()
        },
        "backend_identity": backend_identity,
    }
    if output is not None:
        Path(output).write_text(
            json.dumps(profile, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return profile


def _as_solve_result(
    result: SolverResult,
    solution: np.ndarray[Any, Any] | None,
) -> SolveResult:
    return SolveResult(
        x=solution,
        backend=result.backend,
        dtype=result.dtype,
        status=result.status,
        iterations=result.iterations,
        residual_norm=result.residual_norm,
        relative_residual=result.relative_residual,
        convergence_threshold=result.convergence_threshold,
        setup_seconds=result.setup_seconds,
        solve_seconds=result.solve_seconds,
        total_seconds=result.total_seconds,
        error=result.error,
    )


def solve(
    matrix: MatrixInput,
    rhs: str | Path | np.ndarray[Any, Any] | None = None,
    *,
    profile: str | Path | Mapping[str, Any] | None = None,
    backend: str | None = None,
    selection_mode: str = "end-to-end",
    dtype: str | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    max_iter: int | None = None,
    timeout: float = 300.0,
    assume_spd: bool = False,
    allow_stale_profile: bool = False,
    output: str | Path | None = None,
) -> SolveResult:
    """Solve with exactly one explicit or profile-selected backend."""

    if (profile is None) == (backend is None):
        raise ValueError("exactly one of profile or backend is required")

    loaded_profile: Profile | None = None
    if profile is not None:
        loaded_profile = load_profile(profile)
        backend = _profile_backend(loaded_profile, selection_mode)
        profile_config = loaded_profile.get("config")
        if not isinstance(profile_config, dict):
            raise ProfileMismatchError("Profile configuration is missing")
        selected_dtype = str(profile_config.get("dtype"))
        selected_rtol = float(cast(Any, profile_config.get("rtol")))
        selected_atol = float(cast(Any, profile_config.get("atol")))
        selected_max_iter = int(cast(Any, profile_config.get("max_iter")))
        if dtype is not None and dtype != selected_dtype:
            raise ProfileMismatchError("Profile dtype mismatch")
        if rtol is not None and rtol != selected_rtol:
            raise ProfileMismatchError("Profile rtol mismatch")
        if atol is not None and atol != selected_atol:
            raise ProfileMismatchError("Profile atol mismatch")
        if max_iter is not None and max_iter != selected_max_iter:
            raise ProfileMismatchError("Profile max_iter mismatch")
    else:
        selected_dtype = dtype or "float64"
        selected_rtol = 1.0e-6 if rtol is None else rtol
        selected_atol = 0.0 if atol is None else atol
        selected_max_iter = 10_000 if max_iter is None else max_iter

    matrix_info = diagnose_matrix(matrix)
    if not is_cg_eligible(matrix_info, assume_spd=assume_spd):
        if matrix_info.spd_status == "unknown":
            raise ValueError("matrix requires assume_spd=True for CG")
        raise ValueError("matrix is not eligible for CG")
    if loaded_profile is not None:
        validate_profile(
            loaded_profile,
            matrix_info,
            backend=backend,
            dtype=selected_dtype,
            solver="cg",
            rtol=selected_rtol,
            atol=selected_atol,
            max_iter=selected_max_iter,
            allow_stale_profile=allow_stale_profile,
        )

    assert backend is not None
    with tempfile.TemporaryDirectory(prefix="sparsetune-solve-") as temporary:
        work_dir = Path(temporary)
        matrix_path, canonical = canonicalize_matrix(
            matrix,
            selected_dtype,
            work_dir,
        )
        rhs_path, _ = canonicalize_rhs(
            rhs,
            canonical,
            dtype_str=selected_dtype,
            work_dir=work_dir,
        )
        result, solution = run_solve_in_subprocess(
            backend,
            matrix_path,
            rhs_path,
            {
                "dtype": selected_dtype,
                "rtol": selected_rtol,
                "atol": selected_atol,
                "max_iter": selected_max_iter,
            },
            timeout=timeout,
            expected_size=canonical.shape[0],
        )

    solve_result = _as_solve_result(result, solution)
    if output is not None:
        if solve_result.x is None:
            raise ValueError("cannot write a solution for an unsuccessful solve")
        mmwrite(Path(output), solve_result.x.reshape(-1, 1))
    return solve_result

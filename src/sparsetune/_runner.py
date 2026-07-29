"""Parent-side RHS normalization and isolated worker execution."""

from __future__ import annotations

from dataclasses import fields
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.io import mmread  # type: ignore[import-untyped]
from scipy.sparse import csr_matrix, issparse  # type: ignore[import-untyped]

from ._types import (
    CanonicalMatrix,
    RunSample,
    SolveStatus,
    SolverResult,
)


_DIAGNOSTIC_LIMIT = 500
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b([a-z0-9_-]*(?:api[_-]?key|token|secret|password|authorization)"
    r"[a-z0-9_-]*)\b"
    r"\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|\S+)"
)
_BEARER_TOKEN = re.compile(r"(?i)\bbearer\s+\S+")
_SUPPORTED_DTYPES = {"float32", "float64"}


def _as_csr(matrix: CanonicalMatrix) -> Any:
    return csr_matrix(
        (matrix.data, matrix.indices, matrix.indptr),
        shape=matrix.shape,
        copy=False,
    )


def canonicalize_rhs(
    rhs: str | Path | NDArray[Any] | None,
    matrix: CanonicalMatrix,
    *,
    dtype_str: str,
    work_dir: str | Path,
) -> tuple[Path, NDArray[np.floating[Any]]]:
    """Normalize one RHS vector and save it in worker-ready NPY format."""

    if dtype_str not in _SUPPORTED_DTYPES:
        raise ValueError("dtype must be 'float32' or 'float64'")
    dtype = np.dtype(dtype_str)

    if rhs is None:
        ones = np.ones(matrix.shape[1], dtype=dtype)
        values = np.asarray(_as_csr(matrix) @ ones, dtype=dtype)
    elif isinstance(rhs, (str, Path)):
        loaded = mmread(Path(rhs))
        if issparse(loaded):
            loaded = loaded.toarray()
        if np.iscomplexobj(loaded):
            raise ValueError("complex RHS is not supported")
        values = np.asarray(loaded, dtype=dtype)
    else:
        if np.iscomplexobj(rhs):
            raise ValueError("complex RHS is not supported")
        values = np.asarray(rhs, dtype=dtype)

    if values.ndim == 2 and 1 in values.shape:
        values = values.reshape(-1)
    elif values.ndim != 1:
        raise ValueError("RHS must contain a single vector")
    if values.shape[0] != matrix.shape[0]:
        raise ValueError(
            f"RHS size {values.shape[0]} does not match matrix size {matrix.shape[0]}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("RHS values must be finite")

    output_dir = Path(work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rhs_path = output_dir / f"sparsetune_rhs_{dtype_str}.npy"
    normalized = np.asarray(values, dtype=dtype).copy()
    np.save(rhs_path, normalized, allow_pickle=False)
    return rhs_path, normalized


def classify_solve(
    info: int,
    solution: NDArray[Any],
    residual_norm: float,
    threshold: float,
) -> SolveStatus:
    """Apply the documented finite/info/residual classification order."""

    if not np.all(np.isfinite(solution)):
        return SolveStatus.NAN_INF
    if info == 0:
        if residual_norm <= threshold:
            return SolveStatus.CONVERGED
        return SolveStatus.ACCURACY_FAILED
    if info > 0:
        return SolveStatus.MAX_ITER
    return SolveStatus.BREAKDOWN


def _sanitize_diagnostic(diagnostic: str) -> str:
    sanitized = _SECRET_ASSIGNMENT.sub(
        lambda match: f"{match.group(1)}=[REDACTED]",
        diagnostic,
    )
    sanitized = _BEARER_TOKEN.sub("Bearer [REDACTED]", sanitized)
    return sanitized[:_DIAGNOSTIC_LIMIT]


def _error_result(
    backend_id: str,
    status: SolveStatus,
    error: str,
) -> SolverResult:
    return SolverResult(
        backend=backend_id,
        solver_impl="",
        dtype="",
        transfer_seconds=0.0,
        setup_seconds=0.0,
        solve_seconds=0.0,
        total_seconds=0.0,
        iterations=0,
        residual_norm=0.0,
        relative_residual=None,
        convergence_threshold=0.0,
        pool_used_gb=None,
        status=status,
        error=error,
    )


def _is_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _is_non_negative_number(value: Any) -> bool:
    return _is_number(value) and float(value) >= 0.0


def _parse_sample(payload: Any) -> RunSample:
    if not isinstance(payload, dict):
        raise ValueError
    expected = {item.name for item in fields(RunSample)}
    if set(payload) != expected:
        raise ValueError
    if payload["measure"] not in {"end-to-end", "steady-state"}:
        raise ValueError
    for name in (
        "transfer_seconds",
        "setup_seconds",
        "solve_seconds",
        "total_seconds",
        "residual_norm",
        "convergence_threshold",
    ):
        if not _is_non_negative_number(payload[name]):
            raise ValueError
    if (
        not isinstance(payload["iterations"], int)
        or isinstance(payload["iterations"], bool)
        or payload["iterations"] < 0
    ):
        raise ValueError
    relative = payload["relative_residual"]
    pool_used = payload["pool_used_gb"]
    if relative is not None and not _is_non_negative_number(relative):
        raise ValueError
    if pool_used is not None and not _is_non_negative_number(pool_used):
        raise ValueError
    if payload["error"] is not None and not isinstance(payload["error"], str):
        raise ValueError
    sample = RunSample(
        measure=payload["measure"],
        transfer_seconds=float(payload["transfer_seconds"]),
        setup_seconds=float(payload["setup_seconds"]),
        solve_seconds=float(payload["solve_seconds"]),
        total_seconds=float(payload["total_seconds"]),
        iterations=payload["iterations"],
        residual_norm=float(payload["residual_norm"]),
        relative_residual=(None if relative is None else float(relative)),
        convergence_threshold=float(payload["convergence_threshold"]),
        status=SolveStatus(payload["status"]),
        error=payload["error"],
        pool_used_gb=None if pool_used is None else float(pool_used),
    )
    if sample.total_seconds < (
        sample.setup_seconds + sample.solve_seconds + sample.transfer_seconds
    ):
        raise ValueError
    if sample.measure == "steady-state" and sample.setup_seconds != 0.0:
        raise ValueError
    if sample.status is SolveStatus.CONVERGED and sample.error is not None:
        raise ValueError
    return sample


def _median_sample(samples: list[RunSample], field: str) -> RunSample:
    target = statistics.median(float(getattr(sample, field)) for sample in samples)
    return min(samples, key=lambda sample: abs(float(getattr(sample, field)) - target))


def _validate_aggregate(result: SolverResult) -> None:
    if not result.samples:
        if (
            result.status
            not in {
                SolveStatus.OOM,
                SolveStatus.UNSUPPORTED,
                SolveStatus.INTERNAL_ERROR,
            }
            or not result.error
            or result.solver_impl
            or (
                result.transfer_seconds,
                result.setup_seconds,
                result.solve_seconds,
                result.total_seconds,
                result.iterations,
                result.residual_norm,
                result.convergence_threshold,
            )
            != (0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)
            or result.relative_residual is not None
            or result.pool_used_gb is not None
        ):
            raise ValueError
        return

    end_to_end = [sample for sample in result.samples if sample.measure == "end-to-end"]
    steady_state = [
        sample for sample in result.samples if sample.measure == "steady-state"
    ]
    primary = _median_sample(end_to_end or steady_state, "total_seconds")
    solve_sample = _median_sample(steady_state or end_to_end, "solve_seconds")
    failure = next(
        (
            sample
            for sample in result.samples
            if sample.status is not SolveStatus.CONVERGED
        ),
        None,
    )
    accuracy_sample = failure or primary
    if (
        not result.solver_impl
        or (
            result.transfer_seconds,
            result.setup_seconds,
            result.total_seconds,
        )
        != (
            primary.transfer_seconds,
            primary.setup_seconds,
            primary.total_seconds,
        )
        or result.solve_seconds != solve_sample.solve_seconds
        or (
            result.iterations,
            result.residual_norm,
            result.relative_residual,
            result.convergence_threshold,
            result.pool_used_gb,
            result.status,
            result.error,
        )
        != (
            accuracy_sample.iterations,
            accuracy_sample.residual_norm,
            accuracy_sample.relative_residual,
            accuracy_sample.convergence_threshold,
            accuracy_sample.pool_used_gb,
            accuracy_sample.status,
            accuracy_sample.error,
        )
    ):
        raise ValueError


def _parse_result(
    payload: Any,
    backend_id: str,
    *,
    expected_dtype: str | None = None,
    expected_measures: list[str] | None = None,
    expected_runs: int | None = None,
) -> SolverResult:
    if not isinstance(payload, dict):
        raise ValueError
    expected = {item.name for item in fields(SolverResult)}
    if set(payload) != expected or payload.get("backend") != backend_id:
        raise ValueError
    for name in ("backend", "solver_impl", "dtype"):
        if not isinstance(payload[name], str):
            raise ValueError
    if payload["dtype"] not in _SUPPORTED_DTYPES:
        raise ValueError
    if expected_dtype is not None and payload["dtype"] != expected_dtype:
        raise ValueError
    for name in (
        "transfer_seconds",
        "setup_seconds",
        "solve_seconds",
        "total_seconds",
        "residual_norm",
        "convergence_threshold",
    ):
        if not _is_non_negative_number(payload[name]):
            raise ValueError
    if (
        not isinstance(payload["iterations"], int)
        or isinstance(payload["iterations"], bool)
        or payload["iterations"] < 0
    ):
        raise ValueError
    relative = payload["relative_residual"]
    pool_used = payload["pool_used_gb"]
    if relative is not None and not _is_non_negative_number(relative):
        raise ValueError
    if pool_used is not None and not _is_non_negative_number(pool_used):
        raise ValueError
    if payload["error"] is not None and not isinstance(payload["error"], str):
        raise ValueError
    if not isinstance(payload["samples"], list):
        raise ValueError

    result = SolverResult(
        backend=payload["backend"],
        solver_impl=payload["solver_impl"],
        dtype=payload["dtype"],
        transfer_seconds=float(payload["transfer_seconds"]),
        setup_seconds=float(payload["setup_seconds"]),
        solve_seconds=float(payload["solve_seconds"]),
        total_seconds=float(payload["total_seconds"]),
        iterations=payload["iterations"],
        residual_norm=float(payload["residual_norm"]),
        relative_residual=None if relative is None else float(relative),
        convergence_threshold=float(payload["convergence_threshold"]),
        pool_used_gb=None if pool_used is None else float(pool_used),
        status=SolveStatus(payload["status"]),
        error=payload["error"],
        samples=[_parse_sample(sample) for sample in payload["samples"]],
    )
    if result.samples and expected_measures is not None and expected_runs is not None:
        if any(
            sum(sample.measure == measure for sample in result.samples) != expected_runs
            for measure in expected_measures
        ) or any(sample.measure not in expected_measures for sample in result.samples):
            raise ValueError
    _validate_aggregate(result)
    return result


def run_backend_in_subprocess(
    backend_id: str,
    canonical_npz_path: str | Path,
    rhs_path: str | Path,
    config: dict[str, Any],
    *,
    timeout: float,
) -> SolverResult:
    """Run one backend worker and validate its result before use."""

    with tempfile.TemporaryDirectory(prefix="sparsetune-") as temporary:
        temporary_path = Path(temporary)
        config_path = temporary_path / "config.json"
        result_path = temporary_path / "result.json"
        try:
            config_path.write_text(json.dumps(config), encoding="utf-8")
        except (OSError, TypeError, ValueError):
            return _error_result(
                backend_id,
                SolveStatus.INTERNAL_ERROR,
                "Unable to serialize worker configuration",
            )

        command = [
            sys.executable,
            "-m",
            "sparsetune._worker",
            "--backend",
            backend_id,
            "--matrix-npz",
            str(canonical_npz_path),
            "--rhs",
            str(rhs_path),
            "--config",
            str(config_path),
            "--result",
            str(result_path),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return _error_result(
                backend_id,
                SolveStatus.TIMEOUT,
                f"Timed out after {timeout} seconds",
            )

        if completed.returncode != 0:
            diagnostic = _sanitize_diagnostic(completed.stderr)
            return _error_result(
                backend_id,
                SolveStatus.PROCESS_CRASH,
                diagnostic or "Worker exited without a diagnostic",
            )

        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            return _parse_result(
                payload,
                backend_id,
                expected_dtype=config.get("dtype"),
                expected_measures=config.get("measure"),
                expected_runs=config.get("runs"),
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
            return _error_result(
                backend_id,
                SolveStatus.PROCESS_CRASH,
                "Worker returned a malformed result",
            )


def run_solve_in_subprocess(
    backend_id: str,
    canonical_npz_path: str | Path,
    rhs_path: str | Path,
    config: dict[str, Any],
    *,
    timeout: float,
    expected_size: int,
) -> tuple[SolverResult, NDArray[np.floating[Any]] | None]:
    """Run one isolated solve and validate its result and solution vector."""

    with tempfile.TemporaryDirectory(prefix="sparsetune-solve-") as temporary:
        temporary_path = Path(temporary)
        config_path = temporary_path / "config.json"
        result_path = temporary_path / "result.json"
        solution_path = temporary_path / "solution.npy"
        try:
            config_path.write_text(json.dumps(config), encoding="utf-8")
        except (OSError, TypeError, ValueError):
            return (
                _error_result(
                    backend_id,
                    SolveStatus.INTERNAL_ERROR,
                    "Unable to serialize worker configuration",
                ),
                None,
            )

        command = [
            sys.executable,
            "-m",
            "sparsetune._solve_worker",
            "--backend",
            backend_id,
            "--matrix-npz",
            str(canonical_npz_path),
            "--rhs",
            str(rhs_path),
            "--config",
            str(config_path),
            "--result",
            str(result_path),
            "--solution",
            str(solution_path),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return (
                _error_result(
                    backend_id,
                    SolveStatus.TIMEOUT,
                    f"Timed out after {timeout} seconds",
                ),
                None,
            )

        if completed.returncode != 0:
            diagnostic = _sanitize_diagnostic(completed.stderr)
            return (
                _error_result(
                    backend_id,
                    SolveStatus.PROCESS_CRASH,
                    diagnostic or "Worker exited without a diagnostic",
                ),
                None,
            )

        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            result = _parse_result(
                payload,
                backend_id,
                expected_dtype=config.get("dtype"),
                expected_measures=["end-to-end"],
                expected_runs=1,
            )
            if not solution_path.is_file():
                if result.status in {
                    SolveStatus.OOM,
                    SolveStatus.UNSUPPORTED,
                    SolveStatus.INTERNAL_ERROR,
                }:
                    return result, None
                raise ValueError
            if result.status in {
                SolveStatus.OOM,
                SolveStatus.UNSUPPORTED,
                SolveStatus.INTERNAL_ERROR,
            }:
                raise ValueError
            solution = np.load(solution_path, allow_pickle=False)
            if (
                solution.ndim != 1
                or solution.shape[0] != expected_size
                or solution.dtype.name not in _SUPPORTED_DTYPES
                or not np.all(np.isfinite(solution))
            ):
                raise ValueError
            return result, np.asarray(solution).copy()
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            ValueError,
            TypeError,
        ):
            return (
                _error_result(
                    backend_id,
                    SolveStatus.PROCESS_CRASH,
                    "Worker returned a malformed solve result",
                ),
                None,
            )

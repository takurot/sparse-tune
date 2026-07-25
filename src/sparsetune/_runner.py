"""Parent-side RHS normalization and isolated worker execution."""

from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
import re
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
        values = np.asarray(loaded, dtype=dtype)
    else:
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
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _parse_sample(payload: Any) -> RunSample:
    if not isinstance(payload, dict):
        raise ValueError
    expected = {item.name for item in fields(RunSample)}
    if set(payload) != expected:
        raise ValueError
    if not isinstance(payload["measure"], str):
        raise ValueError
    for name in (
        "transfer_seconds",
        "setup_seconds",
        "solve_seconds",
        "total_seconds",
        "residual_norm",
        "convergence_threshold",
    ):
        if not _is_number(payload[name]):
            raise ValueError
    if not isinstance(payload["iterations"], int) or isinstance(
        payload["iterations"], bool
    ):
        raise ValueError
    relative = payload["relative_residual"]
    pool_used = payload["pool_used_gb"]
    if relative is not None and not _is_number(relative):
        raise ValueError
    if pool_used is not None and not _is_number(pool_used):
        raise ValueError
    if payload["error"] is not None and not isinstance(payload["error"], str):
        raise ValueError
    return RunSample(
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


def _parse_result(payload: Any, backend_id: str) -> SolverResult:
    if not isinstance(payload, dict):
        raise ValueError
    expected = {item.name for item in fields(SolverResult)}
    if set(payload) != expected or payload.get("backend") != backend_id:
        raise ValueError
    for name in ("backend", "solver_impl", "dtype"):
        if not isinstance(payload[name], str):
            raise ValueError
    for name in (
        "transfer_seconds",
        "setup_seconds",
        "solve_seconds",
        "total_seconds",
        "residual_norm",
        "convergence_threshold",
    ):
        if not _is_number(payload[name]):
            raise ValueError
    if not isinstance(payload["iterations"], int) or isinstance(
        payload["iterations"], bool
    ):
        raise ValueError
    relative = payload["relative_residual"]
    pool_used = payload["pool_used_gb"]
    if relative is not None and not _is_number(relative):
        raise ValueError
    if pool_used is not None and not _is_number(pool_used):
        raise ValueError
    if payload["error"] is not None and not isinstance(payload["error"], str):
        raise ValueError
    if not isinstance(payload["samples"], list):
        raise ValueError

    return SolverResult(
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
            return _parse_result(payload, backend_id)
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
            result = _parse_result(payload, backend_id)
            if not solution_path.is_file():
                if result.status in {
                    SolveStatus.OOM,
                    SolveStatus.UNSUPPORTED,
                    SolveStatus.INTERNAL_ERROR,
                }:
                    return result, None
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

"""Isolated worker for one selected backend solve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.sparse import load_npz  # type: ignore[import-untyped]

from ._backends import get_backend
from ._runner import classify_solve
from ._types import RunSample, SolveStatus, SolverResult
from ._worker import status_for_exception


def _read_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "dtype",
        "rtol",
        "atol",
        "max_iter",
    }:
        raise ValueError("invalid solve configuration")
    if payload["dtype"] not in {"float32", "float64"}:
        raise ValueError("invalid solve configuration")
    for name in ("rtol", "atol"):
        value = payload[name]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError("invalid solve configuration")
    if (
        not isinstance(payload["max_iter"], int)
        or isinstance(payload["max_iter"], bool)
        or payload["max_iter"] <= 0
    ):
        raise ValueError("invalid solve configuration")
    return payload


def _error_result(backend_id: str, status: SolveStatus, dtype: str) -> SolverResult:
    messages = {
        SolveStatus.OOM: "Backend ran out of memory",
        SolveStatus.UNSUPPORTED: "Backend is unavailable",
        SolveStatus.INTERNAL_ERROR: "Backend execution failed",
    }
    return SolverResult(
        backend=backend_id,
        solver_impl="",
        dtype=dtype,
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
        error=messages[status],
    )


def solve_once(
    backend_id: str,
    matrix_path: Path,
    rhs_path: Path,
    config_path: Path,
) -> tuple[SolverResult, np.ndarray[Any, Any]]:
    """Execute one solve and return validated metrics and the CPU solution."""

    config = _read_config(config_path)
    dtype = np.dtype(config["dtype"])
    matrix = load_npz(matrix_path).astype(dtype, copy=False)
    rhs = np.asarray(np.load(rhs_path, allow_pickle=False), dtype=dtype)
    if (
        matrix.shape[0] != matrix.shape[1]
        or rhs.ndim != 1
        or rhs.shape[0] != matrix.shape[0]
        or not np.all(np.isfinite(matrix.data))
        or not np.all(np.isfinite(rhs))
    ):
        raise ValueError("invalid solve input")

    backend = get_backend(backend_id)
    total_start = time.perf_counter()
    prepared = backend.prepare(matrix, rhs, dtype=config["dtype"])
    setup_end = time.perf_counter()
    try:
        solve_start = time.perf_counter()
        native = backend.solve_prepared(
            prepared,
            rtol=config["rtol"],
            atol=config["atol"],
            max_iter=config["max_iter"],
        )
        backend.synchronize()
        solve_end = time.perf_counter()
        solution = backend.fetch_solution(native)
        total_end = time.perf_counter()
    finally:
        backend.release(prepared)

    residual_norm = float(np.linalg.norm(matrix @ solution - rhs))
    rhs_norm = float(np.linalg.norm(rhs))
    threshold = max(float(config["rtol"]) * rhs_norm, float(config["atol"]))
    status = classify_solve(native.info, solution, residual_norm, threshold)
    sample = RunSample(
        measure="end-to-end",
        transfer_seconds=total_end - solve_end,
        setup_seconds=setup_end - total_start,
        solve_seconds=solve_end - solve_start,
        total_seconds=total_end - total_start,
        iterations=native.iterations,
        residual_norm=residual_norm,
        relative_residual=residual_norm / rhs_norm if rhs_norm > 0 else None,
        convergence_threshold=threshold,
        status=status,
    )
    return (
        SolverResult(
            backend=backend.id,
            solver_impl=backend.solver_impl,
            dtype=config["dtype"],
            transfer_seconds=sample.transfer_seconds,
            setup_seconds=sample.setup_seconds,
            solve_seconds=sample.solve_seconds,
            total_seconds=sample.total_seconds,
            iterations=sample.iterations,
            residual_norm=sample.residual_norm,
            relative_residual=sample.relative_residual,
            convergence_threshold=sample.convergence_threshold,
            pool_used_gb=None,
            status=status,
            error=None,
            samples=[sample],
        ),
        np.asarray(solution, dtype=dtype),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--matrix-npz", type=Path, required=True)
    parser.add_argument("--rhs", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--solution", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        payload = json.loads(args.config.read_text(encoding="utf-8"))
        requested_dtype = payload.get("dtype") if isinstance(payload, dict) else None
        dtype = requested_dtype if requested_dtype in {"float32", "float64"} else ""
    except (OSError, UnicodeError, json.JSONDecodeError):
        dtype = ""
    try:
        result, solution = solve_once(
            args.backend,
            args.matrix_npz,
            args.rhs,
            args.config,
        )
        np.save(args.solution, solution, allow_pickle=False)
    except Exception as error:
        result = _error_result(args.backend, status_for_exception(error), dtype)
    args.result.write_text(result.to_json(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

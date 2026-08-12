"""Isolated process for running one backend across measurement modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import load_npz  # type: ignore[import-untyped]

from ._backends import Backend, NativeSolveResult, UnsupportedBackendError, get_backend
from ._runner import _SUPPORTED_DTYPES, classify_solve
from ._types import RunSample, SolveStatus, SolverResult


def status_for_exception(error: BaseException) -> SolveStatus:
    """Map worker-local failures to stable public statuses."""

    if isinstance(error, UnsupportedBackendError):
        return SolveStatus.UNSUPPORTED
    if isinstance(error, MemoryError) or error.__class__.__name__ in {
        "MemoryError",
        "OutOfMemoryError",
    }:
        return SolveStatus.OOM
    return SolveStatus.INTERNAL_ERROR


def _error_result(
    backend_id: str,
    status: SolveStatus,
    dtype: str,
) -> SolverResult:
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


def _read_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid worker configuration") from error
    if not isinstance(payload, dict):
        raise ValueError("invalid worker configuration")
    if set(payload) != {"dtype", "rtol", "atol", "max_iter", "runs", "measure"}:
        raise ValueError("invalid worker configuration")
    if payload["dtype"] not in _SUPPORTED_DTYPES:
        raise ValueError("invalid worker configuration")
    for name in ("rtol", "atol"):
        value = payload[name]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError("invalid worker configuration")
    for name in ("max_iter", "runs"):
        value = payload[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("invalid worker configuration")
    measure = payload["measure"]
    if (
        not isinstance(measure, list)
        or not measure
        or any(not isinstance(item, str) for item in measure)
        or not set(measure).issubset({"end-to-end", "steady-state"})
    ):
        raise ValueError("invalid worker configuration")
    return payload


def _make_sample(
    measure: str,
    matrix: Any,
    rhs: NDArray[np.floating[Any]],
    native: NativeSolveResult,
    solution: NDArray[np.floating[Any]],
    *,
    transfer_seconds: float,
    setup_seconds: float,
    solve_seconds: float,
    total_seconds: float,
    rtol: float,
    atol: float,
) -> RunSample:
    residual_norm = float(np.linalg.norm(matrix @ solution - rhs))
    rhs_norm = float(np.linalg.norm(rhs))
    threshold = max(rtol * rhs_norm, atol)
    return RunSample(
        measure=measure,
        transfer_seconds=transfer_seconds,
        setup_seconds=setup_seconds,
        solve_seconds=solve_seconds,
        total_seconds=total_seconds,
        iterations=native.iterations,
        residual_norm=residual_norm,
        relative_residual=residual_norm / rhs_norm if rhs_norm > 0 else None,
        convergence_threshold=threshold,
        status=classify_solve(
            native.info,
            solution,
            residual_norm,
            threshold,
        ),
    )


def _warmup(
    backend: Backend,
    matrix: Any,
    rhs: NDArray[np.floating[Any]],
    config: dict[str, Any],
) -> None:
    prepared = backend.prepare(matrix, rhs, dtype=config["dtype"])
    try:
        backend.warmup(
            prepared,
            rtol=float(config["rtol"]),
            atol=float(config["atol"]),
            max_iter=int(config["max_iter"]),
        )
    finally:
        backend.release(prepared)


def _end_to_end_samples(
    backend: Backend,
    matrix: Any,
    rhs: NDArray[np.floating[Any]],
    config: dict[str, Any],
) -> list[RunSample]:
    _warmup(backend, matrix, rhs, config)
    samples = []
    for _ in range(config["runs"]):
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
            samples.append(
                _make_sample(
                    "end-to-end",
                    matrix,
                    rhs,
                    native,
                    solution,
                    transfer_seconds=total_end - solve_end,
                    setup_seconds=setup_end - total_start,
                    solve_seconds=solve_end - solve_start,
                    total_seconds=total_end - total_start,
                    rtol=config["rtol"],
                    atol=config["atol"],
                )
            )
        finally:
            backend.release(prepared)
    return samples


def _steady_state_samples(
    backend: Backend,
    matrix: Any,
    rhs: NDArray[np.floating[Any]],
    config: dict[str, Any],
) -> list[RunSample]:
    samples = []
    prepared = backend.prepare(matrix, rhs, dtype=config["dtype"])
    try:
        backend.warmup(
            prepared,
            rtol=config["rtol"],
            atol=config["atol"],
            max_iter=config["max_iter"],
        )
        for _ in range(config["runs"]):
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
            samples.append(
                _make_sample(
                    "steady-state",
                    matrix,
                    rhs,
                    native,
                    solution,
                    transfer_seconds=total_end - solve_end,
                    setup_seconds=0.0,
                    solve_seconds=solve_end - solve_start,
                    total_seconds=total_end - solve_start,
                    rtol=config["rtol"],
                    atol=config["atol"],
                )
            )
    finally:
        backend.release(prepared)
    return samples


def _median_sample(samples: list[RunSample], field: str) -> RunSample:
    target = statistics.median(float(getattr(sample, field)) for sample in samples)
    return min(samples, key=lambda sample: abs(float(getattr(sample, field)) - target))


def _aggregate(
    backend: Backend,
    dtype: str,
    samples: list[RunSample],
) -> SolverResult:
    end_to_end = [sample for sample in samples if sample.measure == "end-to-end"]
    steady_state = [sample for sample in samples if sample.measure == "steady-state"]
    primary = _median_sample(end_to_end or steady_state, "total_seconds")
    solve_sample = _median_sample(steady_state or end_to_end, "solve_seconds")
    failure = next(
        (sample for sample in samples if sample.status is not SolveStatus.CONVERGED),
        None,
    )
    accuracy_sample = failure or primary
    return SolverResult(
        backend=backend.id,
        solver_impl=backend.solver_impl,
        dtype=dtype,
        transfer_seconds=primary.transfer_seconds,
        setup_seconds=primary.setup_seconds,
        solve_seconds=solve_sample.solve_seconds,
        total_seconds=primary.total_seconds,
        iterations=accuracy_sample.iterations,
        residual_norm=accuracy_sample.residual_norm,
        relative_residual=accuracy_sample.relative_residual,
        convergence_threshold=accuracy_sample.convergence_threshold,
        pool_used_gb=accuracy_sample.pool_used_gb,
        status=accuracy_sample.status,
        error=accuracy_sample.error,
        samples=samples,
    )


def run_worker(
    backend_id: str,
    matrix_path: Path,
    rhs_path: Path,
    config_path: Path,
) -> SolverResult:
    """Load validated worker inputs and execute all requested samples."""

    config = _read_config(config_path)
    matrix = load_npz(matrix_path)
    rhs = np.load(rhs_path, allow_pickle=False)
    dtype = np.dtype(config["dtype"])
    if (
        matrix.shape[0] != matrix.shape[1]
        or rhs.ndim != 1
        or rhs.shape[0] != matrix.shape[0]
        or not np.all(np.isfinite(matrix.data))
        or not np.all(np.isfinite(rhs))
    ):
        raise ValueError("invalid worker input")
    matrix = matrix.astype(dtype, copy=False)
    rhs = np.asarray(rhs, dtype=dtype)
    if not np.all(np.isfinite(matrix.data)) or not np.all(np.isfinite(rhs)):
        raise ValueError("invalid worker input")
    backend = get_backend(backend_id)

    samples = []
    if "end-to-end" in config["measure"]:
        samples.extend(_end_to_end_samples(backend, matrix, rhs, config))
    if "steady-state" in config["measure"]:
        samples.extend(_steady_state_samples(backend, matrix, rhs, config))
    return _aggregate(backend, config["dtype"], samples)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--matrix-npz", type=Path, required=True)
    parser.add_argument("--rhs", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        payload = json.loads(args.config.read_text(encoding="utf-8"))
        requested_dtype = payload.get("dtype") if isinstance(payload, dict) else None
        dtype = requested_dtype if requested_dtype in _SUPPORTED_DTYPES else ""
    except (OSError, UnicodeError, json.JSONDecodeError):
        dtype = ""
    try:
        result = run_worker(
            args.backend,
            args.matrix_npz,
            args.rhs,
            args.config,
        )
    except Exception as error:
        result = _error_result(args.backend, status_for_exception(error), dtype)

    args.result.write_text(result.to_json(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

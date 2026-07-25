"""Benchmark orchestration and deterministic backend recommendations."""

from __future__ import annotations

import importlib
from importlib import metadata
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import __version__ as scipy_version

from ._inspect import diagnose_matrix, is_cg_eligible
from ._matrix import canonicalize_matrix
from ._runner import canonicalize_rhs, run_backend_in_subprocess
from ._types import (
    BenchmarkResult,
    MatrixInput,
    Recommendation,
    SolveStatus,
    SolverResult,
)


_MEASURES = {"end-to-end", "steady-state"}
_DTYPES = {"float32", "float64"}


def list_backends() -> list[str]:
    """Return backend IDs available when explicitly probed."""

    backends = ["scipy:cpu"]
    try:
        cupy = importlib.import_module("cupy")
        count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception:
        return backends
    backends.extend(f"cupy:cuda:{index}" for index in range(count))
    return backends


def _probe_backend(backend_id: str, dtype: str) -> str | None:
    """Exercise a tiny solve in isolation and return a stable diagnostic."""

    command = [
        sys.executable,
        "-m",
        "sparsetune._probe_worker",
        "--backend",
        backend_id,
        "--dtype",
        dtype,
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "Backend capability probe failed"
    if completed.returncode == 0:
        return None
    if completed.returncode == 2:
        return "Backend is unavailable"
    return "Backend capability probe failed"


def _unsupported_result(backend_id: str, dtype: str, error: str) -> SolverResult:
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
        status=SolveStatus.UNSUPPORTED,
        error=error,
    )


def _mode_seconds(result: SolverResult, mode: str) -> float:
    field = "total_seconds" if mode == "end-to-end" else "solve_seconds"
    samples = [sample for sample in result.samples if sample.measure == mode]
    if not samples:
        return float(getattr(result, field))
    values = sorted(float(getattr(sample, field)) for sample in samples)
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


def _break_even_solves(
    gpu: SolverResult,
    results: Iterable[SolverResult],
) -> int | None:
    cpu_results = [
        result
        for result in results
        if result.backend == "scipy:cpu" and result.status is SolveStatus.CONVERGED
    ]
    if not cpu_results:
        return None
    cpu = min(cpu_results, key=lambda item: _mode_seconds(item, "steady-state"))
    saving = _mode_seconds(cpu, "steady-state") - _mode_seconds(
        gpu,
        "steady-state",
    )
    if saving <= 0:
        return None
    overhead = max(
        _mode_seconds(gpu, "end-to-end") - _mode_seconds(gpu, "steady-state"),
        0.0,
    )
    return math.ceil(overhead / saving)


def recommend(results: Sequence[SolverResult]) -> dict[str, Recommendation]:
    """Select the fastest converged backend for each documented mode."""

    eligible = [result for result in results if result.status is SolveStatus.CONVERGED]
    recommendations: dict[str, Recommendation] = {}
    for mode in ("end-to-end", "steady-state"):
        key = mode.replace("-", "_")
        if not eligible:
            recommendations[key] = Recommendation(
                mode=mode,
                backend=None,
                reason="No backend produced a converged result",
            )
            continue

        ranked = sorted(
            eligible,
            key=lambda result: (_mode_seconds(result, mode), result.backend),
        )
        selected = ranked[0]
        selected_seconds = _mode_seconds(selected, mode)
        speedup = None
        if len(ranked) > 1 and selected_seconds > 0:
            speedup = _mode_seconds(ranked[1], mode) / selected_seconds
        recommendations[key] = Recommendation(
            mode=mode,
            backend=selected.backend,
            reason=(
                f"Fastest converged {mode} result ({selected_seconds:.6g} seconds)"
            ),
            speedup=speedup,
            break_even_solves=(
                _break_even_solves(selected, eligible)
                if selected.backend.startswith("cupy:")
                else None
            ),
        )
    return recommendations


def _environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("cupy-cuda12x", "cupy-cuda13x"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": platform.processor(),
        "cpu_cores": os.cpu_count(),
        "numpy": np.__version__,
        "scipy": scipy_version,
        "gpu_backends": [
            backend for backend in list_backends() if backend.startswith("cupy:")
        ],
        "gpu_packages": packages,
    }


def _validate_options(
    backends: Sequence[str],
    dtype: str,
    measure: Sequence[str],
    runs: int,
    rtol: float,
    atol: float,
    max_iter: int,
    timeout: float,
) -> None:
    if not backends or any(not backend for backend in backends):
        raise ValueError("at least one backend is required")
    if len(set(backends)) != len(backends):
        raise ValueError("backend IDs must be unique")
    if dtype not in _DTYPES:
        raise ValueError("dtype must be 'float32' or 'float64'")
    if not measure or not set(measure).issubset(_MEASURES):
        raise ValueError("measure must contain end-to-end or steady-state")
    if not isinstance(runs, int) or isinstance(runs, bool) or runs <= 0:
        raise ValueError("runs must be a positive integer")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    for name, value in (("rtol", rtol), ("atol", atol)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a non-negative finite number")
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a positive finite number")


def benchmark(
    matrix: MatrixInput,
    *,
    backends: Sequence[str] | None = None,
    dtype: str = "float64",
    measure: Sequence[str] = ("end-to-end", "steady-state"),
    runs: int = 5,
    rtol: float = 1.0e-6,
    atol: float = 0.0,
    max_iter: int = 10_000,
    timeout: float = 300.0,
    assume_spd: bool = False,
    rhs: str | Path | np.ndarray[Any, Any] | None = None,
) -> BenchmarkResult:
    """Benchmark requested backends in isolated workers."""

    requested = list(backends or ("scipy:cpu", "cupy:cuda:0"))
    modes = list(measure)
    _validate_options(
        requested,
        dtype,
        modes,
        runs,
        rtol,
        atol,
        max_iter,
        timeout,
    )
    matrix_info = diagnose_matrix(matrix)
    if not is_cg_eligible(matrix_info, assume_spd=assume_spd):
        if matrix_info.spd_status == "unknown":
            raise ValueError("matrix requires assume_spd=True for CG")
        raise ValueError("matrix is not eligible for CG")

    with tempfile.TemporaryDirectory(prefix="sparsetune-benchmark-") as temporary:
        work_dir = Path(temporary)
        matrix_path, canonical = canonicalize_matrix(matrix, dtype, work_dir)
        rhs_path, _ = canonicalize_rhs(
            rhs,
            canonical,
            dtype_str=dtype,
            work_dir=work_dir,
        )
        config: dict[str, Any] = {
            "dtype": dtype,
            "rtol": rtol,
            "atol": atol,
            "max_iter": max_iter,
            "runs": runs,
            "measure": modes,
        }
        results = []
        for backend_id in requested:
            probe_error = _probe_backend(backend_id, dtype)
            if probe_error is not None:
                results.append(
                    _unsupported_result(backend_id, dtype, probe_error),
                )
                continue
            results.append(
                run_backend_in_subprocess(
                    backend_id,
                    matrix_path,
                    rhs_path,
                    config,
                    timeout=timeout,
                )
            )

    return BenchmarkResult(
        matrix=matrix_info,
        environment=_environment(),
        results=results,
        recommendations=recommend(results),
    )

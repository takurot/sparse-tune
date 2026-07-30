"""Shared public data types."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix  # type: ignore[import-untyped]

_SCHEMA_VERSION = "1.0"


class SolveStatus(str, Enum):
    """Stable outcome values returned by solver runs."""

    CONVERGED = "converged"
    ACCURACY_FAILED = "accuracy_failed"
    MAX_ITER = "max_iter"
    BREAKDOWN = "breakdown"
    NAN_INF = "nan_inf"
    OOM = "oom"
    TIMEOUT = "timeout"
    PROCESS_CRASH = "process_crash"
    UNSUPPORTED = "unsupported"
    INTERNAL_ERROR = "internal_error"


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _json_value(getattr(value, item.name)) for item in fields(value)
        }
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class _Serializable:
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return cast(dict[str, Any], _json_value(self))

    def to_json(self) -> str:
        """Serialize the value to JSON."""

        return json.dumps(self.to_dict())


@dataclass
class RunSample(_Serializable):
    """Raw timing and accuracy values from one solver trial."""

    measure: str
    transfer_seconds: float
    setup_seconds: float
    solve_seconds: float
    total_seconds: float
    iterations: int
    residual_norm: float
    relative_residual: float | None
    convergence_threshold: float
    status: SolveStatus
    error: str | None = None
    pool_used_gb: float | None = None


@dataclass
class SolverResult(_Serializable):
    """Aggregated result for one solver backend."""

    backend: str
    solver_impl: str
    dtype: str
    transfer_seconds: float
    setup_seconds: float
    solve_seconds: float
    total_seconds: float
    iterations: int
    residual_norm: float
    relative_residual: float | None
    convergence_threshold: float
    pool_used_gb: float | None
    status: SolveStatus
    error: str | None
    samples: list[RunSample] = field(default_factory=list)


@dataclass
class SolveResult(_Serializable):
    """Result of one selected backend solve."""

    x: NDArray[np.floating[Any]] | None
    backend: str
    dtype: str
    status: SolveStatus
    iterations: int
    residual_norm: float
    relative_residual: float | None
    convergence_threshold: float
    setup_seconds: float
    solve_seconds: float
    total_seconds: float
    error: str | None

    def to_metrics_dict(self) -> dict[str, Any]:
        """Return JSON-compatible solve metadata without materializing ``x``."""

        return {
            item.name: _json_value(getattr(self, item.name))
            for item in fields(self)
            if item.name != "x"
        }

    def to_metrics_json(self) -> str:
        """Serialize solve metadata without the solution vector."""

        return json.dumps(self.to_metrics_dict())


@dataclass
class MatrixInfo(_Serializable):
    """Structural information about a canonical sparse matrix."""

    path: str
    shape: tuple[int, int]
    nnz: int
    density: float
    is_square: bool
    symmetry_ratio: float
    diagonal_sign: str
    spd_status: str
    fingerprint: str


@dataclass
class Recommendation(_Serializable):
    """Backend recommendation for one measurement mode."""

    mode: str
    backend: str | None
    reason: str
    speedup: float | None = None
    break_even_solves: int | None = None


@dataclass
class BenchmarkResult(_Serializable):
    """Complete benchmark report."""

    matrix: MatrixInfo
    environment: dict[str, Any]
    results: list[SolverResult]
    recommendations: dict[str, Recommendation]
    dtype: str = "float64"
    schema_version: str = _SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the authoritative versioned benchmark payload."""

        return {
            "schema_version": self.schema_version,
            "matrix": {**self.matrix.to_dict(), "dtype": self.dtype},
            "environment": _json_value(self.environment),
            "results": _json_value(self.results),
            "recommendations": _json_value(self.recommendations),
        }

    def best(self, mode: str = "end-to-end") -> SolverResult | None:
        """Return the result selected for a measurement mode."""

        recommendation = self.recommendations.get(mode.replace("-", "_"))
        if recommendation is None or recommendation.backend is None:
            return None
        return next(
            (
                result
                for result in self.results
                if result.backend == recommendation.backend
            ),
            None,
        )


@dataclass
class CanonicalMatrix:
    """Sparse matrix in deterministic, CPU-resident CSR form."""

    data: NDArray[np.float32] | NDArray[np.float64]
    indices: NDArray[np.int32]
    indptr: NDArray[np.int32]
    shape: tuple[int, int]
    fingerprint: str

    @property
    def dtype(self) -> str:
        return self.data.dtype.name

    @property
    def nnz(self) -> int:
        return int(self.data.size)


Profile: TypeAlias = dict[str, Any]
MatrixInput: TypeAlias = str | Path | CanonicalMatrix | spmatrix

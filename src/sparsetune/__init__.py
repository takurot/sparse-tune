"""Public package interface for sparsetune."""

from ._benchmark import benchmark, list_backends
from ._matrix import (
    canonicalize_matrix,
    ensure_canonical_matrix,
    fingerprint_csr,
    load_matrix,
)
from ._inspect import diagnose_matrix
from ._types import (
    BenchmarkResult,
    CanonicalMatrix,
    MatrixInfo,
    MatrixInput,
    Profile,
    Recommendation,
    RunSample,
    SolveStatus,
    SolverResult,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "BenchmarkResult",
    "CanonicalMatrix",
    "MatrixInfo",
    "MatrixInput",
    "Profile",
    "Recommendation",
    "RunSample",
    "SolveStatus",
    "SolverResult",
    "__version__",
    "benchmark",
    "canonicalize_matrix",
    "diagnose_matrix",
    "ensure_canonical_matrix",
    "fingerprint_csr",
    "load_matrix",
    "list_backends",
]

"""Public package interface for sparsetune."""

from ._benchmark import benchmark, list_backends
from ._matrix import (
    canonicalize_matrix,
    ensure_canonical_matrix,
    fingerprint_csr,
    load_matrix,
)
from ._inspect import diagnose_matrix, inspect
from ._profile import (
    ProfileMismatchError,
    load_profile,
    solve,
    tune,
)
from ._types import (
    BenchmarkResult,
    CanonicalMatrix,
    MatrixInfo,
    MatrixInput,
    Profile,
    Recommendation,
    RunSample,
    SolveResult,
    SolveStatus,
    SolverResult,
)

__version__ = "0.1.11"

__all__ = [
    "BenchmarkResult",
    "CanonicalMatrix",
    "MatrixInfo",
    "MatrixInput",
    "Profile",
    "ProfileMismatchError",
    "Recommendation",
    "RunSample",
    "SolveResult",
    "SolveStatus",
    "SolverResult",
    "__version__",
    "benchmark",
    "canonicalize_matrix",
    "diagnose_matrix",
    "ensure_canonical_matrix",
    "fingerprint_csr",
    "inspect",
    "load_matrix",
    "list_backends",
    "load_profile",
    "solve",
    "tune",
]

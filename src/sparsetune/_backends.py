"""Backend boundary and native SciPy CG implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, issparse  # type: ignore[import-untyped]
from scipy.sparse.linalg import cg  # type: ignore[import-untyped]


_SUPPORTED_DTYPES = {"float32", "float64"}


@dataclass
class PreparedSystem:
    """Opaque backend-owned matrix and right-hand-side state."""

    matrix: Any
    rhs: Any


@dataclass
class NativeSolveResult:
    """Unclassified result returned directly by a native solver."""

    solution: Any
    info: int
    iterations: int


class Backend(Protocol):
    """Operations required to keep preparation and solve timing separate."""

    id: str
    solver_impl: str

    def prepare(
        self,
        matrix: Any,
        rhs: NDArray[np.floating[Any]],
        *,
        dtype: str,
    ) -> PreparedSystem: ...

    def warmup(
        self,
        prepared: PreparedSystem,
        *,
        rtol: float,
        atol: float,
        max_iter: int,
    ) -> None: ...

    def solve_prepared(
        self,
        prepared: PreparedSystem,
        *,
        rtol: float,
        atol: float,
        max_iter: int,
    ) -> NativeSolveResult: ...

    def fetch_solution(
        self,
        result: NativeSolveResult,
    ) -> NDArray[np.floating[Any]]: ...

    def synchronize(self) -> None: ...

    def release(self, prepared: PreparedSystem) -> None: ...


class SciPyBackend:
    """CPU backend using :func:`scipy.sparse.linalg.cg`."""

    id = "scipy:cpu"
    solver_impl = "scipy.sparse.linalg.cg"

    def prepare(
        self,
        matrix: Any,
        rhs: NDArray[np.floating[Any]],
        *,
        dtype: str,
    ) -> PreparedSystem:
        if dtype not in _SUPPORTED_DTYPES:
            raise ValueError("dtype must be 'float32' or 'float64'")
        if not issparse(matrix):
            raise TypeError("matrix must be a SciPy sparse matrix")

        np_dtype = np.dtype(dtype)
        prepared_matrix = csr_matrix(matrix, dtype=np_dtype, copy=True)
        prepared_rhs = np.asarray(rhs, dtype=np_dtype)
        if prepared_rhs.ndim != 1 or prepared_rhs.shape[0] != prepared_matrix.shape[0]:
            raise ValueError("RHS must be a vector matching the matrix row count")
        return PreparedSystem(matrix=prepared_matrix, rhs=prepared_rhs.copy())

    def warmup(
        self,
        prepared: PreparedSystem,
        *,
        rtol: float,
        atol: float,
        max_iter: int,
    ) -> None:
        self.solve_prepared(
            prepared,
            rtol=rtol,
            atol=atol,
            max_iter=min(max_iter, 2),
        )

    def solve_prepared(
        self,
        prepared: PreparedSystem,
        *,
        rtol: float,
        atol: float,
        max_iter: int,
    ) -> NativeSolveResult:
        iterations = 0

        def count_iteration(_solution: NDArray[np.floating[Any]]) -> None:
            nonlocal iterations
            iterations += 1

        solution, info = cg(
            prepared.matrix,
            prepared.rhs,
            rtol=rtol,
            atol=atol,
            maxiter=max_iter,
            callback=count_iteration,
        )
        return NativeSolveResult(
            solution=solution,
            info=int(info),
            iterations=iterations,
        )

    def fetch_solution(
        self,
        result: NativeSolveResult,
    ) -> NDArray[np.floating[Any]]:
        return np.asarray(result.solution).copy()

    def synchronize(self) -> None:
        return None

    def release(self, prepared: PreparedSystem) -> None:
        return None

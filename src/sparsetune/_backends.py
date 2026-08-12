"""Backend boundary and native SciPy CG implementation."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, issparse  # type: ignore[import-untyped]
from scipy.sparse.linalg import cg  # type: ignore[import-untyped]


_SUPPORTED_DTYPES = {"float32", "float64"}


def _tolerance_kwargs(
    solver: Any,
    *,
    rtol: float,
    atol: float,
) -> dict[str, float]:
    """Map the public tolerance contract to old and current CG signatures."""

    try:
        parameters = inspect.signature(solver).parameters.values()
    except (TypeError, ValueError):
        return {"rtol": rtol, "atol": atol}
    if any(parameter.name == "rtol" for parameter in parameters) or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
    ):
        return {"rtol": rtol, "atol": atol}
    return {"tol": rtol, "atol": atol}


class UnsupportedBackendError(RuntimeError):
    """Raised when a requested backend cannot run in this environment."""


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
            maxiter=max_iter,
            callback=count_iteration,
            **_tolerance_kwargs(cg, rtol=rtol, atol=atol),
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


class CuPyBackend:
    """CUDA backend using CuPy's native sparse CG implementation."""

    solver_impl = "cupyx.scipy.sparse.linalg.cg"

    def __init__(self, backend_id: str = "cupy:cuda:0") -> None:
        prefix = "cupy:cuda:"
        if not backend_id.startswith(prefix):
            raise UnsupportedBackendError(f"Unknown backend: {backend_id}")
        try:
            device_index = int(backend_id.removeprefix(prefix))
        except ValueError as error:
            raise UnsupportedBackendError(f"Unknown backend: {backend_id}") from error
        if device_index < 0:
            raise UnsupportedBackendError(f"Unknown backend: {backend_id}")

        try:
            self._cp = importlib.import_module("cupy")
            self._sparse = importlib.import_module("cupyx.scipy.sparse")
            self._linalg = importlib.import_module("cupyx.scipy.sparse.linalg")
            if device_index >= int(self._cp.cuda.runtime.getDeviceCount()):
                raise RuntimeError(f"CUDA device {device_index} is unavailable")
            self._cp.cuda.Device(device_index).use()
            # No private pinned-memory pool: CuPy exposes no scoped
            # equivalent of using_allocator() for set_pinned_memory_allocator
            # (it is process-global), and this backend never stages host
            # buffers through pinned memory -- cp.asarray() on a plain NumPy
            # array does not use it -- so there is nothing of ours to
            # isolate or release.
            self._mempool = self._cp.cuda.MemoryPool()
        except Exception as error:
            raise UnsupportedBackendError(f"CuPy is unavailable: {error}") from error

        self.id = backend_id

    def _owned_allocator(self) -> Any:
        """Scope device allocations to this instance's private pool.

        ``cupy.cuda.set_allocator`` is process-global, so a persistent call
        in ``__init__`` would let a second ``CuPyBackend`` silently steal
        allocation routing from the first. ``using_allocator`` instead
        pushes/pops the current allocator only around each call, which stays
        correct as long as calls from different instances do not interleave
        (true for this project's single-threaded, one-backend-per-process
        worker model).
        """

        return self._cp.cuda.using_allocator(self._mempool.malloc)

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

        cpu_matrix = csr_matrix(matrix, copy=False)
        cpu_rhs = np.asarray(rhs)
        if cpu_rhs.ndim != 1 or cpu_rhs.shape[0] != cpu_matrix.shape[0]:
            raise ValueError("RHS must be a vector matching the matrix row count")

        cp_dtype = self._cp.float32 if dtype == "float32" else self._cp.float64
        with self._owned_allocator():
            gpu_matrix = self._sparse.csr_matrix(
                (
                    self._cp.asarray(cpu_matrix.data, dtype=cp_dtype),
                    self._cp.asarray(cpu_matrix.indices),
                    self._cp.asarray(cpu_matrix.indptr),
                ),
                shape=cpu_matrix.shape,
            )
            gpu_rhs = self._cp.asarray(cpu_rhs, dtype=cp_dtype)
        return PreparedSystem(matrix=gpu_matrix, rhs=gpu_rhs)

    def warmup(
        self,
        prepared: PreparedSystem,
        *,
        rtol: float,
        atol: float,
        max_iter: int,
    ) -> None:
        with self._owned_allocator():
            _ = prepared.matrix @ prepared.rhs
        self.solve_prepared(
            prepared,
            rtol=rtol,
            atol=atol,
            max_iter=min(max_iter, 2),
        )
        self.synchronize()

    def solve_prepared(
        self,
        prepared: PreparedSystem,
        *,
        rtol: float,
        atol: float,
        max_iter: int,
    ) -> NativeSolveResult:
        iterations = 0

        def count_iteration(_solution: Any) -> None:
            nonlocal iterations
            iterations += 1

        with self._owned_allocator():
            solution, info = self._linalg.cg(
                prepared.matrix,
                prepared.rhs,
                maxiter=max_iter,
                callback=count_iteration,
                **_tolerance_kwargs(self._linalg.cg, rtol=rtol, atol=atol),
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
        return np.asarray(self._cp.asnumpy(result.solution)).copy()

    def synchronize(self) -> None:
        self._cp.cuda.Stream.null.synchronize()

    def release(self, prepared: PreparedSystem) -> None:
        prepared.matrix = None
        prepared.rhs = None
        self._mempool.free_all_blocks()


def get_backend(backend_id: str) -> Backend:
    """Construct a supported backend without probing at package import time."""

    if backend_id == SciPyBackend.id:
        return SciPyBackend()
    if backend_id.startswith("cupy:cuda:"):
        return CuPyBackend(backend_id)
    raise UnsupportedBackendError(f"Unknown backend: {backend_id}")

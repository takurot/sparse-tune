from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix, diags

import sparsetune._backends as backends
from sparsetune._backends import (
    CuPyBackend,
    SciPyBackend,
    UnsupportedBackendError,
    get_backend,
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_scipy_backend_prepares_and_solves_spd_system(
    dtype: type[np.floating],
) -> None:
    matrix = diags(
        [-np.ones(9), 2.0 * np.ones(10), -np.ones(9)],
        offsets=[-1, 0, 1],
        format="csr",
        dtype=dtype,
    )
    expected = np.ones(10, dtype=dtype)
    rhs = np.asarray(matrix @ expected, dtype=dtype)
    backend = SciPyBackend()

    prepared = backend.prepare(matrix, rhs, dtype=np.dtype(dtype).name)
    backend.warmup(prepared, rtol=1.0e-6, atol=0.0, max_iter=100)
    native = backend.solve_prepared(
        prepared,
        rtol=1.0e-6,
        atol=0.0,
        max_iter=100,
    )
    backend.synchronize()
    solution = backend.fetch_solution(native)
    backend.release(prepared)

    assert prepared.matrix.dtype == dtype
    assert prepared.rhs.dtype == dtype
    assert native.info == 0
    assert native.iterations > 0
    assert solution.dtype == dtype
    np.testing.assert_allclose(matrix @ solution, rhs, rtol=1.0e-5, atol=1.0e-6)


def test_scipy_backend_preserves_native_info_for_classification() -> None:
    matrix = csr_matrix(np.diag(np.arange(1.0, 11.0)))
    rhs = np.ones(10)
    backend = SciPyBackend()
    prepared = backend.prepare(matrix, rhs, dtype="float64")

    native = backend.solve_prepared(
        prepared,
        rtol=0.0,
        atol=0.0,
        max_iter=1,
    )

    assert native.info == 1
    assert native.iterations == 1


def test_scipy_backend_supports_legacy_tol_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def legacy_cg(
        matrix: object,
        rhs: object,
        *,
        tol: float,
        atol: float,
        maxiter: int,
        callback: object,
    ) -> tuple[np.ndarray, int]:
        calls.append(
            {
                "tol": tol,
                "atol": atol,
                "maxiter": maxiter,
                "callback": callback,
            }
        )
        return np.ones(2), 0

    monkeypatch.setattr(backends, "cg", legacy_cg)
    backend = SciPyBackend()
    prepared = backend.prepare(
        csr_matrix(np.eye(2)),
        np.ones(2),
        dtype="float64",
    )

    backend.solve_prepared(
        prepared,
        rtol=1.0e-6,
        atol=1.0e-9,
        max_iter=20,
    )

    assert calls[0]["tol"] == 1.0e-6
    assert calls[0]["atol"] == 1.0e-9
    assert calls[0]["maxiter"] == 20


def test_scipy_synchronize_and_release_are_safe_no_ops() -> None:
    backend = SciPyBackend()
    prepared = backend.prepare(
        csr_matrix(np.eye(2)),
        np.ones(2),
        dtype="float64",
    )

    assert backend.synchronize() is None
    assert backend.release(prepared) is None
    assert backend.release(prepared) is None


class _FakePool:
    def __init__(self) -> None:
        self.free_calls = 0
        self.malloc_calls = 0

    def malloc(self, _size: int) -> None:
        self.malloc_calls += 1

    def free_all_blocks(self) -> None:
        self.free_calls += 1


class _FakeStream:
    def __init__(self) -> None:
        self.synchronize_calls = 0

    def synchronize(self) -> None:
        self.synchronize_calls += 1


class _FakeUsingAllocator:
    """Mirrors cupy.cuda.using_allocator: push/pop the current allocator."""

    def __init__(self, cupy: "_FakeCupy", malloc: object) -> None:
        self._cupy = cupy
        self._malloc = malloc
        self._previous: object | None = None

    def __enter__(self) -> "_FakeUsingAllocator":
        self._previous = self._cupy.current_allocator
        self._cupy.current_allocator = self._malloc
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self._cupy.current_allocator = self._previous


class _FakeCupy:
    float32 = np.float32
    float64 = np.float64

    def __init__(self) -> None:
        self.stream = _FakeStream()
        self.default_memory_pool = _FakePool()
        self.default_pinned_pool = _FakePool()
        self.current_allocator: object = self.default_memory_pool.malloc
        self.asarray_calls = 0
        outer = self

        self.cuda = type(
            "Cuda",
            (),
            {
                "Stream": type("Stream", (), {"null": self.stream}),
                "Device": lambda _index: type(
                    "Device", (), {"use": lambda _self: None}
                )(),
                "runtime": type(
                    "Runtime",
                    (),
                    {"getDeviceCount": staticmethod(lambda: 1)},
                ),
                "MemoryPool": staticmethod(_FakePool),
                "PinnedMemoryPool": staticmethod(_FakePool),
                "using_allocator": staticmethod(
                    lambda malloc: _FakeUsingAllocator(outer, malloc)
                ),
            },
        )

    def asarray(self, value: object, dtype: object | None = None) -> np.ndarray:
        self.asarray_calls += 1
        self.current_allocator(0)
        return np.asarray(value, dtype=dtype)

    @staticmethod
    def asnumpy(value: object) -> np.ndarray:
        return np.asarray(value)

    def get_default_memory_pool(self) -> _FakePool:
        return self.default_memory_pool

    def get_default_pinned_memory_pool(self) -> _FakePool:
        return self.default_pinned_pool


class _FakeCupySparse:
    csr_matrix = staticmethod(csr_matrix)


class _FakeCupyLinalg:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def cg(self, matrix: object, rhs: object, **kwargs: object) -> object:
        self.calls.append(kwargs.copy())
        solution = np.linalg.solve(matrix.toarray(), rhs)
        callback = kwargs.get("callback")
        if callable(callback):
            callback(solution)
        return solution, 0


def _install_fake_cupy(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_FakeCupy, _FakeCupyLinalg]:
    cupy = _FakeCupy()
    linalg = _FakeCupyLinalg()
    modules = {
        "cupy": cupy,
        "cupyx.scipy.sparse": _FakeCupySparse(),
        "cupyx.scipy.sparse.linalg": linalg,
    }
    monkeypatch.setattr(
        backends.importlib,
        "import_module",
        lambda name: modules[name],
    )
    return cupy, linalg


def test_cupy_backend_keeps_warmup_outside_samples_and_releases_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cupy, linalg = _install_fake_cupy(monkeypatch)
    backend = CuPyBackend("cupy:cuda:0")
    matrix = diags([1.0, 2.0, 3.0], format="csr")
    prepared = backend.prepare(matrix, np.ones(3), dtype="float32")

    backend.warmup(prepared, rtol=1.0e-4, atol=1.0e-7, max_iter=50)
    native = backend.solve_prepared(
        prepared,
        rtol=1.0e-6,
        atol=1.0e-8,
        max_iter=20,
    )
    backend.synchronize()
    solution = backend.fetch_solution(native)
    backend.release(prepared)

    assert prepared.matrix is None
    assert prepared.rhs is None
    assert solution.dtype == np.float32
    assert native.info == 0
    assert native.iterations > 0
    assert linalg.calls[0]["maxiter"] == 2
    assert linalg.calls[1]["rtol"] == 1.0e-6
    assert linalg.calls[1]["atol"] == 1.0e-8
    assert cupy.stream.synchronize_calls >= 2
    assert backend._mempool.free_calls == 1
    assert backend._pinned_pool.free_calls == 1
    assert cupy.default_memory_pool.free_calls == 0
    assert cupy.default_pinned_pool.free_calls == 0


def test_cupy_backend_release_does_not_purge_unrelated_prepared_system(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cupy, _linalg = _install_fake_cupy(monkeypatch)
    matrix = diags([1.0, 2.0, 3.0], format="csr")

    first = CuPyBackend("cupy:cuda:0")
    first_prepared = first.prepare(matrix, np.ones(3), dtype="float32")
    second = CuPyBackend("cupy:cuda:0")
    second.prepare(matrix, np.ones(3), dtype="float32")

    # Each prepare() call must have routed its allocations to its own
    # instance's pool, not whichever instance was constructed most recently.
    assert first._mempool.malloc_calls > 0
    assert second._mempool.malloc_calls > 0

    first.release(first_prepared)

    assert first._mempool.free_calls == 1
    assert first._pinned_pool.free_calls == 1
    assert second._mempool.free_calls == 0
    assert second._pinned_pool.free_calls == 0
    assert cupy.default_memory_pool.free_calls == 0
    assert cupy.default_pinned_pool.free_calls == 0
    # Allocator scoping must not leak: after both prepare() calls return,
    # the process-global "current" allocator is back to CuPy's own default.
    assert cupy.current_allocator == cupy.default_memory_pool.malloc


def test_cupy_backend_rejects_invalid_rhs_before_gpu_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cupy, _linalg = _install_fake_cupy(monkeypatch)
    backend = CuPyBackend("cupy:cuda:0")
    matrix = diags([1.0, 2.0, 3.0], format="csr")

    with pytest.raises(
        ValueError, match="RHS must be a vector matching the matrix row count"
    ):
        backend.prepare(matrix, np.ones(4), dtype="float32")

    assert cupy.asarray_calls == 0


def test_missing_cupy_is_reported_as_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_module(_name: str) -> object:
        raise ModuleNotFoundError("No module named 'cupy'")

    monkeypatch.setattr(backends.importlib, "import_module", missing_module)

    with pytest.raises(UnsupportedBackendError, match="CuPy is unavailable"):
        CuPyBackend()


def test_get_backend_rejects_unknown_backend() -> None:
    assert isinstance(get_backend("scipy:cpu"), SciPyBackend)
    with pytest.raises(UnsupportedBackendError, match="Unknown backend"):
        get_backend("other:device")

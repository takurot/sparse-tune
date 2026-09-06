from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.sparse import csr_matrix, diags

import sparsetune._backends as backends
from sparsetune._backends import (
    CuPyBackend,
    SciPyBackend,
    UnsupportedBackendError,
    UnsupportedSolverSignatureError,
    get_backend,
)


def test_tolerance_kwargs_uses_explicit_rtol_when_present() -> None:
    def solver(matrix: object, rhs: object, *, rtol: float, atol: float) -> None:
        raise NotImplementedError

    assert backends._tolerance_kwargs(solver, rtol=1.0e-6, atol=1.0e-9) == {
        "rtol": 1.0e-6,
        "atol": 1.0e-9,
    }


def test_tolerance_kwargs_uses_legacy_tol_when_no_rtol_present() -> None:
    def solver(matrix: object, rhs: object, *, tol: float, atol: float) -> None:
        raise NotImplementedError

    assert backends._tolerance_kwargs(solver, rtol=1.0e-6, atol=1.0e-9) == {
        "tol": 1.0e-6,
        "atol": 1.0e-9,
    }


def test_tolerance_kwargs_rejects_kwargs_only_signature() -> None:
    def solver(matrix: object, rhs: object, **kwargs: object) -> None:
        raise NotImplementedError

    with pytest.raises(UnsupportedSolverSignatureError):
        backends._tolerance_kwargs(solver, rtol=1.0e-6, atol=1.0e-9)


def test_tolerance_kwargs_rejects_uninspectable_callable() -> None:
    with pytest.raises(UnsupportedSolverSignatureError):
        backends._tolerance_kwargs(dict.update, rtol=1.0e-6, atol=1.0e-9)


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

    def cg(
        self,
        matrix: object,
        rhs: object,
        *,
        x0: object = None,
        rtol: float = 1.0e-5,
        atol: float = 0.0,
        maxiter: int | None = None,
        M: object = None,
        callback: object = None,
    ) -> object:
        kwargs = {
            "x0": x0,
            "rtol": rtol,
            "atol": atol,
            "maxiter": maxiter,
            "M": M,
            "callback": callback,
        }
        self.calls.append(kwargs.copy())
        solution = np.linalg.solve(matrix.toarray(), rhs)
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
    assert cupy.default_memory_pool.free_calls == 0
    assert cupy.default_pinned_pool.free_calls == 0


def test_cupy_backend_release_does_not_purge_unrelated_prepared_system(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cupy, _linalg = _install_fake_cupy(monkeypatch)
    matrix = diags([1.0, 2.0, 3.0], format="csr")

    # Both instances are constructed before either prepares, so a
    # persistent (unscoped) allocator override would have the second
    # instance's constructor steal allocation routing out from under the
    # first instance's later prepare() call -- the exact regression this
    # test guards against.
    first = CuPyBackend("cupy:cuda:0")
    second = CuPyBackend("cupy:cuda:0")
    first_prepared = first.prepare(matrix, np.ones(3), dtype="float32")
    second.prepare(matrix, np.ones(3), dtype="float32")

    # Each prepare() call must have routed its allocations to its own
    # instance's pool, not whichever instance was constructed most recently.
    assert first._mempool.malloc_calls > 0
    assert second._mempool.malloc_calls > 0

    first.release(first_prepared)

    assert first._mempool.free_calls == 1
    assert second._mempool.free_calls == 0
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

    with pytest.raises(
        UnsupportedBackendError, match="CuPy is unavailable"
    ) as exc_info:
        CuPyBackend()
    assert isinstance(exc_info.value.__cause__, ImportError)


@pytest.mark.parametrize("backend_id", ["cupy:cuda:-1", "cupy:cuda:x", "cupy:cuda:"])
def test_cupy_invalid_device_index_is_rejected(backend_id: str) -> None:
    with pytest.raises(UnsupportedBackendError, match="Unknown backend"):
        CuPyBackend(backend_id)


def _install_minimal_cupy(
    monkeypatch: pytest.MonkeyPatch,
    *,
    get_device_count: Any = lambda: 1,
    device_cls: type | None = None,
    memory_pool_cls: type | None = None,
    runtime_cls: type | None = None,
    driver_cls: type | None = None,
) -> None:
    if runtime_cls is None:

        class DummyRuntime:
            getDeviceCount = staticmethod(get_device_count)

        runtime_cls = DummyRuntime

    class DummyDevice:
        def __init__(self, index: int) -> None:
            pass

        def use(self) -> None:
            pass

    class DummyCUDA:
        runtime = runtime_cls
        driver = driver_cls
        Device = device_cls or DummyDevice
        MemoryPool = memory_pool_cls or (lambda: object())

    class DummyCuPy:
        cuda = DummyCUDA

    def dummy_import(name: str) -> object:
        if name in ("cupy", "cupyx.scipy.sparse", "cupyx.scipy.sparse.linalg"):
            return DummyCuPy
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(backends.importlib, "import_module", dummy_import)


def test_cupy_out_of_range_device_reports_device_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_minimal_cupy(monkeypatch, get_device_count=lambda: 1)

    with pytest.raises(
        UnsupportedBackendError,
        match=r"CUDA device 2 is unavailable \(requested index 2 >= device count 1\)",
    ) as exc_info:
        CuPyBackend("cupy:cuda:2")
    assert "CuPy is unavailable" not in str(exc_info.value)


def test_cupy_zero_devices_reports_device_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_minimal_cupy(monkeypatch, get_device_count=lambda: 0)

    with pytest.raises(
        UnsupportedBackendError,
        match="CUDA device 0 is unavailable: no CUDA devices detected",
    ) as exc_info:
        CuPyBackend("cupy:cuda:0")
    assert "CuPy is unavailable" not in str(exc_info.value)


def test_cupy_runtime_error_preserves_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_count() -> int:
        raise RuntimeError("cuda driver missing")

    _install_minimal_cupy(monkeypatch, get_device_count=fail_count)

    with pytest.raises(
        UnsupportedBackendError, match="CUDA runtime is unavailable"
    ) as exc_info:
        CuPyBackend("cupy:cuda:0")
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "cuda driver missing" in str(exc_info.value)


def test_cupy_runtime_specific_error_is_converted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CUDARuntimeError(Exception):
        pass

    class CustomRuntime:
        @staticmethod
        def getDeviceCount() -> int:
            raise CUDARuntimeError("no device found via cuda runtime")

    setattr(CustomRuntime, "CUDARuntimeError", CUDARuntimeError)

    _install_minimal_cupy(monkeypatch, runtime_cls=CustomRuntime)

    with pytest.raises(
        UnsupportedBackendError, match="CUDA runtime is unavailable"
    ) as exc_info:
        CuPyBackend("cupy:cuda:0")
    assert isinstance(exc_info.value.__cause__, CUDARuntimeError)
    assert "no device found via cuda runtime" in str(exc_info.value)


def test_cupy_device_activation_failure_preserves_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingDevice:
        def __init__(self, index: int) -> None:
            pass

        def use(self) -> None:
            raise RuntimeError("device busy or lost")

    _install_minimal_cupy(monkeypatch, device_cls=FailingDevice)

    with pytest.raises(
        UnsupportedBackendError, match="Failed to activate CUDA device 0"
    ) as exc_info:
        CuPyBackend("cupy:cuda:0")
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "device busy or lost" in str(exc_info.value)


def test_cupy_memory_pool_failure_preserves_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingMemoryPool:
        def __init__(self) -> None:
            raise RuntimeError("out of host memory for pool")

    _install_minimal_cupy(monkeypatch, memory_pool_cls=FailingMemoryPool)

    with pytest.raises(
        UnsupportedBackendError, match="Failed to initialize CUDA memory pool"
    ) as exc_info:
        CuPyBackend("cupy:cuda:0")
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "out of host memory for pool" in str(exc_info.value)


def test_cupy_unexpected_exception_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_import(_name: str) -> object:
        raise TypeError("unexpected programmer error")

    monkeypatch.setattr(backends.importlib, "import_module", broken_import)

    with pytest.raises(TypeError, match="unexpected programmer error"):
        CuPyBackend("cupy:cuda:0")


def test_cupy_sanitizes_secrets_in_error_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_count() -> int:
        raise RuntimeError(
            "failure with api_key=secret123 and Authorization: Bearer secret_bearer_token and "
            + ("long_tail_" * 70)
        )

    _install_minimal_cupy(monkeypatch, get_device_count=fail_count)

    with pytest.raises(UnsupportedBackendError) as exc_info:
        CuPyBackend("cupy:cuda:0")
    message = str(exc_info.value)
    assert "secret123" not in message
    assert "secret_bearer_token" not in message
    assert "[REDACTED]" in message
    assert "...[truncated]" in message


def test_get_backend_rejects_unknown_backend() -> None:
    assert isinstance(get_backend("scipy:cpu"), SciPyBackend)
    with pytest.raises(UnsupportedBackendError, match="Unknown backend"):
        get_backend("other:device")

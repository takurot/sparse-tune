from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import diags

from sparsetune._backends import CuPyBackend, UnsupportedBackendError


@pytest.mark.gpu
def test_cupy_backend_solves_when_cuda_is_available() -> None:
    pytest.importorskip("cupy")
    try:
        backend = CuPyBackend()
    except UnsupportedBackendError as error:
        pytest.skip(str(error))

    matrix = diags([1.0, 2.0, 3.0], format="csr")
    rhs = np.ones(3)
    prepared = backend.prepare(matrix, rhs, dtype="float64")
    try:
        native = backend.solve_prepared(
            prepared,
            rtol=1.0e-6,
            atol=0.0,
            max_iter=50,
        )
        backend.synchronize()
        solution = backend.fetch_solution(native)
    finally:
        backend.release(prepared)

    assert native.info == 0
    np.testing.assert_allclose(matrix @ solution, rhs, rtol=1.0e-6)


@pytest.mark.gpu
def test_cupy_backend_release_does_not_purge_default_pool() -> None:
    cupy = pytest.importorskip("cupy")
    try:
        backend = CuPyBackend()
    except UnsupportedBackendError as error:
        pytest.skip(str(error))

    default_pool = cupy.get_default_memory_pool()
    default_pinned_pool = cupy.get_default_pinned_memory_pool()
    default_pool.free_all_blocks()
    default_pinned_pool.free_all_blocks()

    # Simulate an unrelated CuPy user's cache living in the default pool:
    # an allocation that is freed (so its blocks become cached, not
    # in-use) but must survive our backend's release().
    unrelated = cupy.arange(4096)
    del unrelated
    baseline_free_blocks = default_pool.n_free_blocks()
    assert baseline_free_blocks > 0

    matrix = diags([1.0, 2.0, 3.0], format="csr")
    rhs = np.ones(3)
    prepared = backend.prepare(matrix, rhs, dtype="float64")
    backend.release(prepared)

    assert default_pool.n_free_blocks() == baseline_free_blocks

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix, diags

from sparsetune._backends import SciPyBackend


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

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

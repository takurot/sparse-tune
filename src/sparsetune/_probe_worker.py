"""Isolated backend capability probe."""

from __future__ import annotations

import argparse

import numpy as np
from scipy.sparse import csr_matrix  # type: ignore[import-untyped]

from ._backends import UnsupportedBackendError, get_backend


def probe(backend_id: str, dtype: str) -> bool:
    """Return whether a tiny native CG solve succeeds."""

    backend = get_backend(backend_id)
    matrix = csr_matrix([[2.0, 0.0], [0.0, 3.0]])
    rhs = np.asarray([2.0, 3.0], dtype=dtype)
    prepared = backend.prepare(matrix, rhs, dtype=dtype)
    try:
        backend.warmup(prepared, rtol=1.0e-6, atol=0.0, max_iter=10)
        native = backend.solve_prepared(
            prepared,
            rtol=1.0e-6,
            atol=0.0,
            max_iter=10,
        )
        backend.synchronize()
        solution = backend.fetch_solution(native)
        return native.info == 0 and bool(np.all(np.isfinite(solution)))
    finally:
        backend.release(prepared)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    args = parser.parse_args()
    try:
        return 0 if probe(args.backend, args.dtype) else 1
    except UnsupportedBackendError:
        return 2
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

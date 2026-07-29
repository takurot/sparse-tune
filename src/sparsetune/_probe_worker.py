"""Isolated backend capability probe."""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
from scipy import __version__ as scipy_version  # type: ignore[import-untyped]
from scipy.sparse import csr_matrix  # type: ignore[import-untyped]

from ._backends import UnsupportedBackendError, get_backend
from ._identity import cpu_identity


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


def identity(backend_id: str) -> dict[str, Any]:
    """Collect backend identity inside the isolated probe process."""

    backend = get_backend(backend_id)
    if backend_id == "scipy:cpu":
        return {
            "backend": backend_id,
            "kind": "cpu",
            "scipy_version": scipy_version,
            **cpu_identity(),
        }

    cupy = backend._cp  # type: ignore[attr-defined]
    device_index = int(backend_id.removeprefix("cupy:cuda:"))
    runtime = cupy.cuda.runtime
    properties = runtime.getDeviceProperties(device_index)
    model = properties.get("name") or properties.get(b"name")
    if isinstance(model, bytes):
        model = model.decode("utf-8", errors="replace")
    uuid = properties.get("uuid") or properties.get(b"uuid")
    if uuid is None and hasattr(runtime, "deviceGetUuid"):
        uuid = runtime.deviceGetUuid(device_index)
    if isinstance(uuid, bytes):
        uuid = uuid.hex()
    if not model or not uuid:
        raise ValueError("CUDA identity is incomplete")
    return {
        "backend": backend_id,
        "kind": "cuda",
        "gpu_uuid": str(uuid),
        "gpu_model": str(model),
        "cuda_driver": int(runtime.driverGetVersion()),
        "cuda_runtime": int(runtime.runtimeGetVersion()),
        "cupy_version": str(cupy.__version__),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    args = parser.parse_args()
    try:
        if not probe(args.backend, args.dtype):
            return 1
        print(json.dumps(identity(args.backend), sort_keys=True))
        return 0
    except UnsupportedBackendError:
        return 2
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

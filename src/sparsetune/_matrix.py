"""Matrix Market loading and deterministic CSR canonicalization."""

from __future__ import annotations

import hashlib
from pathlib import Path
import struct
from typing import Any

import numpy as np
from scipy.io import mmread  # type: ignore[import-untyped]
from scipy.sparse import (  # type: ignore[import-untyped]
    csr_matrix,
    issparse,
    save_npz,
)

from ._types import CanonicalMatrix, MatrixInput


_SUPPORTED_FIELDS = {"real", "integer"}
_SUPPORTED_SYMMETRIES = {"general", "symmetric", "skew-symmetric"}
_SUPPORTED_DTYPES: dict[str, np.dtype[Any]] = {
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
}
_FINGERPRINT_VERSION = b"sparsetune-csr-v1"


def _validate_matrix_market_header(path: Path) -> None:
    with path.open("rb") as matrix_file:
        header = (
            matrix_file.readline().decode("ascii", errors="replace").strip().split()
        )

    if len(header) != 5 or header[0].lower() != "%%matrixmarket":
        raise ValueError("input must have a valid Matrix Market header")

    _, object_type, storage, field, symmetry = (item.lower() for item in header)
    if object_type != "matrix":
        raise ValueError("Matrix Market object must be a matrix")
    if storage != "coordinate":
        raise ValueError("Matrix Market storage must be coordinate")
    if field not in _SUPPORTED_FIELDS:
        raise ValueError("Matrix Market field must be real or integer")
    if symmetry not in _SUPPORTED_SYMMETRIES:
        raise ValueError(
            "Matrix Market symmetry must be general, symmetric, or skew-symmetric"
        )


def _as_sparse(matrix: MatrixInput) -> Any:
    if isinstance(matrix, CanonicalMatrix):
        return csr_matrix(
            (matrix.data, matrix.indices, matrix.indptr),
            shape=matrix.shape,
            copy=True,
        )
    if issparse(matrix):
        return matrix
    raise TypeError(
        "matrix input must be a path, CanonicalMatrix, or SciPy sparse matrix"
    )


def _validate_source_dtype(dtype: np.dtype[Any]) -> None:
    if not (np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.integer)):
        raise ValueError("matrix dtype must be a real floating-point or integer type")


def _as_int32(values: np.ndarray[Any, Any], name: str) -> np.ndarray[Any, Any]:
    limits = np.iinfo(np.int32)
    if values.size and (values.min() < limits.min or values.max() > limits.max):
        raise ValueError(f"matrix {name} exceed the supported int32 range")
    return np.asarray(values, dtype=np.int32).copy()


def _canonical_arrays(
    matrix: Any,
    dtype: np.dtype[Any],
    *,
    require_square: bool = True,
) -> tuple[
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    tuple[int, int],
]:
    if len(matrix.shape) != 2 or (
        require_square and matrix.shape[0] != matrix.shape[1]
    ):
        raise ValueError("matrix must be square")

    _validate_source_dtype(np.dtype(matrix.dtype))
    csr = matrix.tocsr(copy=True)
    csr.sum_duplicates()
    csr.sort_indices()

    data = np.asarray(csr.data, dtype=dtype).copy()
    if not np.all(np.isfinite(data)):
        raise ValueError("matrix values must be finite")

    indices = _as_int32(csr.indices, "column indices")
    indptr = _as_int32(csr.indptr, "row pointers")
    shape = (int(csr.shape[0]), int(csr.shape[1]))
    return data, indices, indptr, shape


def _update_fingerprint(
    digest: Any,
    label: bytes,
    values: np.ndarray[Any, Any],
) -> None:
    payload = values.tobytes(order="C")
    digest.update(label)
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def _fingerprint_arrays(
    data: np.ndarray[Any, Any],
    indices: np.ndarray[Any, Any],
    indptr: np.ndarray[Any, Any],
    shape: tuple[int, int],
) -> str:
    digest = hashlib.sha256()
    digest.update(_FINGERPRINT_VERSION)
    digest.update(b"\0byte-order:little\0")
    _update_fingerprint(digest, b"shape", np.asarray(shape, dtype="<i8"))
    _update_fingerprint(digest, b"data", np.asarray(data, dtype="<f8"))
    _update_fingerprint(digest, b"indices", np.asarray(indices, dtype="<i4"))
    _update_fingerprint(digest, b"indptr", np.asarray(indptr, dtype="<i4"))
    return f"sha256:{digest.hexdigest()}"


def fingerprint_csr(matrix: MatrixInput) -> str:
    """Return the v1 fingerprint of a matrix's canonical float64 CSR form."""

    if isinstance(matrix, (str, Path)):
        return load_matrix(matrix).fingerprint

    sparse = _as_sparse(matrix)
    data, indices, indptr, shape = _canonical_arrays(
        sparse,
        _SUPPORTED_DTYPES["float64"],
        require_square=False,
    )
    return _fingerprint_arrays(data, indices, indptr, shape)


def _canonicalize_sparse(
    matrix: Any,
    dtype: np.dtype[Any],
    *,
    require_square: bool = True,
) -> CanonicalMatrix:
    normalized_data, indices, indptr, shape = _canonical_arrays(
        matrix,
        _SUPPORTED_DTYPES["float64"],
        require_square=require_square,
    )
    data = np.asarray(normalized_data, dtype=dtype).copy()
    if not np.all(np.isfinite(data)):
        raise ValueError(f"matrix values cannot be represented as {dtype.name}")
    return CanonicalMatrix(
        data=data,
        indices=indices,
        indptr=indptr,
        shape=shape,
        fingerprint=_fingerprint_arrays(
            normalized_data,
            indices,
            indptr,
            shape,
        ),
    )


def _load_matrix(path: str | Path, *, require_square: bool) -> CanonicalMatrix:
    matrix_path = Path(path)
    _validate_matrix_market_header(matrix_path)
    return _canonicalize_sparse(
        mmread(matrix_path),
        _SUPPORTED_DTYPES["float64"],
        require_square=require_square,
    )


def load_matrix(path: str | Path) -> CanonicalMatrix:
    """Load a supported square Matrix Market file into canonical float64 CSR form."""

    return _load_matrix(path, require_square=True)


def ensure_canonical_matrix(matrix: MatrixInput) -> CanonicalMatrix:
    """Normalize every supported matrix input into canonical CPU CSR arrays."""

    if isinstance(matrix, (str, Path)):
        return load_matrix(matrix)

    sparse = _as_sparse(matrix)
    dtype = (
        _SUPPORTED_DTYPES["float32"]
        if np.dtype(sparse.dtype) == _SUPPORTED_DTYPES["float32"]
        else _SUPPORTED_DTYPES["float64"]
    )
    return _canonicalize_sparse(sparse, dtype)


def canonicalize_matrix(
    matrix: MatrixInput,
    dtype_str: str,
    work_dir: str | Path,
) -> tuple[Path, CanonicalMatrix]:
    """Canonicalize a matrix and save the worker-ready CSR as an NPZ file."""

    try:
        dtype = _SUPPORTED_DTYPES[dtype_str]
    except KeyError as error:
        raise ValueError("dtype must be 'float32' or 'float64'") from error

    if isinstance(matrix, (str, Path)):
        sparse = _as_sparse(load_matrix(matrix))
    else:
        sparse = _as_sparse(matrix)
    canonical = _canonicalize_sparse(sparse, dtype)

    output_dir = Path(work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = canonical.fingerprint.removeprefix("sha256:")
    npz_path = output_dir / f"sparsetune_{digest[:16]}_{dtype_str}.npz"
    save_npz(npz_path, _as_sparse(canonical), compressed=False)
    return npz_path, canonical

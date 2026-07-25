"""Sparse-matrix structural inspection and CG eligibility screening."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, issparse  # type: ignore[import-untyped]
from scipy.sparse.linalg import norm as sparse_norm  # type: ignore[import-untyped]

from ._matrix import fingerprint_csr, load_matrix
from ._types import CanonicalMatrix, MatrixInfo, MatrixInput


_SYMMETRY_TOLERANCE = 1.0e-10


def _as_csr(matrix: MatrixInput) -> Any:
    if isinstance(matrix, (str, Path)):
        matrix = load_matrix(matrix)
    if isinstance(matrix, CanonicalMatrix):
        return csr_matrix(
            (matrix.data, matrix.indices, matrix.indptr),
            shape=matrix.shape,
            copy=True,
        )
    if not issparse(matrix):
        raise TypeError(
            "matrix input must be a path, CanonicalMatrix, or SciPy sparse matrix"
        )

    result = matrix.tocsr(copy=True)
    result.sum_duplicates()
    result.sort_indices()
    return result


def _symmetry_ratio(matrix: Any) -> float:
    if matrix.shape[0] != matrix.shape[1]:
        return 0.0

    matrix_norm = float(sparse_norm(matrix))
    difference_norm = float(sparse_norm(matrix - matrix.transpose()))
    if matrix_norm == 0.0:
        return 1.0
    return max(0.0, 1.0 - difference_norm / matrix_norm)


def _diagonal_sign(diagonal: np.ndarray[Any, Any]) -> str:
    if diagonal.size == 0 or np.all(diagonal == 0):
        return "zeros"
    if np.all(diagonal > 0):
        return "all_positive"
    if np.all(diagonal < 0):
        return "all_negative"
    return "mixed"


def diagnose_matrix(
    matrix: MatrixInput,
    *,
    path: str | Path | None = None,
) -> MatrixInfo:
    """Return structural metadata and the documented SPD screening status."""

    if path is None and isinstance(matrix, (str, Path)):
        path = matrix
    csr = _as_csr(matrix)
    is_square = csr.shape[0] == csr.shape[1]
    symmetry_ratio = _symmetry_ratio(csr)
    diagonal_sign = _diagonal_sign(np.asarray(csr.diagonal()))
    symmetric = is_square and (1.0 - symmetry_ratio) <= _SYMMETRY_TOLERANCE

    if not is_square or not symmetric:
        spd_status = "failed"
    elif diagonal_sign == "all_positive":
        spd_status = "screen_passed"
    else:
        spd_status = "unknown"

    rows, columns = (int(csr.shape[0]), int(csr.shape[1]))
    return MatrixInfo(
        path="" if path is None else str(path),
        shape=(rows, columns),
        nnz=int(csr.nnz),
        density=float(csr.nnz / (rows * columns)) if rows and columns else 0.0,
        is_square=is_square,
        symmetry_ratio=symmetry_ratio,
        diagonal_sign=diagonal_sign,
        spd_status=spd_status,
        fingerprint=fingerprint_csr(csr),
    )


def is_cg_eligible(
    info: MatrixInfo,
    *,
    assume_spd: bool = False,
) -> bool:
    """Return whether the SPD screening contract permits a CG run."""

    if info.spd_status == "screen_passed":
        return True
    if info.spd_status == "unknown":
        return assume_spd
    return False

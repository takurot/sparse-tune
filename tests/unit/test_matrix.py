from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import coo_matrix, csr_matrix, load_npz

from sparsetune import (
    CanonicalMatrix,
    canonicalize_matrix,
    ensure_canonical_matrix,
    fingerprint_csr,
    load_matrix,
)


FIXTURES = Path(__file__).parents[1] / "fixtures"


def _as_csr(matrix: CanonicalMatrix) -> csr_matrix:
    return csr_matrix(
        (matrix.data, matrix.indices, matrix.indptr),
        shape=matrix.shape,
    )


def test_load_matrix_does_not_expand_symmetric_input_twice() -> None:
    matrix = load_matrix(FIXTURES / "small_symmetric.mtx")

    np.testing.assert_array_equal(
        _as_csr(matrix).toarray(),
        np.array(
            [
                [4.0, 1.0, 0.0],
                [1.0, 3.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        ),
    )
    assert matrix.nnz == 5


def test_load_matrix_sums_duplicates_and_sorts_csr_indices() -> None:
    matrix = load_matrix(FIXTURES / "duplicates_unsorted.mtx")
    csr = _as_csr(matrix)

    np.testing.assert_array_equal(
        csr.toarray(),
        np.array(
            [
                [1.0, 0.0, 4.0],
                [0.0, 2.0, 0.0],
                [5.0, 0.0, 0.0],
            ]
        ),
    )
    assert csr.has_canonical_format
    assert matrix.indices.dtype == np.int32
    assert matrix.indptr.dtype == np.int32


def test_equivalent_inputs_have_the_same_versioned_fingerprint() -> None:
    with_duplicates = coo_matrix(
        (
            np.array([1, 2, 4]),
            (np.array([0, 0, 1]), np.array([0, 0, 1])),
        ),
        shape=(2, 2),
    )
    canonical_float32 = csr_matrix(np.array([[3, 0], [0, 4]], dtype=np.float32))

    first = fingerprint_csr(with_duplicates)
    second = fingerprint_csr(canonical_float32)

    assert first == second
    assert first.startswith("sha256:")


def test_fingerprint_changes_with_shape_values_and_sparsity() -> None:
    base = csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))

    assert fingerprint_csr(base) != fingerprint_csr(
        csr_matrix(np.diag([1.0, 2.0, 0.0]))
    )
    assert fingerprint_csr(base) != fingerprint_csr(
        csr_matrix(np.array([[1.0, 0.0], [0.0, 3.0]]))
    )
    assert fingerprint_csr(base) != fingerprint_csr(
        csr_matrix(np.array([[1.0, 4.0], [0.0, 2.0]]))
    )


def test_supported_input_forms_produce_equivalent_canonical_csr() -> None:
    from_path = ensure_canonical_matrix(FIXTURES / "small_symmetric.mtx")
    from_sparse = ensure_canonical_matrix(_as_csr(from_path).tocsc())
    from_canonical = ensure_canonical_matrix(from_path)

    assert (
        fingerprint_csr(str(FIXTURES / "small_symmetric.mtx")) == from_path.fingerprint
    )
    for matrix in (from_sparse, from_canonical):
        np.testing.assert_array_equal(matrix.data, from_path.data)
        np.testing.assert_array_equal(matrix.indices, from_path.indices)
        np.testing.assert_array_equal(matrix.indptr, from_path.indptr)
        assert matrix.shape == from_path.shape
        assert matrix.fingerprint == from_path.fingerprint


def test_canonicalize_matrix_applies_dtype_and_saves_npz(tmp_path: Path) -> None:
    npz_path, matrix = canonicalize_matrix(
        FIXTURES / "small_symmetric.mtx",
        "float32",
        tmp_path,
    )

    assert npz_path.parent == tmp_path
    assert npz_path.is_file()
    assert matrix.dtype == "float32"
    saved = load_npz(npz_path)
    assert saved.dtype == np.float32
    np.testing.assert_array_equal(saved.toarray(), _as_csr(matrix).toarray())


def test_output_dtype_does_not_change_the_source_matrix_fingerprint(
    tmp_path: Path,
) -> None:
    matrix_path = tmp_path / "fractional.mtx"
    matrix_path.write_text(
        "%%MatrixMarket matrix coordinate real general\n1 1 1\n1 1 0.1\n",
        encoding="ascii",
    )

    _, float32_matrix = canonicalize_matrix(matrix_path, "float32", tmp_path)
    _, float64_matrix = canonicalize_matrix(matrix_path, "float64", tmp_path)

    assert float32_matrix.fingerprint == float64_matrix.fingerprint


@pytest.mark.parametrize("dtype", ["float16", "int64", "complex128", "FLOAT64"])
def test_canonicalize_matrix_rejects_unsupported_output_dtype(
    tmp_path: Path,
    dtype: str,
) -> None:
    with pytest.raises(ValueError, match="dtype must be 'float32' or 'float64'"):
        canonicalize_matrix(
            FIXTURES / "small_symmetric.mtx",
            dtype,
            tmp_path,
        )


@pytest.mark.parametrize(
    ("header", "message"),
    [
        ("%%MatrixMarket matrix array real general", "coordinate"),
        ("%%MatrixMarket matrix coordinate complex general", "real or integer"),
        ("%%MatrixMarket matrix coordinate pattern general", "real or integer"),
        ("%%MatrixMarket matrix coordinate real hermitian", "symmetry"),
    ],
)
def test_load_matrix_rejects_unsupported_matrix_market_variants(
    tmp_path: Path,
    header: str,
    message: str,
) -> None:
    matrix_path = tmp_path / "unsupported.mtx"
    matrix_path.write_text(f"{header}\n1 1 1\n1 1 1\n", encoding="ascii")

    with pytest.raises(ValueError, match=message):
        load_matrix(matrix_path)


def test_matrix_inputs_must_be_square_and_finite(tmp_path: Path) -> None:
    nonsquare = tmp_path / "nonsquare.mtx"
    nonsquare.write_text(
        "%%MatrixMarket matrix coordinate real general\n2 3 1\n1 1 1\n",
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="square"):
        load_matrix(nonsquare)
    with pytest.raises(ValueError, match="finite"):
        ensure_canonical_matrix(csr_matrix([[1.0, np.nan], [0.0, 1.0]]))

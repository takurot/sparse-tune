from pathlib import Path

import numpy as np
import pytest
from scipy.io import mmread
from scipy.sparse import csr_matrix

from sparsetune import (
    benchmark,
    diagnose_matrix,
    fingerprint_csr,
    inspect as inspect_matrix,
    load_matrix,
    solve,
)
from sparsetune._inspect import is_cg_eligible


FIXTURES = Path(__file__).parents[1] / "fixtures"


def _as_csr(path: Path) -> csr_matrix:
    return mmread(path).tocsr()


def test_symmetric_positive_diagonal_passes_spd_screen() -> None:
    matrix = load_matrix(FIXTURES / "small_symmetric.mtx")

    info = diagnose_matrix(matrix, path=FIXTURES / "small_symmetric.mtx")

    assert info.path == str(FIXTURES / "small_symmetric.mtx")
    assert info.shape == (3, 3)
    assert info.nnz == 5
    assert info.density == pytest.approx(5 / 9)
    assert info.is_square is True
    assert info.symmetry_ratio == pytest.approx(1.0)
    assert info.diagonal_sign == "all_positive"
    assert info.spd_status == "screen_passed"
    assert info.fingerprint == matrix.fingerprint
    assert is_cg_eligible(info) is True


def test_nonsquare_and_nonsymmetric_matrices_fail_spd_screen() -> None:
    nonsquare = diagnose_matrix(_as_csr(FIXTURES / "small_nonsquare.mtx"))
    nonsymmetric = diagnose_matrix(csr_matrix([[2.0, 1.0], [0.0, 2.0]]))

    assert nonsquare.is_square is False
    assert nonsquare.symmetry_ratio == 0.0
    assert nonsquare.spd_status == "failed"
    assert nonsquare.fingerprint == fingerprint_csr(
        _as_csr(FIXTURES / "small_nonsquare.mtx")
    )
    assert nonsymmetric.is_square is True
    assert nonsymmetric.symmetry_ratio < 1.0
    assert nonsymmetric.spd_status == "failed"
    assert is_cg_eligible(nonsquare, assume_spd=True) is False
    assert is_cg_eligible(nonsymmetric, assume_spd=True) is False


def test_public_inspect_accepts_nonsquare_matrix_market_path() -> None:
    path = FIXTURES / "small_nonsquare.mtx"

    info = inspect_matrix(path)

    assert info.path == str(path)
    assert info.shape == (2, 3)
    assert info.is_square is False
    assert info.spd_status == "failed"
    assert info.fingerprint == fingerprint_csr(_as_csr(path))


def test_public_inspect_is_consistent_across_supported_inputs() -> None:
    path = FIXTURES / "small_symmetric.mtx"
    canonical = load_matrix(path)
    sparse = _as_csr(path)

    from_path = inspect_matrix(path)
    from_canonical = inspect_matrix(canonical)
    from_sparse = inspect_matrix(sparse)

    assert from_path.path == str(path)
    assert from_path.shape == from_canonical.shape == from_sparse.shape
    assert from_path.nnz == from_canonical.nnz == from_sparse.nnz
    assert from_path.spd_status == from_canonical.spd_status == from_sparse.spd_status
    assert (
        from_path.fingerprint == from_canonical.fingerprint == from_sparse.fingerprint
    )


def test_solve_and_benchmark_still_reject_nonsquare_paths() -> None:
    path = FIXTURES / "small_nonsquare.mtx"

    with pytest.raises(ValueError, match="not eligible"):
        benchmark(path, backends=["scipy:cpu"], runs=1)
    with pytest.raises(ValueError, match="not eligible"):
        solve(path, backend="scipy:cpu")


def test_positive_diagonal_screening_does_not_prove_positive_definiteness() -> None:
    matrix = _as_csr(FIXTURES / "indefinite.mtx")

    assert np.linalg.eigvalsh(matrix.toarray()).min() < 0
    assert diagnose_matrix(matrix).spd_status == "screen_passed"


@pytest.mark.parametrize(
    ("diagonal", "expected"),
    [
        ([1.0, 2.0], "all_positive"),
        ([-1.0, -2.0], "all_negative"),
        ([0.0, 0.0], "zeros"),
        ([1.0, 0.0], "mixed"),
        ([1.0, -1.0], "mixed"),
    ],
)
def test_diagonal_sign_classification(
    diagonal: list[float],
    expected: str,
) -> None:
    info = diagnose_matrix(csr_matrix(np.diag(diagonal)))

    assert info.diagonal_sign == expected
    assert info.spd_status == (
        "screen_passed" if expected == "all_positive" else "unknown"
    )


def test_symmetry_uses_documented_relative_tolerance() -> None:
    within_tolerance = csr_matrix([[2.0, 1.0], [1.0 + 1.0e-11, 2.0]])
    outside_tolerance = csr_matrix([[2.0, 1.0], [1.0 + 1.0e-8, 2.0]])

    assert diagnose_matrix(within_tolerance).spd_status == "screen_passed"
    assert diagnose_matrix(outside_tolerance).spd_status == "failed"


def test_only_unknown_status_can_be_overridden() -> None:
    unknown = diagnose_matrix(csr_matrix(np.diag([0.0, 1.0])))

    assert unknown.spd_status == "unknown"
    assert is_cg_eligible(unknown) is False
    assert is_cg_eligible(unknown, assume_spd=True) is True

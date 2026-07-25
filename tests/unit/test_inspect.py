from pathlib import Path

import numpy as np
import pytest
from scipy.io import mmread
from scipy.sparse import csr_matrix

from sparsetune import (
    diagnose_matrix,
    fingerprint_csr,
    load_matrix,
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

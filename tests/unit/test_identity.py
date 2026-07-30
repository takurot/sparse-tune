from __future__ import annotations

import numpy as np
import pytest

from sparsetune._identity import _blas_implementation


@pytest.mark.parametrize(
    "fallback",
    [
        lambda _name: object(),
        lambda _name: (_ for _ in ()).throw(RuntimeError("broken config")),
    ],
)
def test_blas_identity_tolerates_unavailable_numpy_fallback(
    monkeypatch: pytest.MonkeyPatch,
    fallback: object,
) -> None:
    monkeypatch.setattr(np.__config__, "CONFIG", {}, raising=False)
    monkeypatch.setattr(np.__config__, "get_info", fallback, raising=False)

    assert _blas_implementation() is None

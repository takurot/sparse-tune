from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.sparse import diags

from sparsetune import canonicalize_matrix
from sparsetune._backends import UnsupportedBackendError
from sparsetune._runner import canonicalize_rhs
from sparsetune._types import SolveStatus
from sparsetune._worker import run_worker, status_for_exception


class OutOfMemoryError(RuntimeError):
    pass


def test_worker_exception_statuses_are_structured() -> None:
    assert status_for_exception(MemoryError()) is SolveStatus.OOM
    assert status_for_exception(OutOfMemoryError()) is SolveStatus.OOM
    assert (
        status_for_exception(UnsupportedBackendError("missing"))
        is SolveStatus.UNSUPPORTED
    )
    assert (
        status_for_exception(RuntimeError("unexpected")) is SolveStatus.INTERNAL_ERROR
    )


def test_run_worker_executes_requested_measurement_modes(tmp_path: Path) -> None:
    matrix = diags(
        [-np.ones(4), 2.0 * np.ones(5), -np.ones(4)],
        offsets=[-1, 0, 1],
        format="csr",
    )
    matrix_path, canonical = canonicalize_matrix(matrix, "float64", tmp_path)
    rhs_path, _ = canonicalize_rhs(
        None,
        canonical,
        dtype_str="float64",
        work_dir=tmp_path,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "dtype": "float64",
                "rtol": 1.0e-6,
                "atol": 0.0,
                "max_iter": 50,
                "runs": 2,
                "measure": ["end-to-end", "steady-state"],
            }
        ),
        encoding="utf-8",
    )

    result = run_worker(
        "scipy:cpu",
        matrix_path,
        rhs_path,
        config_path,
    )

    assert result.status is SolveStatus.CONVERGED
    assert len(result.samples) == 4
    assert result.setup_seconds > 0.0
    assert result.solve_seconds > 0.0

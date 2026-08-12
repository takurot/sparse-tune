from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import diags

from sparsetune import canonicalize_matrix
from sparsetune._runner import canonicalize_rhs, run_backend_in_subprocess
from sparsetune._types import SolveStatus


def test_scipy_worker_returns_normal_subprocess_result(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).parents[2]
    monkeypatch.setenv("PYTHONPATH", str(project_root / "src"))
    matrix = diags(
        [-np.ones(9), 2.0 * np.ones(10), -np.ones(9)],
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

    result = run_backend_in_subprocess(
        "scipy:cpu",
        matrix_path,
        rhs_path,
        {
            "dtype": "float64",
            "rtol": 1.0e-6,
            "atol": 0.0,
            "max_iter": 100,
            "runs": 2,
            "measure": ["end-to-end", "steady-state"],
        },
        timeout=10.0,
    )

    assert result.status is SolveStatus.CONVERGED
    assert result.backend == "scipy:cpu"
    assert result.iterations > 0
    assert result.residual_norm <= result.convergence_threshold
    assert {sample.measure for sample in result.samples} == {
        "end-to-end",
        "steady-state",
    }


def test_worker_failures_remain_structured_and_do_not_echo_config(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).parents[2]
    monkeypatch.setenv("PYTHONPATH", str(project_root / "src"))
    matrix_path, canonical = canonicalize_matrix(
        diags([1.0, 2.0], format="csr"),
        "float64",
        tmp_path,
    )
    rhs_path, _ = canonicalize_rhs(
        None,
        canonical,
        dtype_str="float64",
        work_dir=tmp_path,
    )
    config = {
        "dtype": "float64",
        "rtol": 1.0e-6,
        "atol": 0.0,
        "max_iter": 20,
        "runs": 1,
        "measure": ["end-to-end"],
    }

    unsupported = run_backend_in_subprocess(
        "unknown:backend",
        matrix_path,
        rhs_path,
        config,
        timeout=10.0,
    )
    secret = "sentinel-config-secret"
    invalid_config = run_backend_in_subprocess(
        "scipy:cpu",
        matrix_path,
        rhs_path,
        {**config, "API_TOKEN": secret},
        timeout=10.0,
    )

    assert unsupported.status is SolveStatus.UNSUPPORTED
    assert invalid_config.status is SolveStatus.INTERNAL_ERROR
    assert invalid_config.error is not None
    assert secret not in invalid_config.error
    assert secret not in invalid_config.to_json()

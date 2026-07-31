"""Contract tests for the published-package Colab GPU validation notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[2]
NOTEBOOK = ROOT / "notebooks" / "colab_gpu_validation.ipynb"


def _load_notebook() -> dict[str, object]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _sources(notebook: dict[str, object], cell_type: str) -> list[str]:
    cells = notebook["cells"]
    assert isinstance(cells, list)
    return [
        "".join(cell["source"])
        for cell in cells
        if isinstance(cell, dict) and cell.get("cell_type") == cell_type
    ]


def test_colab_notebook_is_valid_compilable_and_committed_without_outputs() -> None:
    notebook = _load_notebook()

    assert notebook["nbformat"] == 4
    code_cells = _sources(notebook, "code")
    assert code_cells
    for index, source in enumerate(code_cells):
        compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec")

    cells = notebook["cells"]
    assert isinstance(cells, list)
    for cell in cells:
        assert isinstance(cell, dict)
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []


def test_colab_notebook_covers_the_gpu_validation_contract() -> None:
    notebook = _load_notebook()
    all_source = "\n".join(_sources(notebook, "markdown") + _sources(notebook, "code"))

    required_fragments = (
        "sparsetune==0.1.11",
        "cupy-cuda12x",
        "cupy-cuda13x",
        "nvidia-smi",
        "sparsetune doctor",
        "scipy:cpu",
        "cupy:cuda:0",
        "end-to-end",
        "steady-state",
        "relative_residual",
        "convergence_threshold",
        "break_even_solves",
        "matrix_fingerprint",
        "colab_runtime_date",
        "cuda_version",
        "MATRIX_SIZES",
        '"small": 10_000',
        '"medium": 100_000',
        '"large": 1_000_000',
        '"dtype": DTYPE',
        '"runs": RUNS',
        '"rtol": RTOL',
        '"max_iterations": MAX_ITER',
        "ENABLE_MEMORY_PRESSURE = False",
        "colab_gpu_validation_results.json",
    )
    for fragment in required_fragments:
        assert fragment in all_source

    assert "git clone" not in all_source
    assert "drive.mount" not in all_source


def test_colab_notebook_can_validate_a_release_candidate_wheel() -> None:
    notebook = _load_notebook()
    all_source = "\n".join(_sources(notebook, "markdown") + _sources(notebook, "code"))

    for fragment in (
        'PACKAGE_SOURCE = "release-candidate-wheel"',
        "files.upload()",
        "wheel_sha256",
        "sparsetune-0.1.11-py3-none-any.whl",
        'PACKAGE_SOURCE == "pypi"',
        '"package_source": PACKAGE_SOURCE',
    ):
        assert fragment in all_source


def test_readme_links_to_colab_notebook() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "notebooks/colab_gpu_validation.ipynb" in readme
    assert "colab.research.google.com/github/takurot/sparse-tune/blob/main/" in readme


def test_tracked_gpu_validation_summary_records_scope_and_reproduction() -> None:
    summary_path = ROOT / "docs" / "GPU_VALIDATION.md"
    summary = summary_path.read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    validation = (ROOT / "docs" / "VALIDATION.md").read_text(encoding="utf-8")

    for fragment in (
        "sparsetune 0.1.11",
        "2026-07-31",
        "Tesla T4",
        "Python 3.12.13",
        "SciPy 1.16.3",
        "CuPy 14.1.1",
        "float64",
        "rtol=1e-6",
        "relative residual",
        "single-session functional validation",
        "physical GPU OOM",
        "multi-GPU",
        "CUDA 13",
        "cross-device",
        "notebooks/colab_gpu_validation.ipynb",
        "colab_gpu_validation_results.json",
        "9ebea9f1e3824e5d57302c3a53cbd2606781b8b819cf9a3319cd1d1765735cfc",
        "7a111ddc30aee0d7a2b846e0eeb4e10db0546dd2115689baf52a6a836464ace3",
        "de0479e4ca23b2cdd125a601a4c18ea2cf654c04e7444b280b9b24e0aa91d478",
        'package_source="pypi"',
        "production PyPI package",
    ):
        assert fragment in summary

    assert "sparsetune 0.1.0" not in summary

    assert "docs/GPU_VALIDATION.md" in readme
    assert "2026-07-31" in readme
    assert "GPU_VALIDATION.md" in validation


def test_colab_notebook_compares_relative_residual_with_rtol() -> None:
    notebook = _load_notebook()
    code_source = "\n".join(_sources(notebook, "code"))

    assert 'result["relative_residual"] <= RTOL' in code_source
    assert (
        'result["relative_residual"] <= result["convergence_threshold"]'
        not in code_source
    )

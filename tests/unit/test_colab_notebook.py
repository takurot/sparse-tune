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
        "sparsetune==0.1.0",
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


def test_readme_links_to_colab_notebook() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "notebooks/colab_gpu_validation.ipynb" in readme
    assert "colab.research.google.com/github/takurot/sparse-tune/blob/main/" in readme

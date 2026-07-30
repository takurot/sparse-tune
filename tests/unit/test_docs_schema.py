from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
import re

import sparsetune
from sparsetune import MatrixInfo, Recommendation, RunSample, SolverResult

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _parse_json(block: str, source: str) -> dict[str, object]:
    try:
        return json.loads(block)
    except json.JSONDecodeError as error:
        raise AssertionError(f"{source} contains an invalid JSON code fence") from error


def _json_between(
    text: str,
    start: str,
    end: str,
    source: str,
) -> dict[str, object]:
    assert start in text, f"{source} is missing the start sentinel marker"
    assert end in text, f"{source} is missing the end sentinel marker"
    section = text.split(start, 1)[1].split(end, 1)[0]
    payload = re.search(r"```json\s*(.*?)\s*```", section, re.DOTALL)
    assert payload is not None, f"{source} contains no marked JSON code fence"
    return _parse_json(payload.group(1), source)


def test_spec_benchmark_example_matches_live_schema() -> None:
    spec = (_PROJECT_ROOT / "docs" / "SPEC.md").read_text(encoding="utf-8")
    payload = _json_between(
        spec,
        "<!-- benchmark-schema-example:start -->",
        "<!-- benchmark-schema-example:end -->",
        "docs/SPEC.md",
    )

    assert set(payload) == {
        "schema_version",
        "matrix",
        "environment",
        "results",
        "recommendations",
    }
    assert payload["schema_version"] == "1.0"
    assert set(payload["matrix"]) == {  # type: ignore[arg-type]
        *(field.name for field in fields(MatrixInfo)),
        "dtype",
    }
    assert payload["results"], "SPEC.md benchmark example needs a result"
    result = payload["results"][0]  # type: ignore[index]
    assert set(result) == {field.name for field in fields(SolverResult)}
    assert result["samples"], "SPEC.md benchmark example needs a sample"
    assert set(result["samples"][0]) == {  # type: ignore[index]
        field.name for field in fields(RunSample)
    }
    assert all(
        set(recommendation) == {field.name for field in fields(Recommendation)}
        for recommendation in payload["recommendations"].values()  # type: ignore[union-attr]
    )


def test_readme_recommendation_examples_match_live_shape() -> None:
    readme = (_PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    raw_blocks = re.findall(r"```json\s*(.*?)\s*```", readme, re.DOTALL)
    recommendations = [
        _parse_json(block, "README.md") for block in raw_blocks if '"reason"' in block
    ]

    assert len(recommendations) == 2
    recommendation_keys = {field.name for field in fields(Recommendation)}
    assert all(set(example) == recommendation_keys for example in recommendations)


def test_documented_public_api_and_repository_facts_match_runtime() -> None:
    readme = (_PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    spec = (_PROJECT_ROOT / "docs" / "SPEC.md").read_text(encoding="utf-8")
    workflow = (_PROJECT_ROOT / "docs" / "WORKFLOW.md").read_text(encoding="utf-8")
    release_notes = (_PROJECT_ROOT / "RELEASE_NOTES.md").read_text(encoding="utf-8")

    for public_name in (
        "list_backends",
        "load_matrix",
        "inspect",
        "benchmark",
        "tune",
        "solve",
    ):
        assert public_name in sparsetune.__all__
        assert f"`{public_name}()`" in readme

    assert "CanonicalMatrix" in spec
    assert "SparseMatrix" not in spec
    assert "LICENSE                 # Apache 2.0" not in spec
    assert "test_autoselect.py" not in spec
    assert "test_api.py" not in spec
    assert "## 13. プロジェクト構成" not in spec
    assert "## 10. 現在のプロジェクト構成" not in workflow
    assert "Python 3.10-3.14 CPU CI" in release_notes


def test_docs_distinguish_inspection_from_solver_shape_requirements() -> None:
    readme = (_PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    workflow = (_PROJECT_ROOT / "docs" / "WORKFLOW.md").read_text(encoding="utf-8")

    assert "inspect` は非正方行列も受け付け" in readme
    assert "bench`、`tune`、`solve` は正方行列だけ" in readme
    assert "inspect` では非正方行列を診断できる" in workflow

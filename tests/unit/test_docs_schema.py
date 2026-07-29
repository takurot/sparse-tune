from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
import re

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
    section = text.split(start, 1)[1].split(end, 1)[0]
    payload = re.search(r"```json\s*(.*?)\s*```", section, re.DOTALL)
    assert payload is not None
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
    result = payload["results"][0]  # type: ignore[index]
    assert set(result) == {field.name for field in fields(SolverResult)}
    assert set(result["samples"][0]) == {  # type: ignore[index]
        field.name for field in fields(RunSample)
    }
    assert all(
        set(recommendation) == {field.name for field in fields(Recommendation)}
        for recommendation in payload["recommendations"].values()  # type: ignore[union-attr]
    )


def test_readme_recommendation_examples_match_live_shape() -> None:
    readme = (_PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    examples = [
        _parse_json(block, "README.md")
        for block in re.findall(r"```json\s*(.*?)\s*```", readme, re.DOTALL)
    ]
    recommendations = [example for example in examples if "reason" in example]

    assert len(recommendations) == 2
    assert all(
        set(example) == {field.name for field in fields(Recommendation)}
        for example in recommendations
    )

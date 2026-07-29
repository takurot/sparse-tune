from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
import re

from sparsetune import MatrixInfo, Recommendation, RunSample, SolverResult


def _json_between(text: str, start: str, end: str) -> dict[str, object]:
    section = text.split(start, 1)[1].split(end, 1)[0]
    payload = re.search(r"```json\s*(.*?)\s*```", section, re.DOTALL)
    assert payload is not None
    return json.loads(payload.group(1))


def test_spec_benchmark_example_matches_live_schema() -> None:
    spec = Path("docs/SPEC.md").read_text(encoding="utf-8")
    payload = _json_between(
        spec,
        "<!-- benchmark-schema-example:start -->",
        "<!-- benchmark-schema-example:end -->",
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
    readme = Path("README.md").read_text(encoding="utf-8")
    examples = [
        json.loads(block)
        for block in re.findall(r"```json\s*(.*?)\s*```", readme, re.DOTALL)
    ]
    recommendations = [example for example in examples if "reason" in example]

    assert len(recommendations) == 2
    assert all(
        set(example) == {field.name for field in fields(Recommendation)}
        for example in recommendations
    )

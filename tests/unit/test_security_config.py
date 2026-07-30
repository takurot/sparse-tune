from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = PROJECT_ROOT / ".github" / "workflows"
PINNED_ACTION = re.compile(
    r"^\s*-\s+uses:\s+[^@\s]+@[0-9a-f]{40}\s+#\s+\S+",
    re.MULTILINE,
)


def test_all_workflow_actions_are_pinned_with_version_comments() -> None:
    for workflow_path in sorted(WORKFLOW_DIR.glob("*.yml")):
        workflow = workflow_path.read_text(encoding="utf-8")
        uses_lines = [
            line
            for line in workflow.splitlines()
            if line.lstrip().startswith("- uses:")
        ]
        assert uses_lines, f"{workflow_path.name} must contain at least one action"
        assert len(PINNED_ACTION.findall(workflow)) == len(uses_lines), (
            f"{workflow_path.name} contains an unpinned action or a pin without "
            "a version comment"
        )


def test_dependabot_updates_actions_and_python_dependencies() -> None:
    config_path = PROJECT_ROOT / ".github" / "dependabot.yml"
    config = config_path.read_text(encoding="utf-8")

    assert "package-ecosystem: github-actions" in config
    assert "package-ecosystem: pip" in config
    assert config.count("directory: /") == 2


def test_security_workflow_runs_dependency_and_code_scans() -> None:
    workflow_path = WORKFLOW_DIR / "security.yml"
    workflow = workflow_path.read_text(encoding="utf-8")

    assert "pip-audit" in workflow
    assert "github/codeql-action/init@" in workflow
    assert "github/codeql-action/analyze@" in workflow
    assert "security-events: write" in workflow
    assert "schedule:" in workflow

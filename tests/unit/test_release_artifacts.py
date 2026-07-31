from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tarfile
import zipfile

import pytest

from scripts.artifact_version import artifact_version


def _metadata(version: str) -> bytes:
    return (f"Metadata-Version: 2.4\nName: sparsetune\nVersion: {version}\n\n").encode()


def _wheel(path: Path, version: str) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"sparsetune-{version}.dist-info/METADATA",
            _metadata(version),
        )


def _sdist(path: Path, version: str) -> None:
    payload = _metadata(version)
    info = tarfile.TarInfo(f"sparsetune-{version}/PKG-INFO")
    info.size = len(payload)
    nested = tarfile.TarInfo(f"sparsetune-{version}/src/sparsetune.egg-info/PKG-INFO")
    nested.size = len(payload)
    with tarfile.open(path, "w:gz") as archive:
        archive.addfile(info, BytesIO(payload))
        archive.addfile(nested, BytesIO(payload))


def test_artifact_version_reads_matching_wheel_and_sdist(tmp_path: Path) -> None:
    wheel = tmp_path / "sparsetune-0.1.11-py3-none-any.whl"
    sdist = tmp_path / "sparsetune-0.1.11.tar.gz"
    _wheel(wheel, "0.1.11")
    _sdist(sdist, "0.1.11")

    assert artifact_version([wheel, sdist]) == "0.1.11"


def test_artifact_version_rejects_metadata_mismatch(tmp_path: Path) -> None:
    wheel = tmp_path / "sparsetune-0.1.11-py3-none-any.whl"
    sdist = tmp_path / "sparsetune-0.1.12.tar.gz"
    _wheel(wheel, "0.1.11")
    _sdist(sdist, "0.1.12")

    with pytest.raises(ValueError, match="versions do not match"):
        artifact_version([wheel, sdist])


def test_artifact_version_requires_one_wheel_and_one_sdist(tmp_path: Path) -> None:
    wheel = tmp_path / "sparsetune-0.1.11-py3-none-any.whl"
    _wheel(wheel, "0.1.11")

    with pytest.raises(ValueError, match="exactly one wheel and one sdist"):
        artifact_version([wheel])


def test_python_policy_and_workflows_stay_aligned() -> None:
    root = Path(__file__).parents[2]
    project = (root / "pyproject.toml").read_text(encoding="utf-8")
    test_workflow = (root / ".github/workflows/test.yml").read_text(encoding="utf-8")
    testpypi_workflow = (root / ".github/workflows/testpypi.yml").read_text(
        encoding="utf-8"
    )

    assert 'requires-python = ">=3.10,<3.15"' in project
    assert 'python-version: ["3.10", "3.11", "3.12", "3.13", "3.14"]' in test_workflow
    assert "scripts/artifact_version.py dist/*" in testpypi_workflow
    assert 'version="$(python scripts/artifact_version.py dist/*)"' in testpypi_workflow
    assert 'echo "version=$(python' not in testpypi_workflow
    assert '"sparsetune==${VERSION}"' in testpypi_workflow
    assert "VERSION: ${{ steps.package.outputs.version }}" in testpypi_workflow
    assert "sparsetune==0.1.0" not in testpypi_workflow


def test_v011_release_metadata_is_consistent() -> None:
    root = Path(__file__).parents[2]
    project = (root / "pyproject.toml").read_text(encoding="utf-8")
    package = (root / "src/sparsetune/__init__.py").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    release_notes = (root / "RELEASE_NOTES.md").read_text(encoding="utf-8")

    assert 'version = "0.1.11"' in project
    assert '__version__ = "0.1.11"' in package
    assert "**開発状況:** v0.1.11" in readme
    assert 'pip install "sparsetune==0.1.11"' in readme
    assert "sparsetune==0.1.0" not in readme
    assert "| 項目 | v0.1.11 |" in readme
    assert "# sparsetune 0.1.11" in release_notes

    for fragment in (
        "Compatibility and correctness",
        "Security and CI",
        "Validation",
        "Known limitations",
        "docs/VALIDATION.md",
        "docs/GPU_VALIDATION.md",
        "#25",
        "#37",
    ):
        assert fragment in release_notes


def test_v011_validation_report_records_release_gates() -> None:
    root = Path(__file__).parents[2]
    report = (root / "docs" / "VALIDATION.md").read_text(encoding="utf-8")

    assert "# v0.1.11 release-candidate validation report" in report
    assert "**Version:** `0.1.11`" in report
    assert "**Release commit:** `356062dac4035c79ac7d74a678111d034e19f22e`" in report
    assert "**Overall result:** passed" in report
    assert "7a111ddc30aee0d7a2b846e0eeb4e10db0546dd2115689baf52a6a836464ace3" in report

    for fragment in (
        "Python 3.10",
        "Python 3.11",
        "Python 3.12",
        "Python 3.13",
        "Python 3.14",
        "NumPy 1.24.0 / SciPy 1.10.0",
        "built wheel",
        "sparsetune doctor",
        "sparsetune inspect",
        "sparsetune bench",
        "sparsetune tune",
        "sparsetune solve",
        "malformed worker output",
        "unsupported",
        "profile mismatch",
        "pip-audit",
        "Tesla T4",
        "not a stable performance baseline",
    ):
        assert fragment in report


def test_v011_publication_evidence_is_complete() -> None:
    root = Path(__file__).parents[2]
    evidence = (root / "docs" / "RELEASE_EVIDENCE.md").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    validation = (root / "docs" / "VALIDATION.md").read_text(encoding="utf-8")

    for fragment in (
        "a73f75b4c49f984f7a2e344a29fc647f288498fe",
        "https://github.com/takurot/sparse-tune/releases/tag/v0.1.11",
        "https://github.com/takurot/sparse-tune/actions/runs/30633795456",
        "https://github.com/takurot/sparse-tune/actions/runs/30633927610",
        "170eeb839180b666e56da3b40cee8077956a2c2307e863a1a303442582120bd2",
        "4829164f3c57128969b577e2095d0b4bd284115180c583922d57ac786cc03dd8",
        "1f0a5fe179d698adfc91e1b28cc626d0739a4fec217b6791693066b73510c72b",
        "ecb4aed86b98482966a467b83873b9d74366305775d260e5c70149a2fb608988",
        "OIDC Trusted Publishing",
        "testpypi",
        "pypi",
        "pip check",
        "Apple M1",
        "Tesla T4",
        "de0479e4ca23b2cdd125a601a4c18ea2cf654c04e7444b280b9b24e0aa91d478",
        "post-publication production-index GPU smoke",
        'package_source="pypi"',
    ):
        assert fragment in evidence

    assert "docs/RELEASE_EVIDENCE.md" in readme
    assert "docs/RELEASE_EVIDENCE.md" in validation

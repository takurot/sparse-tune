from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tarfile
import tomllib
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


def test_python_policy_and_workflows_stay_aligned() -> None:
    root = Path(__file__).parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    test_workflow = (root / ".github/workflows/test.yml").read_text(encoding="utf-8")
    testpypi_workflow = (root / ".github/workflows/testpypi.yml").read_text(
        encoding="utf-8"
    )

    assert project["project"]["requires-python"] == ">=3.10,<3.15"
    assert 'python-version: ["3.10", "3.11", "3.12", "3.13", "3.14"]' in test_workflow
    assert "scripts/artifact_version.py dist/*" in testpypi_workflow
    assert "steps.package.outputs.version" in testpypi_workflow
    assert "sparsetune==0.1.0" not in testpypi_workflow

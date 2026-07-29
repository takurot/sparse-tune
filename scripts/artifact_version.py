"""Read and cross-check the version embedded in built distribution artifacts."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
import tarfile
import zipfile


def _metadata_version(payload: bytes, source: Path) -> str:
    metadata = BytesParser(policy=default).parsebytes(payload)
    if metadata.get("Name", "").lower().replace("-", "_") != "sparsetune":
        raise ValueError(f"{source}: artifact package name is not sparsetune")
    version = metadata.get("Version")
    if not version or "\n" in version or "\r" in version:
        raise ValueError(f"{source}: artifact version is missing or invalid")
    return version


def _wheel_version(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(names) != 1:
            raise ValueError(f"{path}: expected exactly one wheel METADATA file")
        return _metadata_version(archive.read(names[0]), path)


def _sdist_version(path: Path) -> str:
    with tarfile.open(path, "r:*") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.isfile()
            and member.name.endswith("/PKG-INFO")
            and member.name.count("/") == 1
        ]
        if len(members) != 1:
            raise ValueError(f"{path}: expected exactly one sdist PKG-INFO file")
        extracted = archive.extractfile(members[0])
        if extracted is None:
            raise ValueError(f"{path}: could not read sdist PKG-INFO")
        return _metadata_version(extracted.read(), path)


def artifact_version(paths: list[Path]) -> str:
    """Return the common embedded version for one wheel and one sdist."""

    if not paths:
        raise ValueError("no distribution artifacts were provided")
    versions = {
        _wheel_version(path) if path.suffix == ".whl" else _sdist_version(path)
        for path in paths
    }
    if len(versions) != 1:
        raise ValueError("artifact versions do not match")
    return versions.pop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args()
    print(artifact_version(args.artifacts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

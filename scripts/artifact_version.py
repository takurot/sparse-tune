"""Read and cross-check the version embedded in built distribution artifacts."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
from email.policy import default
from pathlib import Path, PurePosixPath
import re
import tarfile
import zipfile

_SAFE_VERSION_PATTERN = re.compile(
    r"^([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*((a|b|rc|c|alpha|beta|pre|preview)[0-9]*)?(\.?(post|rev|r)[0-9]*)?(\.?dev[0-9]*)?(\+[a-z0-9]+(\.[a-z0-9]+)*)?$",
    re.IGNORECASE,
)


def _metadata_version(payload: bytes, source: Path) -> str:
    for line in payload.splitlines(keepends=True):
        if line.startswith((b"Version:", b"version:")):
            if line.endswith(b"\r\n"):
                content = line[:-2]
            elif line.endswith(b"\n"):
                content = line[:-1]
            else:
                raise ValueError(f"{source}: artifact version is missing or invalid")
            _, _, header_val = content.partition(b":")
            if any(b < 32 or b == 127 for b in header_val):
                raise ValueError(f"{source}: artifact version is missing or invalid")
            break

    try:
        metadata = BytesParser(policy=default).parsebytes(payload)
    except Exception as error:
        raise ValueError(f"{source}: artifact metadata is malformed") from error

    if metadata.get("Name", "").lower().replace("-", "_") != "sparsetune":
        raise ValueError(f"{source}: artifact package name is not sparsetune")

    version = metadata.get("Version")
    if (
        not version
        or not isinstance(version, str)
        or any(ord(char) < 32 or ord(char) == 127 for char in version)
        or not _SAFE_VERSION_PATTERN.match(version)
    ):
        raise ValueError(f"{source}: artifact version is missing or invalid")

    return version


def _wheel_version(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            candidates = []
            for name in archive.namelist():
                normalized = PurePosixPath(name)
                if (
                    not normalized.is_absolute()
                    and len(normalized.parts) == 2
                    and normalized.parts[0] != ".."
                    and normalized.parts[0].endswith(".dist-info")
                    and normalized.parts[1] == "METADATA"
                ):
                    candidates.append(name)
            if len(candidates) != 1:
                raise ValueError(f"{path}: expected exactly one wheel METADATA file")
            payload = archive.read(candidates[0])
    except ValueError:
        raise
    except (
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
        OSError,
        EOFError,
        KeyError,
    ) as error:
        raise ValueError(f"{path}: corrupt or unreadable wheel archive") from error
    return _metadata_version(payload, path)


def _sdist_version(path: Path) -> str:
    try:
        with tarfile.open(path, "r:*") as archive:
            candidates = []
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                normalized = PurePosixPath(member.name)
                if (
                    not normalized.is_absolute()
                    and len(normalized.parts) == 2
                    and normalized.parts[0] != ".."
                    and normalized.parts[1] == "PKG-INFO"
                ):
                    candidates.append(member)
            if len(candidates) != 1:
                raise ValueError(f"{path}: expected exactly one sdist PKG-INFO file")
            extracted = archive.extractfile(candidates[0])
            if extracted is None:
                raise ValueError(f"{path}: could not read sdist PKG-INFO")
            payload = extracted.read()
    except ValueError:
        raise
    except (tarfile.TarError, OSError, EOFError) as error:
        raise ValueError(f"{path}: corrupt or unreadable sdist archive") from error
    return _metadata_version(payload, path)


def artifact_version(paths: list[Path]) -> str:
    """Return the common embedded version for one wheel and one sdist."""

    wheels = [path for path in paths if path.suffix == ".whl"]
    sdists = [path for path in paths if path.name.endswith(".tar.gz")]
    if len(paths) != 2 or len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("expected exactly one wheel and one sdist")
    versions = {_wheel_version(wheels[0]), _sdist_version(sdists[0])}
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

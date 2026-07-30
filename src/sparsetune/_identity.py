"""Portable runtime identity helpers."""

from __future__ import annotations

from pathlib import Path
import platform
import subprocess

import numpy as np


def _sysctl(name: str) -> str | None:
    try:
        completed = subprocess.run(
            ["sysctl", "-n", name],
            capture_output=True,
            check=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _linux_cpuinfo() -> str | None:
    try:
        return Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _cpu_model() -> str | None:
    if platform.system() == "Darwin":
        model = _sysctl("machdep.cpu.brand_string")
        if model:
            return model
    cpuinfo = _linux_cpuinfo()
    if cpuinfo is not None:
        for line in cpuinfo.splitlines():
            if line.lower().startswith(("model name", "hardware")):
                value = line.partition(":")[2].strip()
                if value:
                    return value
    model = platform.processor().strip()
    if model:
        return model
    return None


def _physical_core_count() -> int | None:
    if platform.system() == "Darwin":
        value = _sysctl("hw.physicalcpu")
        return int(value) if value and value.isdigit() else None
    cpuinfo = _linux_cpuinfo()
    if cpuinfo is None:
        return None
    packages: set[tuple[str, str]] = set()
    physical_id: str | None = None
    core_id: str | None = None
    for line in (*cpuinfo.splitlines(), ""):
        if not line.strip():
            if physical_id is not None and core_id is not None:
                packages.add((physical_id, core_id))
            physical_id = None
            core_id = None
        elif line.startswith("physical id"):
            physical_id = line.partition(":")[2].strip()
        elif line.startswith("core id"):
            core_id = line.partition(":")[2].strip()
    return len(packages) or None


def _blas_implementation() -> str | None:
    config = getattr(np.__config__, "CONFIG", None)
    if isinstance(config, dict):
        dependencies = config.get("Build Dependencies")
        if isinstance(dependencies, dict):
            blas = dependencies.get("blas")
            if isinstance(blas, dict) and blas.get("found", True):
                name = blas.get("name")
                version = blas.get("version")
                if isinstance(name, str) and name:
                    if isinstance(version, str) and version not in {"", "unknown"}:
                        return f"{name} {version}"
                    return name
    get_info = getattr(np.__config__, "get_info", None)
    if callable(get_info):
        try:
            info = get_info("blas_opt_info")
        except Exception:
            return None
        if isinstance(info, dict):
            libraries = info.get("libraries")
            if isinstance(libraries, list) and libraries:
                return ",".join(str(library) for library in libraries)
    return None


def cpu_identity() -> dict[str, str | int | None]:
    """Return stable CPU fields, using null when a field is unavailable."""

    return {
        "cpu_model": _cpu_model(),
        "cpu_cores_physical": _physical_core_count(),
        "blas_implementation": _blas_implementation(),
    }

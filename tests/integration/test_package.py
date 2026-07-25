from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_import_is_quiet_and_does_not_import_cupy() -> None:
    project_root = Path(__file__).parents[2]
    environment = os.environ.copy()
    python_path = str(project_root / "src")
    if environment.get("PYTHONPATH"):
        python_path = os.pathsep.join((python_path, environment["PYTHONPATH"]))
    environment["PYTHONPATH"] = python_path

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import sparsetune; "
                "assert not any(name == 'cupy' or name.startswith('cupy.') "
                "for name in sys.modules); "
                "print(sparsetune.__version__)"
            ),
        ],
        capture_output=True,
        check=True,
        env=environment,
        text=True,
    )

    assert completed.stdout == "0.1.0\n"
    assert completed.stderr == ""

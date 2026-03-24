from __future__ import annotations

import importlib.metadata as metadata
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample" / "xmas-cowboy.jpeg"


def test_console_scripts_include_primary_and_compat_aliases() -> None:
    entry_points = {entry_point.name: entry_point.value for entry_point in metadata.entry_points(group="console_scripts")}
    assert entry_points["stampify"] == "ink_print.cli:main"
    assert entry_points["ink-stamp"] == "ink_print.cli:main"


def test_module_cli_generates_stamp(tmp_path: Path) -> None:
    output = tmp_path / "cli.stl"
    result = subprocess.run(
        [sys.executable, "-m", "ink_print", str(SAMPLE), "-o", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert output.exists()
    assert "Wrote" in result.stdout


def test_cli_reports_validation_errors_cleanly() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ink_print", str(SAMPLE), "--size", "2", "--border", "2"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert result.stderr.startswith("error: size must be larger than twice border.")
    assert "Traceback" not in result.stderr

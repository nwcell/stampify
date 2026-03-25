from __future__ import annotations

import importlib.metadata as metadata
import subprocess
import sys
from pathlib import Path

import trimesh


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample" / "xmas-cowboy.jpeg"


def test_package_import_is_lazy() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import sys, ink_print; print('ink_print.core' in sys.modules)"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_console_scripts_include_stampify() -> None:
    entry_points = {entry_point.name: entry_point.value for entry_point in metadata.entry_points(group="console_scripts")}
    assert entry_points["stampify"] == "ink_print.cli:main"


def test_module_cli_generates_stamp(tmp_path: Path) -> None:
    output = tmp_path / "cli.stl"
    result = subprocess.run(
        [sys.executable, "-m", "ink_print", str(SAMPLE), "-o", str(output), "--width", "90", "--height", "70"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert output.exists()
    assert "Wrote" in result.stdout


def test_module_cli_uses_legacy_size_fallback(tmp_path: Path) -> None:
    output = tmp_path / "cli-size.stl"
    result = subprocess.run(
        [sys.executable, "-m", "ink_print", str(SAMPLE), "-o", str(output), "--size", "90"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert output.exists()

    mesh = trimesh.load_mesh(output)
    assert round(float(max(mesh.extents)), 1) == 90.0


def test_cli_reports_validation_errors_cleanly() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ink_print", str(SAMPLE), "--width", "2", "--height", "2", "--border", "2"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert result.stderr.startswith("error: max dimension must be larger than twice border.")
    assert "Traceback" not in result.stderr

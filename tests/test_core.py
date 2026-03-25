from __future__ import annotations

from pathlib import Path

from ink_print import StampOptions, build_stamp_mesh, default_output_path, load_mask, write_stamp
from ink_print.core import validate_size


SAMPLE = Path(__file__).resolve().parents[1] / "sample" / "xmas-cowboy.jpeg"


def test_default_output_path_uses_input_stem() -> None:
    assert default_output_path(SAMPLE).name == "xmas-cowboy-stamp.stl"


def test_build_stamp_mesh_vector_is_watertight() -> None:
    mesh = build_stamp_mesh(SAMPLE, StampOptions())
    assert mesh.is_watertight
    assert round(float(max(mesh.extents)), 1) == 80.0


def test_build_stamp_mesh_respects_rectangular_dimensions() -> None:
    mesh = build_stamp_mesh(SAMPLE, StampOptions(width=90.0, height=70.0))
    assert mesh.is_watertight
    assert round(float(mesh.extents[0]), 1) == 90.0
    assert round(float(mesh.extents[1]), 1) == 70.0


def test_validate_size_uses_the_larger_inner_dimension() -> None:
    assert validate_size(StampOptions(width=90.0, height=70.0, border=2.0)) == 86.0


def test_write_stamp_writes_expected_file(tmp_path: Path) -> None:
    output_path, mesh = write_stamp(SAMPLE, tmp_path / "stamp.stl", StampOptions(mode="voxel", resolution=300))
    assert output_path.exists()
    assert mesh.is_watertight


def test_load_mask_raises_for_empty_artwork() -> None:
    try:
        load_mask(SAMPLE, StampOptions(threshold=0))
    except ValueError as exc:
        assert "No stamp pixels found" in str(exc)
    else:
        raise AssertionError("Expected load_mask to raise for an empty thresholded image")

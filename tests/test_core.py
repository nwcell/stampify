from __future__ import annotations

from pathlib import Path

from ink_print import StampOptions, build_stamp_mesh, default_output_path, load_mask, write_stamp
from ink_print.core import _rasterize_svg, validate_size


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


def test_build_stamp_mesh_uses_legacy_size_fallback() -> None:
    mesh = build_stamp_mesh(SAMPLE, StampOptions(size=90.0))
    assert mesh.is_watertight
    assert round(float(max(mesh.extents)), 1) == 90.0


def test_validate_size_uses_the_larger_inner_dimension() -> None:
    assert validate_size(StampOptions(width=90.0, height=70.0, border=2.0)) == 86.0


def test_validate_size_uses_legacy_size_fallback() -> None:
    assert validate_size(StampOptions(size=90.0, border=2.0)) == 86.0


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


def test_rasterize_svg_preserves_path_holes(tmp_path: Path) -> None:
    svg_path = tmp_path / "hole.svg"
    svg_path.write_text(
        """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
          <path fill="black" fill-rule="evenodd" d="M 10 10 H 90 V 90 H 10 Z M 30 30 H 70 V 70 H 30 Z" />
        </svg>
        """.strip(),
        encoding="utf-8",
    )

    image = _rasterize_svg(svg_path)
    assert image.getpixel((20, 20)) == 0
    assert image.getpixel((50, 50)) == 255


def test_rasterize_svg_samples_arc_segments(tmp_path: Path) -> None:
    svg_path = tmp_path / "arc.svg"
    svg_path.write_text(
        """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
          <path fill="black" d="M 50 10 A 40 40 0 0 1 90 50 A 40 40 0 0 1 50 90 A 40 40 0 0 1 10 50 A 40 40 0 0 1 50 10" />
        </svg>
        """.strip(),
        encoding="utf-8",
    )

    image = _rasterize_svg(svg_path)
    assert image.getpixel((80, 30)) == 0
    assert image.getpixel((95, 50)) == 255


def test_rasterize_svg_handles_rotated_ellipses(tmp_path: Path) -> None:
    svg_path = tmp_path / "ellipse.svg"
    svg_path.write_text(
        """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
          <ellipse cx="50" cy="50" rx="35" ry="10" transform="rotate(45 50 50)" fill="black" />
        </svg>
        """.strip(),
        encoding="utf-8",
    )

    image = _rasterize_svg(svg_path)
    assert image.getpixel((70, 70)) == 0
    assert image.getpixel((70, 50)) == 255

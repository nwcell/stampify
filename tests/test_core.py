from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

from ink_print import (
    StampOptions,
    build_stamp_mesh,
    build_stamp_svg,
    default_output_path,
    load_mask,
    write_stamp,
)
import ink_print.core as core
from ink_print.core import (
    _geometry_vertex_count,
    _matrix_scale,
    _parse_svg_path,
    _rasterize_svg,
    _sample_svg_ellipse,
    build_stamp_mesh_from_geometry,
    prepare_stamp_geometry,
    resolve_artwork,
    trace_geometry,
    validate_size,
)


SAMPLE = Path(__file__).resolve().parents[1] / "sample" / "xmas-cowboy.jpeg"


def test_default_output_path_uses_input_stem() -> None:
    assert default_output_path(SAMPLE).name == "xmas-cowboy-stamp.stl"


def test_build_stamp_mesh_vector_is_watertight() -> None:
    mesh = build_stamp_mesh(SAMPLE, StampOptions())
    assert mesh.is_watertight
    assert round(float(max(mesh.extents)), 1) == 80.0


def test_build_stamp_mesh_respects_max_dimension() -> None:
    mesh = build_stamp_mesh(SAMPLE, StampOptions(width=90.0, height=70.0))
    assert mesh.is_watertight
    extents = sorted(float(value) for value in mesh.extents)
    assert round(extents[-1], 1) == 90.0
    assert extents[0] < 70.0


def test_build_stamp_mesh_uses_single_max_dimension_when_one_is_blank() -> None:
    mesh = build_stamp_mesh(SAMPLE, StampOptions(width=90.0))
    assert mesh.is_watertight
    assert round(float(max(mesh.extents)), 1) == 90.0


def test_build_stamp_mesh_uses_legacy_size_fallback() -> None:
    mesh = build_stamp_mesh(SAMPLE, StampOptions(size=90.0))
    assert mesh.is_watertight
    assert round(float(max(mesh.extents)), 1) == 90.0


def test_build_stamp_svg_returns_browser_ready_markup() -> None:
    svg = build_stamp_svg(SAMPLE, StampOptions())

    assert svg.startswith("<svg")
    assert "shape-rendering=\"geometricPrecision\"" in svg


def test_validate_size_uses_the_larger_inner_dimension() -> None:
    assert validate_size(StampOptions(width=90.0, height=70.0, border=2.0)) == 86.0


def test_validate_size_uses_single_max_dimension_when_one_is_blank() -> None:
    assert validate_size(StampOptions(width=90.0, border=2.0)) == 86.0


def test_validate_size_uses_legacy_size_fallback() -> None:
    assert validate_size(StampOptions(size=90.0, border=2.0)) == 86.0


def test_write_stamp_writes_expected_file(tmp_path: Path) -> None:
    output_path, mesh = write_stamp(SAMPLE, tmp_path / "stamp.stl", StampOptions(resolution=300))
    assert output_path.exists()
    assert mesh.is_watertight


def test_load_mask_auto_tunes_raster_threshold_and_resolution(tmp_path: Path) -> None:
    image = Image.new("L", (12, 12), 245)
    draw = ImageDraw.Draw(image)
    draw.rectangle((3, 3, 8, 8), fill=60)
    image_path = tmp_path / "auto-raster.png"
    image.save(image_path)

    mask = load_mask(image_path, StampOptions())

    assert mask.shape[0] > 100
    assert mask.shape[1] > 100
    assert mask[mask.shape[0] // 2, mask.shape[1] // 2]
    assert not mask[0, 0]


def test_resolve_artwork_auto_smooths_raster_jpeg() -> None:
    resolved = resolve_artwork(SAMPLE, StampOptions())

    assert resolved.source_kind == "raster"
    assert resolved.profile.simplify >= 0.03
    assert _geometry_vertex_count(resolved.geometry) < 16000
    assert resolved.preview_svg.startswith("<svg")


def test_resolve_artwork_uses_vtracer_for_raster_input(tmp_path: Path, monkeypatch) -> None:
    raster_path = tmp_path / "raster.png"
    image = Image.new("L", (24, 24), 255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((6, 6, 18, 18), fill=0)
    image.save(raster_path)

    calls: list[tuple[str | None, dict[str, object]]] = []

    class FakeVTracer:
        def convert_raw_image_to_svg(self, img_bytes, img_format=None, **kwargs):
            calls.append((img_format, kwargs))
            return (
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
                '<path fill="black" d="M 1 1 H 3 V 4 H 2 V 3 H 1 Z" />'
                "</svg>"
            )

    def fail_trace_geometry(mask) -> object:
        raise AssertionError("contour tracing should not run when vtracer is available")

    monkeypatch.setattr(core, "vtracer", FakeVTracer())
    monkeypatch.setattr(core, "trace_geometry", fail_trace_geometry)

    resolved = resolve_artwork(raster_path, StampOptions())
    geometry_mirror, _ = core._vectorize_raster_artwork(raster_path, StampOptions())
    geometry_no_mirror, _ = core._vectorize_raster_artwork(raster_path, StampOptions(mirror=False))

    assert resolved.source_kind == "raster"
    assert calls and calls[0][0] == "png"
    assert resolved.geometry.area > 0
    assert geometry_mirror.centroid.x > geometry_no_mirror.centroid.x
    assert resolved.preview_svg.startswith("<svg")


def test_prepare_stamp_geometry_auto_simplifies_complex_geometry() -> None:
    coords: list[tuple[float, float]] = []
    coords.extend((float(x), 0.0) for x in np.linspace(0.0, 100.0, 80, endpoint=False))
    coords.extend((100.0, float(y)) for y in np.linspace(0.0, 100.0, 80, endpoint=False))
    coords.extend((float(x), 100.0) for x in np.linspace(100.0, 0.0, 80, endpoint=False))
    coords.extend((0.0, float(y)) for y in np.linspace(100.0, 0.0, 80, endpoint=False))
    geometry = Polygon(coords)

    prepared = prepare_stamp_geometry(geometry, StampOptions(width=90.0, height=70.0))

    assert _geometry_vertex_count(prepared) < _geometry_vertex_count(geometry)


def test_resolve_artwork_uses_direct_svg_geometry(tmp_path: Path, monkeypatch) -> None:
    svg_path = tmp_path / "direct-geometry.svg"
    svg_path.write_text(
        """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
          <g transform="translate(4 6) scale(1.1)">
            <path fill="black" fill-rule="evenodd" d="M 20 20 C 35 5 65 5 80 20 A 30 30 0 0 1 80 80 C 65 95 35 95 20 80 A 30 30 0 0 1 20 20 M 40 40 H 60 V 60 H 40 Z" />
          </g>
        </svg>
        """.strip(),
        encoding="utf-8",
    )

    def fail_rasterize(*args, **kwargs) -> Image.Image:
        raise AssertionError("direct SVG geometry should not rasterize supported artwork")

    monkeypatch.setattr(core, "_rasterize_svg", fail_rasterize)

    resolved = resolve_artwork(svg_path, StampOptions(width=90.0, height=70.0))

    assert resolved.source_kind == "svg"
    assert resolved.profile.simplify == 0.0
    assert resolved.geometry.area > 0
    assert resolved.preview_svg.startswith("<svg")

    mesh = build_stamp_mesh_from_geometry(resolved.geometry, StampOptions(width=90.0, height=70.0), prepared=True)
    assert mesh.is_watertight


def test_build_stamp_mesh_falls_back_for_unsupported_svg(tmp_path: Path, monkeypatch) -> None:
    svg_path = tmp_path / "fallback.svg"
    svg_path.write_text(
        """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
          <text x="10" y="30">fallback</text>
        </svg>
        """.strip(),
        encoding="utf-8",
    )

    calls: list[Path] = []
    vector_calls: list[tuple[str | None, dict[str, object]]] = []

    def fake_rasterize(image_path: Path, minimum_side: int = 1) -> Image.Image:
        calls.append(image_path)
        image = Image.new("L", (48, 48), 255)
        draw = ImageDraw.Draw(image)
        draw.rectangle((12, 12, 36, 36), fill=0)
        return image

    class FakeVTracer:
        def convert_raw_image_to_svg(self, img_bytes, img_format=None, **kwargs):
            vector_calls.append((img_format, kwargs))
            return (
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
                '<path fill="black" d="M 1 1 H 9 V 9 H 1 Z" />'
                "</svg>"
            )

    def fail_trace_geometry(mask) -> object:
        raise AssertionError("fallback contour tracing should not run when vtracer is available")

    monkeypatch.setattr(core, "_rasterize_svg", fake_rasterize)
    monkeypatch.setattr(core, "vtracer", FakeVTracer())
    monkeypatch.setattr(core, "trace_geometry", fail_trace_geometry)

    resolved = resolve_artwork(svg_path, StampOptions(width=90.0, height=70.0))
    assert resolved.source_kind == "svg"
    assert resolved.profile.simplify >= 0.03
    assert vector_calls and vector_calls[0][0] == "png"

    calls.clear()
    mesh = build_stamp_mesh(svg_path, StampOptions(width=90.0, height=70.0))

    assert calls == [svg_path]
    assert mesh.is_watertight


def test_load_mask_raises_for_empty_artwork() -> None:
    try:
        load_mask(SAMPLE, StampOptions(threshold=0))
    except ValueError as exc:
        assert "No stamp pixels found" in str(exc)
    else:
        raise AssertionError("Expected load_mask to raise for an empty thresholded image")


def test_trace_geometry_handles_artwork_touching_the_crop_edge() -> None:
    geometry = trace_geometry(np.array([[True, True], [True, True]]))
    assert geometry.geom_type == "Polygon"
    assert round(float(geometry.area), 1) == 3.5
    assert tuple(round(value, 1) for value in geometry.bounds) == (-0.5, 0.5, 1.5, 2.5)


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


def test_load_mask_upsamples_small_svg_viewbox(tmp_path: Path) -> None:
    svg_path = tmp_path / "upsampled.svg"
    svg_path.write_text(
        """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
          <circle cx="50" cy="50" r="40" fill="black" />
        </svg>
        """.strip(),
        encoding="utf-8",
    )

    mask = load_mask(svg_path, StampOptions())

    assert mask.shape[0] > 300
    assert mask.shape[1] > 300


def test_parse_svg_path_increases_arc_sampling_when_scaled() -> None:
    path_data = "M 50 10 A 40 40 0 0 1 90 50"

    normal = _parse_svg_path(path_data)
    scaled = _parse_svg_path(path_data, [_matrix_scale(8)])

    assert len(scaled[0][0]) >= len(normal[0][0]) * 4


def test_sample_svg_ellipse_increases_sampling_when_scaled() -> None:
    points = _sample_svg_ellipse(50, 50, 35, 10, [_matrix_scale(8)])

    assert len(points) > 200


def test_rasterize_svg_applies_nested_transforms_in_svg_order(tmp_path: Path) -> None:
    svg_path = tmp_path / "nested-transform.svg"
    svg_path.write_text(
        """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
          <g transform="scale(2)">
            <rect x="0" y="0" width="10" height="10" transform="translate(20 0)" fill="black" />
          </g>
        </svg>
        """.strip(),
        encoding="utf-8",
    )

    image = _rasterize_svg(svg_path)
    assert image.getpixel((50, 10)) == 0
    assert image.getpixel((30, 10)) == 255


def test_rasterize_svg_applies_viewbox_offset_before_scaling(tmp_path: Path) -> None:
    svg_path = tmp_path / "viewbox-offset.svg"
    svg_path.write_text(
        """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="10 10 20 20" width="40" height="40">
          <rect x="10" y="10" width="10" height="10" fill="black" />
        </svg>
        """.strip(),
        encoding="utf-8",
    )

    image = _rasterize_svg(svg_path)
    assert image.getpixel((5, 5)) == 0
    assert image.getpixel((25, 5)) == 255

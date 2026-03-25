from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageOps
from shapely import unary_union
from shapely.affinity import scale as scale_geometry, translate
from shapely.geometry import GeometryCollection, Polygon, box
from shapely.validation import make_valid
from skimage import measure


@dataclass(slots=True)
class StampOptions:
    size: float = 80.0
    width: float | None = None
    height: float | None = None
    border: float = 0.5
    base: float = 4.0
    relief: float = 1.0
    simplify: float = 0.08
    min_area: float = 0.02
    threshold: int = 190
    resolution: int = 0
    raised_border: bool = True
    invert: bool = False
    mirror: bool = True


def default_output_path(image: str | Path) -> Path:
    image_path = Path(image)
    return image_path.with_name(f"{image_path.stem}-stamp.stl")


_SVG_COMMAND_RE = re.compile(r"([MmZzLlHhVvCcSsQqTtAa])|([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")
_SVG_TRANSFORM_RE = re.compile(r"([a-zA-Z]+)\(([^)]*)\)")


def _parse_svg_number(value: str | None, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    match = re.match(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value.strip())
    return float(match.group(0)) if match else default


def _svg_tag_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _parse_svg_style(element: ET.Element) -> dict[str, str]:
    style: dict[str, str] = {}
    style_text = element.attrib.get("style", "")
    for part in style_text.split(";"):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        style[key.strip()] = value.strip()
    for key in ("fill", "stroke", "stroke-width", "fill-rule", "opacity", "fill-opacity", "stroke-opacity"):
        if key in element.attrib:
            style[key] = element.attrib[key]
    return style


def _svg_paint_enabled(value: str | None) -> bool:
    if not value:
        return False
    normalized = value.strip().lower()
    return normalized not in {"none", "transparent"}


def _parse_svg_points(text: str) -> list[tuple[float, float]]:
    values = [float(part) for part in re.split(r"[\s,]+", text.strip()) if part]
    return list(zip(values[::2], values[1::2]))


def _matrix_translate(tx: float, ty: float) -> tuple[float, float, float, float, float, float]:
    return (1.0, 0.0, 0.0, 1.0, tx, ty)


def _matrix_scale(sx: float, sy: float | None = None) -> tuple[float, float, float, float, float, float]:
    sy = sx if sy is None else sy
    return (sx, 0.0, 0.0, sy, 0.0, 0.0)


def _matrix_rotate(angle_degrees: float, cx: float = 0.0, cy: float = 0.0) -> tuple[float, float, float, float, float, float]:
    radians = math.radians(angle_degrees)
    cos_theta = math.cos(radians)
    sin_theta = math.sin(radians)
    return (
        cos_theta,
        sin_theta,
        -sin_theta,
        cos_theta,
        cx - cos_theta * cx + sin_theta * cy,
        cy - sin_theta * cx - cos_theta * cy,
    )


def _parse_svg_transforms(transform: str | None) -> list[tuple[float, float, float, float, float, float]]:
    if not transform:
        return []

    matrices: list[tuple[float, float, float, float, float, float]] = []
    for name, args in _SVG_TRANSFORM_RE.findall(transform):
        values = [float(part) for part in re.split(r"[\s,]+", args.strip()) if part]
        name = name.lower()
        if name == "translate":
            matrices.append(_matrix_translate(values[0] if values else 0.0, values[1] if len(values) > 1 else 0.0))
        elif name == "scale":
            matrices.append(_matrix_scale(values[0] if values else 1.0, values[1] if len(values) > 1 else None))
        elif name == "rotate":
            if len(values) >= 3:
                matrices.append(_matrix_rotate(values[0], values[1], values[2]))
            elif values:
                matrices.append(_matrix_rotate(values[0]))
        elif name == "matrix" and len(values) == 6:
            matrices.append(tuple(values))
    return matrices


def _apply_svg_transform(point: tuple[float, float], transforms: list[tuple[float, float, float, float, float, float]]) -> tuple[float, float]:
    x, y = point
    for a, b, c, d, e, f in reversed(transforms):
        x, y = a * x + c * y + e, b * x + d * y + f
    return x, y


def _apply_svg_transforms(points: list[tuple[float, float]], transforms: list[tuple[float, float, float, float, float, float]]) -> list[tuple[float, float]]:
    return [_apply_svg_transform(point, transforms) for point in points]


def _sample_cubic(
    start: tuple[float, float],
    control1: tuple[float, float],
    control2: tuple[float, float],
    end: tuple[float, float],
    segments: int = 16,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for index in range(1, segments + 1):
        t = index / segments
        inv = 1.0 - t
        x = (inv**3) * start[0] + 3 * (inv**2) * t * control1[0] + 3 * inv * (t**2) * control2[0] + (t**3) * end[0]
        y = (inv**3) * start[1] + 3 * (inv**2) * t * control1[1] + 3 * inv * (t**2) * control2[1] + (t**3) * end[1]
        points.append((x, y))
    return points


def _sample_quadratic(
    start: tuple[float, float],
    control: tuple[float, float],
    end: tuple[float, float],
    segments: int = 12,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for index in range(1, segments + 1):
        t = index / segments
        inv = 1.0 - t
        x = (inv**2) * start[0] + 2 * inv * t * control[0] + (t**2) * end[0]
        y = (inv**2) * start[1] + 2 * inv * t * control[1] + (t**2) * end[1]
        points.append((x, y))
    return points


def _close_svg_ring(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    ring = list(points)
    if len(ring) >= 2 and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def _svg_ring_area(points: list[tuple[float, float]]) -> float:
    ring = _close_svg_ring(points)
    if len(ring) < 4:
        return 0.0

    area = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        area += x1 * y2 - x2 * y1
    return area / 2.0


def _sample_svg_arc(
    start: tuple[float, float],
    rx: float,
    ry: float,
    rotation_degrees: float,
    large_arc: bool,
    sweep: bool,
    end: tuple[float, float],
) -> list[tuple[float, float]]:
    if start == end:
        return []

    rx = abs(rx)
    ry = abs(ry)
    if rx == 0 or ry == 0:
        return [end]

    phi = math.radians(rotation_degrees % 360.0)
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)

    dx2 = (start[0] - end[0]) / 2.0
    dy2 = (start[1] - end[1]) / 2.0
    x1p = cos_phi * dx2 + sin_phi * dy2
    y1p = -sin_phi * dx2 + cos_phi * dy2

    rx_sq = rx * rx
    ry_sq = ry * ry
    x1p_sq = x1p * x1p
    y1p_sq = y1p * y1p

    radii_check = x1p_sq / rx_sq + y1p_sq / ry_sq
    if radii_check > 1.0:
        scale = math.sqrt(radii_check)
        rx *= scale
        ry *= scale
        rx_sq = rx * rx
        ry_sq = ry * ry

    numerator = max(0.0, rx_sq * ry_sq - rx_sq * y1p_sq - ry_sq * x1p_sq)
    denominator = rx_sq * y1p_sq + ry_sq * x1p_sq
    if denominator == 0:
        return [end]

    sign = -1.0 if large_arc == sweep else 1.0
    coef = sign * math.sqrt(numerator / denominator)
    cxp = coef * (rx * y1p) / ry
    cyp = coef * (-ry * x1p) / rx

    cx = cos_phi * cxp - sin_phi * cyp + (start[0] + end[0]) / 2.0
    cy = sin_phi * cxp + cos_phi * cyp + (start[1] + end[1]) / 2.0

    def _vector_angle(ux: float, uy: float, vx: float, vy: float) -> float:
        return math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)

    ux = (x1p - cxp) / rx
    uy = (y1p - cyp) / ry
    vx = (-x1p - cxp) / rx
    vy = (-y1p - cyp) / ry
    theta1 = math.atan2(uy, ux)
    delta_theta = _vector_angle(ux, uy, vx, vy)
    if not sweep and delta_theta > 0:
        delta_theta -= 2.0 * math.pi
    elif sweep and delta_theta < 0:
        delta_theta += 2.0 * math.pi

    segments = max(8, math.ceil(abs(delta_theta) / (math.pi / 12.0)))
    points: list[tuple[float, float]] = []
    for index in range(1, segments + 1):
        theta = theta1 + delta_theta * (index / segments)
        x = cos_phi * rx * math.cos(theta) - sin_phi * ry * math.sin(theta) + cx
        y = sin_phi * rx * math.cos(theta) + cos_phi * ry * math.sin(theta) + cy
        points.append((x, y))
    return points


def _sample_svg_ellipse(
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    transforms: list[tuple[float, float, float, float, float, float]],
    segments: int = 96,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for index in range(segments):
        theta = (2.0 * math.pi * index) / segments
        point = (cx + rx * math.cos(theta), cy + ry * math.sin(theta))
        points.append(_apply_svg_transform(point, transforms))
    return points


def _svg_subpaths_to_geometry(
    subpaths: list[tuple[list[tuple[float, float]], bool]],
    fill_rule: str | None,
):
    polygons: list[tuple[Polygon, float]] = []
    for points, _closed in subpaths:
        ring = _close_svg_ring(points)
        if len(ring) < 4:
            continue
        ring_area = _svg_ring_area(ring)
        geometry = make_valid(Polygon(ring))
        polygons.extend((poly, ring_area) for poly in iter_polygons(geometry))

    if not polygons:
        return GeometryCollection()

    rule = (fill_rule or "nonzero").strip().lower()
    if rule == "evenodd":
        ordered = sorted((poly for poly, _ring_area in polygons), key=lambda poly: poly.area, reverse=True)
        depths: list[int] = []
        for index, poly in enumerate(ordered):
            point = poly.representative_point()
            parent = next((candidate for candidate in range(index) if ordered[candidate].contains(point)), None)
            depths.append(0 if parent is None else depths[parent] + 1)

        geometry = GeometryCollection()
        for depth, poly in sorted(zip(depths, ordered), key=lambda item: (item[0], -item[1].area)):
            geometry = geometry.union(poly) if depth % 2 == 0 else geometry.difference(poly)
        return make_valid(geometry)

    root_sign = 1.0
    for _poly, ring_area in sorted(polygons, key=lambda item: item[0].area, reverse=True):
        if ring_area != 0:
            root_sign = 1.0 if ring_area > 0 else -1.0
            break

    geometry = GeometryCollection()
    for poly, ring_area in sorted(polygons, key=lambda item: item[0].area, reverse=True):
        geometry = geometry.union(poly) if ring_area * root_sign >= 0 else geometry.difference(poly)
    return make_valid(geometry)


def _draw_svg_geometry(mask: Image.Image, geometry) -> None:
    drawer = ImageDraw.Draw(mask)
    for poly in iter_polygons(make_valid(geometry)):
        exterior = list(poly.exterior.coords)
        if len(exterior) >= 3:
            drawer.polygon(exterior, fill=1)
        for interior in poly.interiors:
            hole = list(interior.coords)
            if len(hole) >= 3:
                drawer.polygon(hole, fill=0)


def _parse_svg_path(path_data: str) -> list[tuple[list[tuple[float, float]], bool]]:
    tokens: list[str | float] = []
    for command, number in _SVG_COMMAND_RE.findall(path_data):
        if command:
            tokens.append(command)
        elif number:
            tokens.append(float(number))

    subpaths: list[tuple[list[tuple[float, float]], bool]] = []
    points: list[tuple[float, float]] = []
    current = (0.0, 0.0)
    start_point = (0.0, 0.0)
    command: str | None = None
    index = 0
    closed = False
    last_cubic_control: tuple[float, float] | None = None
    last_quadratic_control: tuple[float, float] | None = None

    def flush_subpath() -> None:
        nonlocal points, closed
        if points:
            subpaths.append((points, closed))
        points = []
        closed = False

    while index < len(tokens):
        token = tokens[index]
        if isinstance(token, str):
            command = token
            index += 1
            if command in {"Z", "z"}:
                if points and points[0] != points[-1]:
                    points.append(points[0])
                closed = True
                flush_subpath()
                current = start_point
                last_cubic_control = None
                last_quadratic_control = None
                command = None
            continue

        if command is None:
            index += 1
            continue

        absolute = command.isupper()
        opcode = command.upper()

        if opcode == "M":
            if index + 1 >= len(tokens) or isinstance(tokens[index + 1], str):
                break
            x = float(tokens[index])
            y = float(tokens[index + 1])
            index += 2
            if not absolute:
                x += current[0]
                y += current[1]
            flush_subpath()
            current = start_point = (x, y)
            points = [current]
            last_cubic_control = None
            last_quadratic_control = None
            command = "L" if absolute else "l"
            continue

        if opcode == "L":
            if index + 1 >= len(tokens) or isinstance(tokens[index + 1], str):
                break
            x = float(tokens[index])
            y = float(tokens[index + 1])
            index += 2
            if not absolute:
                x += current[0]
                y += current[1]
            current = (x, y)
            points.append(current)
            last_cubic_control = None
            last_quadratic_control = None
            continue

        if opcode == "H":
            x = float(tokens[index])
            index += 1
            if not absolute:
                x += current[0]
            current = (x, current[1])
            points.append(current)
            last_cubic_control = None
            last_quadratic_control = None
            continue

        if opcode == "V":
            y = float(tokens[index])
            index += 1
            if not absolute:
                y += current[1]
            current = (current[0], y)
            points.append(current)
            last_cubic_control = None
            last_quadratic_control = None
            continue

        if opcode == "C":
            if index + 5 >= len(tokens) or isinstance(tokens[index + 5], str):
                break
            c1 = (float(tokens[index]), float(tokens[index + 1]))
            c2 = (float(tokens[index + 2]), float(tokens[index + 3]))
            end = (float(tokens[index + 4]), float(tokens[index + 5]))
            index += 6
            if not absolute:
                c1 = (c1[0] + current[0], c1[1] + current[1])
                c2 = (c2[0] + current[0], c2[1] + current[1])
                end = (end[0] + current[0], end[1] + current[1])
            points.extend(_sample_cubic(current, c1, c2, end))
            current = end
            last_cubic_control = c2
            last_quadratic_control = None
            continue

        if opcode == "S":
            if index + 3 >= len(tokens) or isinstance(tokens[index + 3], str):
                break
            if last_cubic_control is None:
                c1 = current
            else:
                c1 = (2 * current[0] - last_cubic_control[0], 2 * current[1] - last_cubic_control[1])
            c2 = (float(tokens[index]), float(tokens[index + 1]))
            end = (float(tokens[index + 2]), float(tokens[index + 3]))
            index += 4
            if not absolute:
                c2 = (c2[0] + current[0], c2[1] + current[1])
                end = (end[0] + current[0], end[1] + current[1])
            points.extend(_sample_cubic(current, c1, c2, end))
            current = end
            last_cubic_control = c2
            last_quadratic_control = None
            continue

        if opcode == "Q":
            if index + 3 >= len(tokens) or isinstance(tokens[index + 3], str):
                break
            control = (float(tokens[index]), float(tokens[index + 1]))
            end = (float(tokens[index + 2]), float(tokens[index + 3]))
            index += 4
            if not absolute:
                control = (control[0] + current[0], control[1] + current[1])
                end = (end[0] + current[0], end[1] + current[1])
            points.extend(_sample_quadratic(current, control, end))
            current = end
            last_quadratic_control = control
            last_cubic_control = None
            continue

        if opcode == "T":
            if index + 1 >= len(tokens) or isinstance(tokens[index + 1], str):
                break
            if last_quadratic_control is None:
                control = current
            else:
                control = (2 * current[0] - last_quadratic_control[0], 2 * current[1] - last_quadratic_control[1])
            end = (float(tokens[index]), float(tokens[index + 1]))
            index += 2
            if not absolute:
                end = (end[0] + current[0], end[1] + current[1])
            points.extend(_sample_quadratic(current, control, end))
            current = end
            last_quadratic_control = control
            last_cubic_control = None
            continue

        if opcode == "A":
            if index + 6 >= len(tokens) or isinstance(tokens[index + 6], str):
                break
            rx = float(tokens[index])
            ry = float(tokens[index + 1])
            rotation = float(tokens[index + 2])
            large_arc = bool(float(tokens[index + 3]))
            sweep = bool(float(tokens[index + 4]))
            end = (float(tokens[index + 5]), float(tokens[index + 6]))
            index += 7
            if not absolute:
                end = (end[0] + current[0], end[1] + current[1])
            points.extend(_sample_svg_arc(current, rx, ry, rotation, large_arc, sweep, end))
            current = end
            last_cubic_control = None
            last_quadratic_control = None
            continue

        index += 1

    flush_subpath()
    return subpaths


def _svg_canvas_size(root: ET.Element) -> tuple[int, int, list[tuple[float, float, float, float, float, float]]]:
    view_box = root.attrib.get("viewBox")
    if view_box:
        minx, miny, vb_width, vb_height = [float(part) for part in re.split(r"[\s,]+", view_box.strip()) if part]
    else:
        minx = miny = 0.0
        vb_width = _parse_svg_number(root.attrib.get("width"), 1024.0)
        vb_height = _parse_svg_number(root.attrib.get("height"), 1024.0)

    width = max(1, round(_parse_svg_number(root.attrib.get("width"), vb_width)))
    height = max(1, round(_parse_svg_number(root.attrib.get("height"), vb_height)))
    sx = width / vb_width if vb_width else 1.0
    sy = height / vb_height if vb_height else 1.0
    transforms = [_matrix_scale(sx, sy), _matrix_translate(-minx, -miny)]
    return width, height, transforms


def _draw_svg_element(mask: Image.Image, element: ET.Element, inherited: dict[str, object]) -> None:
    tag = _svg_tag_name(element)
    state = dict(inherited)
    state.update(_parse_svg_style(element))
    state["transforms"] = inherited["transforms"] + _parse_svg_transforms(element.attrib.get("transform"))

    if tag in {"defs", "clipPath", "mask", "pattern", "metadata", "title", "desc"}:
        return

    if tag in {"svg", "g", "symbol"}:
        for child in element:
            _draw_svg_element(mask, child, state)
        return

    opacity = _parse_svg_number(state.get("opacity"), 1.0)
    if opacity <= 0:
        return

    fill_enabled = _svg_paint_enabled(state.get("fill")) and _parse_svg_number(state.get("fill-opacity"), 1.0) > 0
    stroke_enabled = _svg_paint_enabled(state.get("stroke")) and _parse_svg_number(state.get("stroke-opacity"), 1.0) > 0
    transforms = state["transforms"]
    drawer = ImageDraw.Draw(mask)

    def transform_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        return _apply_svg_transforms(points, transforms)

    def xor_fill(points: list[tuple[float, float]]) -> None:
        if len(points) < 3:
            return
        drawer.polygon(points, fill=1)

    if tag == "rect":
        x = _parse_svg_number(element.attrib.get("x"), 0.0)
        y = _parse_svg_number(element.attrib.get("y"), 0.0)
        width = _parse_svg_number(element.attrib.get("width"), 0.0)
        height = _parse_svg_number(element.attrib.get("height"), 0.0)
        points = transform_points([(x, y), (x + width, y), (x + width, y + height), (x, y + height)])
        if fill_enabled:
            xor_fill(points)
        if stroke_enabled:
            drawer.line(points + [points[0]], fill=1, width=max(1, round(_parse_svg_number(state.get("stroke-width"), 1.0))))
        return

    if tag == "circle" or tag == "ellipse":
        cx = _parse_svg_number(element.attrib.get("cx"), 0.0)
        cy = _parse_svg_number(element.attrib.get("cy"), 0.0)
        rx = _parse_svg_number(element.attrib.get("r"), 0.0) if tag == "circle" else _parse_svg_number(element.attrib.get("rx"), 0.0)
        ry = rx if tag == "circle" else _parse_svg_number(element.attrib.get("ry"), rx)
        points = _sample_svg_ellipse(cx, cy, rx, ry, transforms)
        if fill_enabled and len(points) >= 3:
            drawer.polygon(points, fill=1)
        if stroke_enabled and len(points) > 1:
            drawer.line(points + [points[0]], fill=1, width=max(1, round(_parse_svg_number(state.get("stroke-width"), 1.0))))
        return

    if tag in {"polygon", "polyline"}:
        points_text = element.attrib.get("points", "")
        if not points_text.strip():
            return
        points = transform_points(_parse_svg_points(points_text))
        if tag == "polygon" and fill_enabled:
            xor_fill(points)
        if stroke_enabled or tag == "polyline":
            drawer.line(points + ([points[0]] if tag == "polygon" else []), fill=1, width=max(1, round(_parse_svg_number(state.get("stroke-width"), 1.0))))
        return

    if tag == "line":
        x1 = _parse_svg_number(element.attrib.get("x1"), 0.0)
        y1 = _parse_svg_number(element.attrib.get("y1"), 0.0)
        x2 = _parse_svg_number(element.attrib.get("x2"), 0.0)
        y2 = _parse_svg_number(element.attrib.get("y2"), 0.0)
        drawer.line(transform_points([(x1, y1), (x2, y2)]), fill=1, width=max(1, round(_parse_svg_number(state.get("stroke-width"), 1.0))))
        return

    if tag == "path":
        path_data = element.attrib.get("d", "")
        if not path_data.strip():
            return
        subpaths = [(transform_points(subpath_points), closed) for subpath_points, closed in _parse_svg_path(path_data)]
        if fill_enabled:
            geometry = _svg_subpaths_to_geometry(subpaths, state.get("fill-rule"))
            _draw_svg_geometry(mask, geometry)
        if stroke_enabled:
            for points, _closed in subpaths:
                if len(points) > 1:
                    drawer.line(points, fill=1, width=max(1, round(_parse_svg_number(state.get("stroke-width"), 1.0))))


def _rasterize_svg(image_path: Path) -> Image.Image:
    try:
        root = ET.parse(image_path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Could not parse SVG artwork: {exc}") from exc

    width, height, transforms = _svg_canvas_size(root)
    mask = Image.new("1", (width, height), 0)
    _draw_svg_element(
        mask,
        root,
        {
            "fill": "black",
            "stroke": "none",
            "stroke-width": 1.0,
            "opacity": 1.0,
            "fill-opacity": 1.0,
            "stroke-opacity": 1.0,
            "transforms": transforms,
        },
    )
    return ImageOps.invert(mask.convert("L"))


def _load_artwork_image(image: str | Path) -> Image.Image:
    image_path = Path(image)

    if image_path.suffix.lower() == ".svg":
        return _rasterize_svg(image_path)

    with Image.open(image_path) as raster_image:
        return raster_image.copy()


def load_mask(image: str | Path, options: StampOptions) -> np.ndarray:
    gray = ImageOps.autocontrast(_load_artwork_image(image).convert("L"), cutoff=1)
    coarse = np.asarray(gray) < options.threshold
    coarse = ~coarse if options.invert else coarse
    ys, xs = np.nonzero(coarse)
    if not len(xs):
        raise ValueError("No stamp pixels found. Lower threshold or try invert=True.")
    pad = max(1, round(0.02 * max(gray.size)))
    gray = gray.crop(
        (
            max(0, xs.min() - pad),
            max(0, ys.min() - pad),
            min(gray.size[0], xs.max() + 1 + pad),
            min(gray.size[1], ys.max() + 1 + pad),
        )
    )
    if options.mirror:
        gray = ImageOps.mirror(gray)
    if options.resolution > 0:
        scale = options.resolution / max(gray.size)
        gray = gray.resize(tuple(max(1, round(v * scale)) for v in gray.size), Image.Resampling.LANCZOS)
    mask = np.asarray(gray) < options.threshold
    return ~mask if options.invert else mask


def iter_polygons(geometry) -> list[Polygon]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [poly for geom in geometry.geoms for poly in iter_polygons(geom)]
    return []


def trace_geometry(mask: np.ndarray):
    padded_mask = np.pad(mask.astype(bool), 1, constant_values=False)
    contours = []
    for points in measure.find_contours(padded_mask.astype(float), 0.5, positive_orientation="high"):
        if len(points) < 3:
            continue
        geometry = make_valid(
            Polygon([(col - 1.0, padded_mask.shape[0] - row - 1.0) for row, col in points]).buffer(0)
        )
        contours.extend(iter_polygons(geometry))
    if not contours:
        raise ValueError("No contours found after tracing the image.")
    contours.sort(key=lambda poly: poly.area, reverse=True)
    depths = []
    for index, poly in enumerate(contours):
        point = poly.representative_point()
        parent = next((candidate for candidate in range(index) if contours[candidate].contains(point)), None)
        depths.append(0 if parent is None else depths[parent] + 1)
    geometry = GeometryCollection()
    for depth, poly in sorted(zip(depths, contours), key=lambda item: (item[0], -item[1].area)):
        geometry = geometry.union(poly) if depth % 2 == 0 else geometry.difference(poly)
    return make_valid(geometry)


def _resolve_dimensions(options: StampOptions) -> tuple[float, float]:
    if options.width is not None and options.height is not None:
        return options.width, options.height
    if options.width is not None:
        return options.width, options.width
    if options.height is not None:
        return options.height, options.height
    return options.size, options.size


def validate_dimensions(options: StampOptions) -> tuple[float, float, float, float]:
    width, height = _resolve_dimensions(options)
    inner_width = width - 2 * options.border
    inner_height = height - 2 * options.border
    if inner_width <= 0 or inner_height <= 0:
        raise ValueError("max dimension must be larger than twice border.")
    return width, height, inner_width, inner_height


def _resolve_size(options: StampOptions) -> float:
    width, height = _resolve_dimensions(options)
    return max(width, height)


def validate_size(options: StampOptions) -> float:
    inner_size = _resolve_size(options) - 2 * options.border
    if inner_size <= 0:
        raise ValueError("max dimension must be larger than twice border.")
    return inner_size


def build_vector_mesh(mask: np.ndarray, options: StampOptions) -> trimesh.Trimesh:
    inner_size = validate_size(options)
    geometry = trace_geometry(mask)
    minx, miny, maxx, maxy = geometry.bounds
    scale = inner_size / max(maxx - minx, maxy - miny)
    if options.simplify > 0:
        geometry = make_valid(geometry.simplify(options.simplify / scale, preserve_topology=True))
    geometry = make_valid(scale_geometry(geometry, xfact=scale, yfact=scale, origin=(0, 0)))
    polygons = [poly for poly in iter_polygons(geometry) if poly.area >= options.min_area]
    if not polygons:
        raise ValueError("Tracing removed all geometry. Lower min_area or simplify.")
    geometry = unary_union(polygons)
    minx, miny, maxx, maxy = geometry.bounds
    geometry = translate(geometry, xoff=options.border - minx, yoff=options.border - miny)
    art_w, art_h = geometry.bounds[2] - geometry.bounds[0], geometry.bounds[3] - geometry.bounds[1]
    outer = box(0, 0, art_w + 2 * options.border, art_h + 2 * options.border)
    relief_geometry = iter_polygons(geometry)
    if options.raised_border and options.border > 0:
        relief_geometry.append(
            outer.difference(box(options.border, options.border, options.border + art_w, options.border + art_h))
        )
    base_mesh = trimesh.creation.extrude_polygon(outer, options.base, engine="triangle")
    relief_meshes = [trimesh.creation.extrude_polygon(poly, options.relief, engine="triangle") for poly in relief_geometry]
    for mesh in relief_meshes:
        mesh.apply_translation((0, 0, options.base))
    merged = trimesh.boolean.union([base_mesh, trimesh.util.concatenate(relief_meshes)], engine="manifold")
    if merged is None:
        raise ValueError("Boolean union failed while merging the traced artwork with the base slab.")
    merged.apply_translation(-merged.bounds[0])
    return merged


def build_stamp_mesh_from_mask(mask: np.ndarray, options: StampOptions | None = None) -> trimesh.Trimesh:
    options = options or StampOptions()
    return build_vector_mesh(mask, options)


def build_stamp_mesh(image: str | Path, options: StampOptions | None = None) -> trimesh.Trimesh:
    options = options or StampOptions()
    return build_stamp_mesh_from_mask(load_mask(image, options), options)


def write_stamp(
    image: str | Path,
    output: str | Path | None = None,
    options: StampOptions | None = None,
) -> tuple[Path, trimesh.Trimesh]:
    image_path = Path(image)
    output_path = Path(output) if output else default_output_path(image_path)
    mesh = build_stamp_mesh(image_path, options)
    mesh.export(output_path)
    return output_path, mesh

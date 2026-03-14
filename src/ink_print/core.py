from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh
from PIL import Image, ImageOps
from shapely import unary_union
from shapely.affinity import scale as scale_geometry, translate
from shapely.geometry import GeometryCollection, Polygon, box
from shapely.validation import make_valid
from skimage import measure


@dataclass(slots=True)
class StampOptions:
    mode: Literal["vector", "voxel"] = "vector"
    size: float = 80.0
    border: float = 2.0
    base: float = 4.0
    relief: float = 2.0
    layer: float = 0.5
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


def load_mask(image: str | Path, options: StampOptions) -> np.ndarray:
    gray = ImageOps.autocontrast(Image.open(image).convert("L"), cutoff=1)
    coarse = np.asarray(gray) < options.threshold
    coarse = ~coarse if options.invert else coarse
    ys, xs = np.nonzero(coarse)
    if not len(xs):
        raise ValueError("No stamp pixels found. Lower threshold or try invert=True.")
    gray = gray.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
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
    contours = []
    for points in measure.find_contours(mask.astype(float), 0.5, positive_orientation="high"):
        if len(points) < 3:
            continue
        geometry = make_valid(Polygon([(col, mask.shape[0] - row) for row, col in points]).buffer(0))
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


def validate_size(options: StampOptions) -> float:
    inner_size = options.size - 2 * options.border
    if inner_size <= 0:
        raise ValueError("size must be larger than twice border.")
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


def build_voxel_mesh(mask: np.ndarray, options: StampOptions) -> trimesh.Trimesh:
    inner_size = validate_size(options)
    max_pixels = max(mask.shape)
    border_px = max(0, round(options.border * max_pixels / inner_size))
    xy_pitch = options.size / (max_pixels + 2 * border_px)
    mask = np.pad(mask, border_px, constant_values=False)
    raised = mask.copy()
    if options.raised_border and border_px > 0:
        raised[:border_px, :] = True
        raised[-border_px:, :] = True
        raised[:, :border_px] = True
        raised[:, -border_px:] = True
    base_layers = max(1, round(options.base / options.layer))
    relief_layers = max(1, round(options.relief / options.layer))
    voxels = np.zeros((mask.shape[1], mask.shape[0], base_layers + relief_layers), dtype=bool)
    voxels[:, :, :base_layers] = True
    voxels[:, :, base_layers:] = raised.T[:, :, None]
    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(voxels, pitch=(xy_pitch, xy_pitch, options.layer))
    mesh.apply_translation(-mesh.bounds[0])
    return mesh


def build_stamp_mesh_from_mask(mask: np.ndarray, options: StampOptions | None = None) -> trimesh.Trimesh:
    options = options or StampOptions()
    return build_vector_mesh(mask, options) if options.mode == "vector" else build_voxel_mesh(mask, options)


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

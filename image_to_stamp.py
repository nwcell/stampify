#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageOps
from shapely import unary_union
from shapely.affinity import scale as scale_geometry, translate
from shapely.geometry import GeometryCollection, Polygon, box
from shapely.validation import make_valid
from skimage import measure


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Turn a black-and-white image into an STL stamp plate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("image", type=Path)
    p.add_argument("-o", "--output", type=Path)
    p.add_argument("--mode", choices=("vector", "voxel"), default="vector", help="Geometry pipeline to use.")
    p.add_argument("--size", type=float, default=80.0, help="Longest side of the final stamp, including the border, in mm.")
    p.add_argument("--border", type=float, default=2.0, help="Target border width around the artwork, in mm.")
    p.add_argument("--base", type=float, default=4.0, help="Backing thickness under the raised artwork, in mm.")
    p.add_argument("--relief", type=float, default=2.0, help="Raised height of the inked areas, in mm.")
    p.add_argument("--layer", type=float, default=0.5, help="Z voxel size in mm for --mode voxel.")
    p.add_argument("--simplify", type=float, default=0.08, help="Path simplification tolerance in mm for --mode vector.")
    p.add_argument("--min-area", type=float, default=0.02, help="Discard traced islands smaller than this area, in mm^2.")
    p.add_argument("--threshold", type=int, default=190, help="Pixels darker than this become stamp features.")
    p.add_argument("--resolution", type=int, default=0, help="Longest-side raster resolution for tracing or voxelization. Use 0 to keep the source resolution.")
    p.add_argument(
        "--raised-border",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Raise the outer border ring. Disable with --no-raised-border.",
    )
    p.add_argument("--invert", action="store_true", help="Raise light pixels instead of dark ones.")
    p.add_argument("--no-mirror", action="store_true", help="Skip the horizontal flip.")
    return p.parse_args()


def load_mask(args: argparse.Namespace) -> np.ndarray:
    gray = ImageOps.autocontrast(Image.open(args.image).convert("L"), cutoff=1)
    coarse = np.asarray(gray) < args.threshold
    coarse = ~coarse if args.invert else coarse
    ys, xs = np.nonzero(coarse)
    if not len(xs):
        raise SystemExit("No stamp pixels found. Lower --threshold or try --invert.")
    gray = gray.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    if not args.no_mirror:
        gray = ImageOps.mirror(gray)
    if args.resolution > 0:
        scale = args.resolution / max(gray.size)
        gray = gray.resize(tuple(max(1, round(v * scale)) for v in gray.size), Image.Resampling.LANCZOS)
    mask = np.asarray(gray) < args.threshold
    return ~mask if args.invert else mask


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
        raise SystemExit("No contours found after tracing the image.")
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


def validate_size(args: argparse.Namespace) -> float:
    inner_size = args.size - 2 * args.border
    if inner_size <= 0:
        raise SystemExit("--size must be larger than twice --border.")
    return inner_size


def write_vector_stamp(mask: np.ndarray, args: argparse.Namespace, output: Path) -> None:
    inner_size = validate_size(args)
    geometry = trace_geometry(mask)
    minx, miny, maxx, maxy = geometry.bounds
    scale = inner_size / max(maxx - minx, maxy - miny)
    if args.simplify > 0:
        geometry = make_valid(geometry.simplify(args.simplify / scale, preserve_topology=True))
    geometry = make_valid(scale_geometry(geometry, xfact=scale, yfact=scale, origin=(0, 0)))
    polygons = [poly for poly in iter_polygons(geometry) if poly.area >= args.min_area]
    if not polygons:
        raise SystemExit("Tracing removed all geometry. Lower --min-area or --simplify.")
    geometry = unary_union(polygons)
    minx, miny, maxx, maxy = geometry.bounds
    geometry = translate(geometry, xoff=args.border - minx, yoff=args.border - miny)
    art_w, art_h = geometry.bounds[2] - geometry.bounds[0], geometry.bounds[3] - geometry.bounds[1]
    outer = box(0, 0, art_w + 2 * args.border, art_h + 2 * args.border)
    relief_geometry = iter_polygons(geometry)
    if args.raised_border and args.border > 0:
        relief_geometry.append(outer.difference(box(args.border, args.border, args.border + art_w, args.border + art_h)))
    base_mesh = trimesh.creation.extrude_polygon(outer, args.base, engine="triangle")
    relief_meshes = [trimesh.creation.extrude_polygon(poly, args.relief, engine="triangle") for poly in relief_geometry]
    for mesh in relief_meshes:
        mesh.apply_translation((0, 0, args.base))
    merged = trimesh.boolean.union([base_mesh, trimesh.util.concatenate(relief_meshes)], engine="manifold")
    merged.apply_translation(-merged.bounds[0])
    merged.export(output)
    print(f"Wrote {output} at {merged.extents[0]:.1f} x {merged.extents[1]:.1f} x {merged.extents[2]:.1f} mm")


def write_voxel_stamp(mask: np.ndarray, args: argparse.Namespace, output: Path) -> None:
    inner_size = validate_size(args)
    max_pixels = max(mask.shape)
    border_px = max(0, round(args.border * max_pixels / inner_size))
    xy_pitch = args.size / (max_pixels + 2 * border_px)
    mask = np.pad(mask, border_px, constant_values=False)
    raised = mask.copy()
    if args.raised_border and border_px > 0:
        raised[:border_px, :] = True
        raised[-border_px:, :] = True
        raised[:, :border_px] = True
        raised[:, -border_px:] = True
    base_layers = max(1, round(args.base / args.layer))
    relief_layers = max(1, round(args.relief / args.layer))
    voxels = np.zeros((mask.shape[1], mask.shape[0], base_layers + relief_layers), dtype=bool)
    voxels[:, :, :base_layers] = True
    voxels[:, :, base_layers:] = raised.T[:, :, None]
    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(voxels, pitch=(xy_pitch, xy_pitch, args.layer))
    mesh.apply_translation(-mesh.bounds[0])
    mesh.export(output)
    print(f"Wrote {output} at {mesh.extents[0]:.1f} x {mesh.extents[1]:.1f} x {mesh.extents[2]:.1f} mm")


def main() -> None:
    args = parse_args()
    mask = load_mask(args)
    output = args.output or args.image.with_name(f"{args.image.stem}-stamp.stl")
    if args.mode == "vector":
        write_vector_stamp(mask, args, output)
    else:
        write_voxel_stamp(mask, args, output)


if __name__ == "__main__":
    main()

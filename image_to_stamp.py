#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageOps


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Turn a black-and-white image into an STL stamp plate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("image", type=Path)
    p.add_argument("-o", "--output", type=Path)
    p.add_argument("--size", type=float, default=80.0, help="Longest side of the final stamp, including the border, in mm.")
    p.add_argument("--border", type=float, default=2.0, help="Target border width around the artwork, in mm.")
    p.add_argument("--base", type=float, default=4.0, help="Backing thickness under the raised artwork, in mm.")
    p.add_argument("--relief", type=float, default=2.0, help="Raised height of the inked areas, in mm.")
    p.add_argument("--layer", type=float, default=0.5, help="Z voxel size in mm. Smaller is crisper but heavier.")
    p.add_argument("--threshold", type=int, default=190, help="Pixels darker than this become stamp features.")
    p.add_argument("--resolution", type=int, default=300, help="Longest-side raster resolution for the model.")
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
    scale = args.resolution / max(gray.size)
    gray = gray.resize(tuple(max(1, round(v * scale)) for v in gray.size), Image.Resampling.LANCZOS)
    mask = np.asarray(gray) < args.threshold
    return ~mask if args.invert else mask


def write_stamp(mask: np.ndarray, args: argparse.Namespace, output: Path) -> None:
    max_pixels = max(mask.shape)
    inner_size = args.size - 2 * args.border
    if inner_size <= 0:
        raise SystemExit("--size must be larger than twice --border.")
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
    write_stamp(mask, args, output)


if __name__ == "__main__":
    main()

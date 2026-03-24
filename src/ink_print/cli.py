from __future__ import annotations

import argparse
from pathlib import Path

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Turn a black-and-white image into an STL stamp plate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--mode", choices=("vector", "voxel"), default="vector", help="Geometry pipeline to use.")
    parser.add_argument("--size", type=float, default=80.0, help="Longest side of the final stamp, including the border, in mm.")
    parser.add_argument("--border", type=float, default=2.0, help="Target border width around the artwork, in mm.")
    parser.add_argument("--base", type=float, default=4.0, help="Backing thickness under the raised artwork, in mm.")
    parser.add_argument("--relief", type=float, default=2.0, help="Raised height of the inked areas, in mm.")
    parser.add_argument("--layer", type=float, default=0.5, help="Z voxel size in mm for --mode voxel.")
    parser.add_argument("--simplify", type=float, default=0.08, help="Path simplification tolerance in mm for --mode vector.")
    parser.add_argument("--min-area", type=float, default=0.02, help="Discard traced islands smaller than this area, in mm^2.")
    parser.add_argument("--threshold", type=int, default=190, help="Pixels darker than this become stamp features.")
    parser.add_argument("--resolution", type=int, default=0, help="Longest-side raster resolution for tracing or voxelization. Use 0 to keep the source resolution.")
    parser.add_argument(
        "--raised-border",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Raise the outer border ring. Disable with --no-raised-border.",
    )
    parser.add_argument("--invert", action="store_true", help="Raise light pixels instead of dark ones.")
    parser.add_argument("--no-mirror", action="store_true", help="Skip the horizontal flip.")
    return parser


def namespace_to_options(args: argparse.Namespace) -> StampOptions:
    from .core import StampOptions

    return StampOptions(
        mode=args.mode,
        size=args.size,
        border=args.border,
        base=args.base,
        relief=args.relief,
        layer=args.layer,
        simplify=args.simplify,
        min_area=args.min_area,
        threshold=args.threshold,
        resolution=args.resolution,
        raised_border=args.raised_border,
        invert=args.invert,
        mirror=not args.no_mirror,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    from PIL import UnidentifiedImageError
    from .core import write_stamp
    try:
        output, mesh = write_stamp(args.image, args.output, namespace_to_options(args))
    except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Wrote {output} at {mesh.extents[0]:.1f} x {mesh.extents[1]:.1f} x {mesh.extents[2]:.1f} mm")
    return 0

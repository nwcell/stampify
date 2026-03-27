from __future__ import annotations

import argparse
from pathlib import Path

from .core import StampOptions

DEFAULT_SIZE = 80.0
DEFAULT_WIDTH = None
DEFAULT_HEIGHT = None
DEFAULT_BORDER = 0.5
DEFAULT_BASE = 4.0
DEFAULT_RELIEF = 1.0
DEFAULT_SIMPLIFY = 0.08
DEFAULT_MIN_AREA = 0.02
DEFAULT_THRESHOLD = 190
DEFAULT_RESOLUTION = 0
DEFAULT_RAISED_BORDER = True
DEFAULT_INVERT = False
DEFAULT_MIRROR = True

DEFAULT_OPTIONS = StampOptions(
    size=DEFAULT_SIZE,
    width=DEFAULT_WIDTH,
    height=DEFAULT_HEIGHT,
    border=DEFAULT_BORDER,
    base=DEFAULT_BASE,
    relief=DEFAULT_RELIEF,
    simplify=DEFAULT_SIMPLIFY,
    min_area=DEFAULT_MIN_AREA,
    threshold=DEFAULT_THRESHOLD,
    resolution=DEFAULT_RESOLUTION,
    raised_border=DEFAULT_RAISED_BORDER,
    invert=DEFAULT_INVERT,
    mirror=DEFAULT_MIRROR,
)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Turn artwork into an STL stamp plate. Trace threshold, raster resolution, and cleanup "
            "defaults are auto-tuned from the upload unless you override them."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--size", type=float, default=DEFAULT_SIZE, help="Legacy square stamp size fallback, in mm, used when width/height are not provided.")
    parser.add_argument("--width", type=float, default=DEFAULT_WIDTH, help="Maximum stamp width, in mm. Leave unset to defer to the other dimension, or to --size if both are omitted.")
    parser.add_argument("--height", type=float, default=DEFAULT_HEIGHT, help="Maximum stamp height, in mm. Leave unset to defer to the other dimension, or to --size if both are omitted.")
    parser.add_argument("--border", type=float, default=DEFAULT_BORDER, help="Target border width around the artwork, in mm.")
    parser.add_argument("--base", type=float, default=DEFAULT_BASE, help="Backing thickness under the raised artwork, in mm.")
    parser.add_argument("--relief", type=float, default=DEFAULT_RELIEF, help="Raised height of the inked areas, in mm.")
    parser.add_argument(
        "--simplify",
        type=float,
        default=DEFAULT_SIMPLIFY,
        help="Path simplification tolerance in mm. Leave at the default to auto-tune from the traced geometry.",
    )
    parser.add_argument("--min-area", type=float, default=DEFAULT_MIN_AREA, help="Discard traced islands smaller than this area, in mm^2.")
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD,
        help="Grayscale cutoff for raster artwork. Leave at the default to auto-pick a threshold from the image.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help="Longest-side raster resolution for tracing. Leave at the default to auto-pick a size from the artwork.",
    )
    parser.add_argument(
        "--raised-border",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RAISED_BORDER,
        help="Raise the outer border ring. Disable with --no-raised-border.",
    )
    parser.add_argument("--invert", action="store_true", default=DEFAULT_INVERT, help="Raise light pixels instead of dark ones.")
    parser.add_argument("--no-mirror", action="store_true", default=not DEFAULT_MIRROR, help="Skip the horizontal flip.")
    return parser


def namespace_to_options(args: argparse.Namespace) -> StampOptions:
    return StampOptions(
        size=args.size if args.size is not None else DEFAULT_SIZE,
        width=args.width if args.width is not None else DEFAULT_WIDTH,
        height=args.height if args.height is not None else DEFAULT_HEIGHT,
        border=args.border if args.border is not None else DEFAULT_BORDER,
        base=args.base if args.base is not None else DEFAULT_BASE,
        relief=args.relief if args.relief is not None else DEFAULT_RELIEF,
        simplify=args.simplify if args.simplify is not None else DEFAULT_SIMPLIFY,
        min_area=args.min_area if args.min_area is not None else DEFAULT_MIN_AREA,
        threshold=args.threshold if args.threshold is not None else DEFAULT_THRESHOLD,
        resolution=args.resolution if args.resolution is not None else DEFAULT_RESOLUTION,
        raised_border=args.raised_border if args.raised_border is not None else DEFAULT_RAISED_BORDER,
        invert=args.invert if args.invert is not None else DEFAULT_INVERT,
        mirror=not args.no_mirror if args.no_mirror is not None else DEFAULT_MIRROR,
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

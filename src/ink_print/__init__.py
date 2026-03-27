from __future__ import annotations

from importlib import import_module

__all__ = [
    "StampOptions",
    "build_stamp_mesh",
    "build_stamp_mesh_from_geometry",
    "build_stamp_mesh_from_mask",
    "build_stamp_svg",
    "default_output_path",
    "load_mask",
    "prepare_stamp_geometry",
    "resolve_artwork",
    "resolve_trace_profile",
    "write_stamp",
]


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(".core", __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)

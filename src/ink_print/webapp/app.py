from __future__ import annotations

import os
import secrets
import shutil
import time
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.background import BackgroundTask

from ink_print.core import StampOptions, build_stamp_mesh_from_mask, load_mask, trace_geometry

APP_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(APP_DIR / "templates"))
app = FastAPI(
    title="Stampify Studio",
    version="0.1.0",
    description="Guided browser UI for turning artwork into stamp STL files.",
)

DEFAULT_OPTIONS = StampOptions()


@dataclass(slots=True)
class WebSession:
    token: str
    stage: Literal["preview", "approved", "result"]
    created_at: float
    last_accessed: float
    workspace: Path
    upload_path: Path
    upload_name: str
    options: StampOptions
    preview_svg: str
    preview_info: dict[str, str]
    result_path: Path | None = None
    result_info: dict[str, str] | None = None


_SESSIONS: dict[str, WebSession] = {}
_SESSION_TTL_SECONDS = 24 * 60 * 60


def _human_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def _format_mm(value: float) -> str:
    return f"{value:.1f} mm"


def _build_options(
    mode: Literal["vector", "voxel"],
    size: float,
    border: float,
    base: float,
    relief: float,
    threshold: int,
    resolution: int,
    simplify: float,
    raised_border: bool,
    invert: bool,
) -> StampOptions:
    return StampOptions(
        mode=mode,
        size=size,
        border=border,
        base=base,
        relief=relief,
        layer=DEFAULT_OPTIONS.layer,
        simplify=simplify,
        min_area=DEFAULT_OPTIONS.min_area,
        threshold=threshold,
        resolution=resolution,
        raised_border=raised_border,
        invert=invert,
        mirror=DEFAULT_OPTIONS.mirror,
    )


def _iter_polygons(geometry) -> list:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if hasattr(geometry, "geoms"):
        polygons: list = []
        for geom in geometry.geoms:
            polygons.extend(_iter_polygons(geom))
        return polygons
    return []


def _svg_path_from_polygon(polygon, minx: float, maxy: float, pad: float) -> str:
    def convert(coords: list[tuple[float, float]]) -> str:
        return " ".join(f"{x - minx + pad:.2f},{maxy - y + pad:.2f}" for x, y in coords)

    parts = [f"M {convert(list(polygon.exterior.coords))} Z"]
    for ring in polygon.interiors:
        parts.append(f"M {convert(list(ring.coords))} Z")
    return " ".join(parts)


def _render_preview_svg(geometry) -> str:
    polygons = _iter_polygons(geometry)
    if not polygons:
        raise ValueError("No contours were found in the artwork.")

    minx, _miny, maxx, maxy = geometry.bounds
    width = max(maxx - minx, 1.0)
    height = max(maxy - _miny, 1.0)
    pad = max(width, height) * 0.08
    total_width = width + 2 * pad
    total_height = height + 2 * pad
    paths = [
        _svg_path_from_polygon(polygon, minx, maxy, pad)
        for polygon in polygons
    ]

    return f"""
    <svg viewBox=\"0 0 {total_width:.2f} {total_height:.2f}\" xmlns=\"http://www.w3.org/2000/svg\" role=\"img\" aria-label=\"Vector preview\">
      <defs>
        <linearGradient id=\"stamp-gradient\" x1=\"0%\" y1=\"0%\" x2=\"100%\" y2=\"100%\">
          <stop offset=\"0%\" stop-color=\"#7c3aed\" />
          <stop offset=\"100%\" stop-color=\"#22d3ee\" />
        </linearGradient>
        <filter id=\"preview-shadow\" x=\"-20%\" y=\"-20%\" width=\"140%\" height=\"140%\">
          <feDropShadow dx=\"0\" dy=\"6\" stdDeviation=\"8\" flood-color=\"#000000\" flood-opacity=\"0.28\" />
        </filter>
        <pattern id=\"preview-grid\" width=\"24\" height=\"24\" patternUnits=\"userSpaceOnUse\">
          <path d=\"M 24 0 L 0 0 0 24\" fill=\"none\" stroke=\"rgba(255,255,255,.07)\" stroke-width=\"1\" />
        </pattern>
      </defs>
      <rect x=\"0\" y=\"0\" width=\"100%\" height=\"100%\" rx=\"24\" fill=\"#101828\" />
      <rect x=\"0\" y=\"0\" width=\"100%\" height=\"100%\" rx=\"24\" fill=\"url(#preview-grid)\" opacity=\"0.8\" />
      <g filter=\"url(#preview-shadow)\">
        <path d=\"{' '.join(paths)}\" fill=\"url(#stamp-gradient)\" fill-rule=\"evenodd\" stroke=\"rgba(255,255,255,.95)\" stroke-width=\"1.6\" stroke-linejoin=\"round\" />
      </g>
    </svg>
    """.strip()


def _cleanup_session(token: str) -> None:
    session = _SESSIONS.pop(token, None)
    if session is None:
        return
    shutil.rmtree(session.workspace, ignore_errors=True)


def _touch_session(session: WebSession) -> None:
    session.last_accessed = time.time()


def _prune_expired_sessions() -> None:
    now = time.time()
    stale_tokens = [
        token for token, session in _SESSIONS.items() if now - session.last_accessed > _SESSION_TTL_SECONDS
    ]
    for token in stale_tokens:
        _cleanup_session(token)


def _get_session(token: str) -> WebSession | None:
    _prune_expired_sessions()
    session = _SESSIONS.get(token)
    if session is not None:
        _touch_session(session)
    return session


def _render_page(request: Request, session: WebSession | None = None, error: str | None = None) -> HTMLResponse:
    return TEMPLATES.TemplateResponse(
        request,
        "index.html",
        {
            "session": session,
            "defaults": DEFAULT_OPTIONS,
            "error": error,
        },
    )


async def _store_upload(upload: UploadFile) -> tuple[Path, str]:
    if not upload.filename:
        raise ValueError("Please choose an image file before generating a preview.")

    payload = await upload.read()
    if not payload:
        raise ValueError("The uploaded file was empty.")

    workspace = Path(tempfile.mkdtemp(prefix="stampify-web-"))
    upload_name = Path(upload.filename).name or "upload.png"
    upload_path = workspace / upload_name
    upload_path.write_bytes(payload)
    return workspace, upload_name


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return _render_page(request)


@app.post("/preview", response_class=HTMLResponse)
async def preview(
    request: Request,
    artwork: UploadFile = File(...),
    mode: Literal["vector", "voxel"] = Form("vector"),
    size: float = Form(80.0),
    border: float = Form(2.0),
    base: float = Form(4.0),
    relief: float = Form(2.0),
    threshold: int = Form(190),
    resolution: int = Form(0),
    simplify: float = Form(0.08),
    raised_border: bool = Form(False),
    invert: bool = Form(False),
) -> HTMLResponse:
    options = _build_options(mode, size, border, base, relief, threshold, resolution, simplify, raised_border, invert)
    workspace: Path | None = None

    try:
        workspace, upload_name = await _store_upload(artwork)
        mask = load_mask(workspace / upload_name, options)
        geometry = trace_geometry(mask)
    except (FileNotFoundError, OSError, ValueError) as exc:
        if workspace is not None:
            shutil.rmtree(workspace, ignore_errors=True)
        return _render_page(request, error=str(exc))

    token = secrets.token_urlsafe(16)
    preview_svg = _render_preview_svg(geometry)
    preview_info = {
        "contours": str(len(_iter_polygons(geometry))),
        "area": f"{geometry.area:.2f} px²",
        "bounds": f"{geometry.bounds[2] - geometry.bounds[0]:.1f} px × {geometry.bounds[3] - geometry.bounds[1]:.1f} px",
        "mode": options.mode,
    }
    session = WebSession(
        token=token,
        stage="preview",
        created_at=time.time(),
        last_accessed=time.time(),
        workspace=workspace,
        upload_path=workspace / upload_name,
        upload_name=upload_name,
        options=options,
        preview_svg=preview_svg,
        preview_info=preview_info,
    )
    _SESSIONS[token] = session
    return _render_page(request, session=session)


@app.post("/approve", response_class=HTMLResponse)
def approve(request: Request, token: str = Form(...)) -> HTMLResponse:
    session = _get_session(token)
    if session is None:
        return _render_page(request, error="That preview session expired. Please generate a new preview.")

    session.stage = "approved"
    return _render_page(request, session=session)


@app.post("/generate", response_class=HTMLResponse)
def generate(request: Request, token: str = Form(...)) -> HTMLResponse:
    session = _get_session(token)
    if session is None:
        return _render_page(request, error="That preview session expired. Please generate a new preview.")

    try:
        mask = load_mask(session.upload_path, session.options)
        mesh = build_stamp_mesh_from_mask(mask, session.options)
        result_path = session.workspace / f"{session.upload_path.stem}-stamp.stl"
        mesh.export(result_path)
    except (FileNotFoundError, OSError, ValueError) as exc:
        return _render_page(request, error=str(exc), session=session)

    session.stage = "result"
    _touch_session(session)
    session.result_path = result_path
    session.result_info = {
        "dimensions": f"{mesh.extents[0]:.1f} × {mesh.extents[1]:.1f} × {mesh.extents[2]:.1f} mm",
        "size": _human_size(result_path.stat().st_size),
        "mode": session.options.mode,
    }
    return _render_page(request, session=session)


@app.get("/artifact/{token}/mesh", name="mesh_view")
def mesh_view(token: str) -> FileResponse:
    session = _get_session(token)
    if session is None or session.result_path is None or not session.result_path.exists():
        raise HTTPException(status_code=404, detail="That generated model is no longer available.")

    return FileResponse(session.result_path, media_type="model/stl")


@app.get("/artifact/{token}/download", name="download_model")
def download_model(token: str) -> FileResponse:
    session = _get_session(token)
    if session is None or session.result_path is None or not session.result_path.exists():
        raise HTTPException(status_code=404, detail="That generated model is no longer available.")

    return FileResponse(
        session.result_path,
        media_type="model/stl",
        filename=session.result_path.name,
        background=BackgroundTask(_cleanup_session, token),
    )


def main() -> int:
    uvicorn.run(
        app,
        host=os.environ.get("STAMPIFY_WEB_HOST", "127.0.0.1"),
        port=int(os.environ.get("STAMPIFY_WEB_PORT", "8000")),
        log_level=os.environ.get("STAMPIFY_WEB_LOG_LEVEL", "info"),
    )
    return 0

from __future__ import annotations

from base64 import b64encode
import mimetypes
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from ink_print.cli import DEFAULT_OPTIONS as CLI_DEFAULT_OPTIONS
from ink_print.core import (
    StampOptions,
    build_stamp_mesh_from_geometry,
    resolve_artwork,
    iter_polygons,
    validate_dimensions,
)

APP_DIR = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent.parent.parent
SAMPLE_ARTWORK_PATH = REPO_DIR / "sample" / "xmas-cowboy.jpeg"
TEMPLATES = Jinja2Templates(directory=str(APP_DIR / "templates"))
app = FastAPI(
    title="Stampify Studio",
    version="0.1.0",
    description="Guided browser UI for turning artwork into stamp STL files.",
)

DEFAULT_OPTIONS = CLI_DEFAULT_OPTIONS
WEBAPP_IMPORT = "ink_print.webapp.app:app"


@dataclass(slots=True)
class DraftValues:
    size: str
    width: str
    height: str
    border: str
    base: str
    relief: str
    raised_border: bool
    invert: bool


@dataclass(slots=True)
class WorkflowState:
    render_stage: Literal["idle", "preview", "result"]
    artwork_name: str
    artwork_data_url: str
    preview_svg: str
    preview_info: dict[str, str]
    result_name: str | None = None
    result_data_url: str | None = None
    result_info: dict[str, str] | None = None


def _guess_media_type(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _human_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def _parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


def _parse_float(value: str | None, default: float) -> float:
    parsed = _parse_optional_float(value)
    return default if parsed is None else parsed


def _data_url(media_type: str, payload: bytes) -> str:
    return f"data:{media_type};base64,{b64encode(payload).decode('ascii')}"


if SAMPLE_ARTWORK_PATH.exists():
    SAMPLE_ARTWORK_DATA_URL = _data_url(
        _guess_media_type(SAMPLE_ARTWORK_PATH),
        SAMPLE_ARTWORK_PATH.read_bytes(),
    )
else:
    SAMPLE_ARTWORK_DATA_URL = ""


def _form_values_from_defaults(render_stage: Literal["idle", "preview", "result"] = "idle") -> DraftValues:
    placeholder = str(DEFAULT_OPTIONS.size) if render_stage == "idle" else ""
    return DraftValues(
        size=str(DEFAULT_OPTIONS.size),
        width=placeholder if DEFAULT_OPTIONS.width is None else f"{DEFAULT_OPTIONS.width}",
        height=placeholder if DEFAULT_OPTIONS.height is None else f"{DEFAULT_OPTIONS.height}",
        border=f"{DEFAULT_OPTIONS.border}",
        base=f"{DEFAULT_OPTIONS.base}",
        relief=f"{DEFAULT_OPTIONS.relief}",
        raised_border=DEFAULT_OPTIONS.raised_border,
        invert=DEFAULT_OPTIONS.invert,
    )


def _form_values_from_submission(
    render_stage: Literal["idle", "preview", "result"],
    width: str | None,
    height: str | None,
    border: str | None,
    base: str | None,
    relief: str | None,
    raised_border: str | None,
    invert: str | None,
) -> DraftValues:
    placeholder = str(DEFAULT_OPTIONS.size) if render_stage == "idle" else ""
    return DraftValues(
        size=str(DEFAULT_OPTIONS.size),
        width=width if width is not None else placeholder,
        height=height if height is not None else placeholder,
        border=border if border is not None else f"{DEFAULT_OPTIONS.border}",
        base=base if base is not None else f"{DEFAULT_OPTIONS.base}",
        relief=relief if relief is not None else f"{DEFAULT_OPTIONS.relief}",
        raised_border=bool(raised_border and raised_border.strip()),
        invert=bool(invert and invert.strip()),
    )


def _build_options(values: DraftValues) -> StampOptions:
    return StampOptions(
        size=DEFAULT_OPTIONS.size,
        width=_parse_optional_float(values.width),
        height=_parse_optional_float(values.height),
        border=_parse_float(values.border, DEFAULT_OPTIONS.border),
        base=_parse_float(values.base, DEFAULT_OPTIONS.base),
        relief=_parse_float(values.relief, DEFAULT_OPTIONS.relief),
        simplify=DEFAULT_OPTIONS.simplify,
        min_area=DEFAULT_OPTIONS.min_area,
        threshold=DEFAULT_OPTIONS.threshold,
        resolution=DEFAULT_OPTIONS.resolution,
        raised_border=values.raised_border,
        invert=values.invert,
        mirror=DEFAULT_OPTIONS.mirror,
    )


def _build_preview_artifacts(image_path: Path, options: StampOptions) -> tuple[object, str, dict[str, str]]:
    preview_width, preview_height, _, _ = validate_dimensions(options)
    resolved = resolve_artwork(image_path, options)
    preview_info = {
        "contours": str(len(iter_polygons(resolved.geometry))),
        "area": f"{resolved.geometry.area:.2f} mm²",
        "bounds": f"{resolved.geometry.bounds[2] - resolved.geometry.bounds[0]:.1f} mm × {resolved.geometry.bounds[3] - resolved.geometry.bounds[1]:.1f} mm",
        "dimensions": f"{preview_width:.1f} × {preview_height:.1f} mm",
    }
    return resolved, resolved.preview_svg, preview_info


def _render_page(
    request: Request,
    *,
    session: WorkflowState | None = None,
    defaults: DraftValues | None = None,
    error: str | None = None,
    auto_scroll_result: bool = False,
) -> HTMLResponse:
    response = TEMPLATES.TemplateResponse(
        request,
        "index.html",
        {
            "session": session,
            "defaults": defaults or _form_values_from_defaults(),
            "sample_artwork_data_url": SAMPLE_ARTWORK_DATA_URL,
            "error": error,
            "auto_scroll_result": auto_scroll_result,
        },
    )
    return response


async def _read_upload_artwork(upload: UploadFile) -> tuple[bytes, str, str]:
    if not upload.filename:
        raise ValueError("Please choose an image file before preparing the SVG.")

    payload = await upload.read()
    if not payload:
        raise ValueError("The uploaded file was empty.")

    upload_name = Path(upload.filename).name or "upload.png"
    media_type = upload.content_type or _guess_media_type(Path(upload_name))
    return payload, upload_name, media_type


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return _render_page(request)


@app.get("/sample/xmas-cowboy.jpeg", name="sample_artwork")
def sample_artwork() -> FileResponse:
    if not SAMPLE_ARTWORK_PATH.exists():
        raise HTTPException(status_code=404, detail="That sample artwork is no longer available.")

    return FileResponse(SAMPLE_ARTWORK_PATH, media_type=_guess_media_type(SAMPLE_ARTWORK_PATH))


@app.post("/preview", response_class=HTMLResponse)
async def preview(
    request: Request,
    artwork: UploadFile = File(...),
    width: str = Form(""),
    height: str = Form(""),
    border: str = Form(str(DEFAULT_OPTIONS.border)),
    base: str = Form(str(DEFAULT_OPTIONS.base)),
    relief: str = Form(str(DEFAULT_OPTIONS.relief)),
    raised_border: str = Form(""),
    invert: str = Form(""),
) -> HTMLResponse:
    form_values = _form_values_from_submission(
        "idle",
        width,
        height,
        border,
        base,
        relief,
        raised_border,
        invert,
    )

    try:
        payload, upload_name, artwork_media_type = await _read_upload_artwork(artwork)
    except (FileNotFoundError, OSError, ValueError) as exc:
        return _render_page(request, defaults=form_values, error=str(exc))

    artwork_data_url = _data_url(artwork_media_type, payload)

    try:
        options = _build_options(form_values)
    except (TypeError, ValueError):
        return _render_page(
            request,
            defaults=form_values,
            error="Please enter valid numbers in the settings.",
            session=WorkflowState(
                render_stage="preview",
                artwork_name=upload_name,
                artwork_data_url=artwork_data_url,
                preview_svg="",
                preview_info={},
            ),
        )

    if options.width is None and options.height is None:
        return _render_page(
            request,
            defaults=form_values,
            error="Please enter at least one max dimension: width, height, or both.",
            session=WorkflowState(
                render_stage="preview",
                artwork_name=upload_name,
                artwork_data_url=artwork_data_url,
                preview_svg="",
                preview_info={},
            ),
        )

    with tempfile.TemporaryDirectory(prefix="stampify-web-") as tmpdir:
        upload_path = Path(tmpdir) / upload_name
        upload_path.write_bytes(payload)
        try:
            _resolved, preview_svg, preview_info = _build_preview_artifacts(upload_path, options)
        except (FileNotFoundError, OSError, ValueError) as exc:
            return _render_page(
                request,
                defaults=form_values,
                error=str(exc),
                session=WorkflowState(
                    render_stage="preview",
                    artwork_name=upload_name,
                    artwork_data_url=artwork_data_url,
                    preview_svg="",
                    preview_info={},
                ),
            )

    session = WorkflowState(
        render_stage="preview",
        artwork_name=upload_name,
        artwork_data_url=artwork_data_url,
        preview_svg=preview_svg,
        preview_info=preview_info,
    )
    return _render_page(request, session=session, defaults=form_values)


@app.post("/generate", response_class=HTMLResponse)
async def generate(
    request: Request,
    artwork: UploadFile = File(...),
    width: str = Form(""),
    height: str = Form(""),
    border: str = Form(str(DEFAULT_OPTIONS.border)),
    base: str = Form(str(DEFAULT_OPTIONS.base)),
    relief: str = Form(str(DEFAULT_OPTIONS.relief)),
    raised_border: str = Form(""),
    invert: str = Form(""),
) -> HTMLResponse:
    form_values = _form_values_from_submission(
        "idle",
        width,
        height,
        border,
        base,
        relief,
        raised_border,
        invert,
    )

    try:
        payload, upload_name, artwork_media_type = await _read_upload_artwork(artwork)
    except (FileNotFoundError, OSError, ValueError) as exc:
        return _render_page(request, defaults=form_values, error=str(exc))

    artwork_data_url = _data_url(artwork_media_type, payload)

    try:
        options = _build_options(form_values)
    except (TypeError, ValueError):
        return _render_page(
            request,
            defaults=form_values,
            error="Please enter valid numbers in the settings.",
            session=WorkflowState(
                render_stage="preview",
                artwork_name=upload_name,
                artwork_data_url=artwork_data_url,
                preview_svg="",
                preview_info={},
            ),
        )

    if options.width is None and options.height is None:
        return _render_page(
            request,
            defaults=form_values,
            error="Please enter at least one max dimension: width, height, or both.",
            session=WorkflowState(
                render_stage="preview",
                artwork_name=upload_name,
                artwork_data_url=artwork_data_url,
                preview_svg="",
                preview_info={},
            ),
        )

    with tempfile.TemporaryDirectory(prefix="stampify-web-") as tmpdir:
        upload_path = Path(tmpdir) / upload_name
        upload_path.write_bytes(payload)
        try:
            resolved, preview_svg, preview_info = _build_preview_artifacts(upload_path, options)
            mesh = build_stamp_mesh_from_geometry(resolved.geometry, options, prepared=True)
            stl_bytes = mesh.export(file_type="stl")
        except (FileNotFoundError, OSError, ValueError) as exc:
            return _render_page(
                request,
                defaults=form_values,
                error=str(exc),
                session=WorkflowState(
                    render_stage="preview",
                    artwork_name=upload_name,
                    artwork_data_url=artwork_data_url,
                    preview_svg=preview_svg if "preview_svg" in locals() else "",
                    preview_info=preview_info if "preview_info" in locals() else {},
                ),
            )

    result_name = f"{Path(upload_name).stem}-stamp.stl"
    session = WorkflowState(
        render_stage="result",
        artwork_name=upload_name,
        artwork_data_url=artwork_data_url,
        preview_svg=preview_svg,
        preview_info=preview_info,
        result_name=result_name,
        result_data_url=_data_url("model/stl", stl_bytes),
        result_info={
            "dimensions": f"{mesh.extents[0]:.1f} × {mesh.extents[1]:.1f} × {mesh.extents[2]:.1f} mm",
            "size": _human_size(len(stl_bytes)),
        },
    )
    return _render_page(request, session=session, defaults=form_values, auto_scroll_result=True)


def main() -> int:
    uvicorn.run(
        WEBAPP_IMPORT,
        host=os.environ.get("STAMPIFY_WEB_HOST", "127.0.0.1"),
        port=int(os.environ.get("STAMPIFY_WEB_PORT", "8000")),
        log_level=os.environ.get("STAMPIFY_WEB_LOG_LEVEL", "info"),
        reload=True,
        reload_dirs=[str(APP_DIR.parent.parent)],
    )
    return 0

from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

from ink_print.cli import DEFAULT_OPTIONS as CLI_DEFAULT_OPTIONS
import ink_print.webapp.app as webapp_app
from ink_print.webapp.app import app

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample" / "xmas-cowboy.jpeg"


def _preview_payload() -> dict[str, object]:
    return {
        "width": "90",
        "height": "",
        "border": "0.5",
        "base": "4",
        "relief": "1",
        "threshold": "190",
        "resolution": "0",
        "simplify": "0.08",
        "raised_border": "on",
        "invert": "",
    }


def _extract_token(html: str) -> str:
    match = re.search(r'name="token" value="([^"]+)"', html)
    if not match:
        raise AssertionError("Expected a session token in the rendered HTML")
    return match.group(1)


def test_webapp_direct_preview_to_generation_flow() -> None:
    client = TestClient(app)

    root = client.get("/")
    assert root.status_code == 200
    assert "Preview" in root.text
    assert "Stampify Studio" in root.text
    assert f'value="{CLI_DEFAULT_OPTIONS.threshold}"' in root.text
    assert f'value="{CLI_DEFAULT_OPTIONS.resolution}"' in root.text
    assert f'value="{CLI_DEFAULT_OPTIONS.border}"' in root.text
    assert f'value="{CLI_DEFAULT_OPTIONS.relief}"' in root.text
    assert 'name="width" value="80.0"' in root.text
    assert 'name="height" value="80.0"' in root.text
    assert "width, height, or both" in root.text.lower()

    with SAMPLE.open("rb") as image:
        preview = client.post(
            "/preview",
            data=_preview_payload(),
            files={"artwork": (SAMPLE.name, image, "image/jpeg")},
        )
    assert preview.status_code == 200
    assert "Generate STL" in preview.text
    assert "Preview" in preview.text
    assert "name=\"token\"" in preview.text
    assert "90.0 × 90.0 mm" in preview.text
    assert '<rect x="0" y="0" width="100%" height="100%" fill="#fff" />' in preview.text
    assert 'fill="#000" fill-rule="evenodd" stroke="#000"' in preview.text
    assert "linearGradient" not in preview.text
    assert "feDropShadow" not in preview.text
    assert "Max width" in root.text
    assert "Max height" in root.text
    assert "mm" in root.text

    token = _extract_token(preview.text)

    preview_view = client.get(f"/?token={token}&view=preview")
    assert preview_view.status_code == 200
    assert "Preview" in preview_view.text
    assert "Generate STL" in preview_view.text
    assert "90.0 × 90.0 mm" in preview_view.text

    upload_view = client.get(f"/?token={token}&view=upload")
    assert upload_view.status_code == 200
    assert "Session saved" in upload_view.text
    assert "Reset to blank" in upload_view.text
    assert "Generate STL" not in upload_view.text

    generated = client.post("/generate", data={"token": token})
    assert generated.status_code == 200
    assert "Result" in generated.text
    assert "Download STL" in generated.text
    assert "data-mesh-url" in generated.text

    result_view = client.get(f"/?token={token}&view=result")
    assert result_view.status_code == 200
    assert "Result" in result_view.text
    assert "Download STL" in result_view.text

    reset_view = client.post("/reset", data={"token": token})
    assert reset_view.status_code == 200
    assert "Session saved" not in reset_view.text
    assert "Reset to blank" not in reset_view.text
    assert "Generate STL" not in reset_view.text

    expired_generate = client.post("/generate", data={"token": token})
    assert expired_generate.status_code == 200
    assert "That preview session expired" in expired_generate.text


def test_webapp_main_enables_reload_by_default(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(*args, **kwargs) -> None:
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(webapp_app.uvicorn, "run", fake_run)

    assert webapp_app.main() == 0
    assert captured["args"] == (webapp_app.WEBAPP_IMPORT,)
    assert captured["kwargs"]["reload"] is True
    assert captured["kwargs"]["reload_dirs"] == [str(webapp_app.APP_DIR.parent.parent)]


def test_webapp_exposes_reload_boot_id() -> None:
    client = TestClient(app)

    root = client.get("/")
    assert root.status_code == 200

    meta_match = re.search(r'<meta name="stampify-boot-id" content="([^"]+)" />', root.text)
    assert meta_match is not None

    reload_response = client.get("/__reload-id")
    assert reload_response.status_code == 200
    assert reload_response.json()["boot_id"] == meta_match.group(1)
    assert "/__reload-id" in root.text
    assert "window.location.reload()" in root.text


def test_webapp_accepts_svg_artwork() -> None:
    client = TestClient(app)

    svg_artwork = """<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 100 100\">
      <rect x=\"20\" y=\"20\" width=\"60\" height=\"60\" fill=\"black\" />
    </svg>"""

    preview = client.post(
        "/preview",
        data=_preview_payload(),
        files={"artwork": ("badge.svg", svg_artwork.encode("utf-8"), "image/svg+xml")},
    )
    assert preview.status_code == 200
    assert "Generate STL" in preview.text
    assert "Preview" in preview.text
    assert "name=\"token\"" in preview.text

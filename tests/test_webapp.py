from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

from ink_print.cli import DEFAULT_OPTIONS as CLI_DEFAULT_OPTIONS
from ink_print.webapp.app import app

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample" / "xmas-cowboy.jpeg"


def _preview_payload() -> dict[str, object]:
    return {
        "mode": "vector",
        "width": "90",
        "height": "70",
        "border": "2",
        "base": "4",
        "relief": "2",
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
    assert "Generate vector preview" in root.text
    assert "Stampify Studio" in root.text
    assert f'value="{CLI_DEFAULT_OPTIONS.width}"' in root.text
    assert f'value="{CLI_DEFAULT_OPTIONS.height}"' in root.text

    with SAMPLE.open("rb") as image:
        preview = client.post(
            "/preview",
            data=_preview_payload(),
            files={"artwork": (SAMPLE.name, image, "image/jpeg")},
        )
    assert preview.status_code == 200
    assert "Generate STL" in preview.text
    assert "Vector preview" in preview.text
    assert "name=\"token\"" in preview.text
    assert "Max width" in root.text
    assert "Max height" in root.text

    token = _extract_token(preview.text)

    preview_view = client.get(f"/?token={token}&view=preview")
    assert preview_view.status_code == 200
    assert "Vector preview" in preview_view.text
    assert "Generate STL" in preview_view.text

    generated = client.post("/generate", data={"token": token})
    assert generated.status_code == 200
    assert "Interactive 3D preview" in generated.text
    assert "Download STL" in generated.text
    assert "data-mesh-url" in generated.text

    result_view = client.get(f"/?token={token}&view=result")
    assert result_view.status_code == 200
    assert "Interactive 3D preview" in result_view.text
    assert "Download STL" in result_view.text


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
    assert "Vector preview" in preview.text
    assert "name=\"token\"" in preview.text

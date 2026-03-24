from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

from ink_print.webapp.app import app

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample" / "xmas-cowboy.jpeg"


def _preview_payload() -> dict[str, object]:
    return {
        "mode": "vector",
        "size": "80",
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


def test_webapp_guided_flow_renders_preview_and_result() -> None:
    client = TestClient(app)

    root = client.get("/")
    assert root.status_code == 200
    assert "Generate vector preview" in root.text
    assert "Stampify Studio" in root.text

    with SAMPLE.open("rb") as image:
        preview = client.post(
            "/preview",
            data=_preview_payload(),
            files={"artwork": (SAMPLE.name, image, "image/jpeg")},
        )
    assert preview.status_code == 200
    assert "Approve vector" in preview.text
    assert "Vector preview" in preview.text
    assert "name=\"token\"" in preview.text

    token = _extract_token(preview.text)

    approved = client.post("/approve", data={"token": token})
    assert approved.status_code == 200
    assert "Generate STL" in approved.text
    assert "Vector approved" in approved.text

    generated = client.post("/generate", data={"token": token})
    assert generated.status_code == 200
    assert "Interactive 3D preview" in generated.text
    assert "Download STL" in generated.text
    assert "data-mesh-url" in generated.text

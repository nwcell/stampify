from __future__ import annotations

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


def test_webapp_preview_and_generate_are_stateless() -> None:
    client = TestClient(app)

    root = client.get("/")
    assert root.status_code == 200
    assert 'data-render-stage="idle"' in root.text
    assert "Draft saved" not in root.text
    assert "Generate STL" in root.text
    assert "No trace yet" in root.text
    assert 'stampify-boot-id' not in root.text
    assert '/__reload-id' not in root.text
    assert '/reset' not in root.text
    assert str(app.url_path_for("sample_artwork")) not in root.text
    assert 'data-start-over-button="true"' in root.text
    assert "HX-Push-Url" not in root.headers
    assert f'value="{CLI_DEFAULT_OPTIONS.threshold}"' in root.text
    assert f'value="{CLI_DEFAULT_OPTIONS.resolution}"' in root.text
    assert f'value="{CLI_DEFAULT_OPTIONS.border}"' in root.text
    assert f'value="{CLI_DEFAULT_OPTIONS.relief}"' in root.text
    assert 'data:image/jpeg;base64,' in root.text

    with SAMPLE.open("rb") as image:
        preview = client.post(
            "/preview",
            data=_preview_payload(),
            files={"artwork": (SAMPLE.name, image, "image/jpeg")},
        )

    assert preview.status_code == 200
    assert 'data-render-stage="preview"' in preview.text
    assert "Draft saved" in preview.text
    assert 'data-artwork-url="data:image/jpeg;base64,' in preview.text
    assert "xmas-cowboy.jpeg" in preview.text
    assert 'data-submit-stage="preview"' in preview.text
    assert 'data-submit-stage="result"' in preview.text
    assert "No trace yet" not in preview.text
    assert "HX-Push-Url" not in preview.headers
    assert 'title="Prepare first"' not in preview.text

    with SAMPLE.open("rb") as image:
        generated = client.post(
            "/generate",
            data=_preview_payload(),
            files={"artwork": (SAMPLE.name, image, "image/jpeg")},
        )

    assert generated.status_code == 200
    assert 'data-render-stage="result"' in generated.text
    assert 'data-mesh-url="data:model/stl;base64,' in generated.text
    assert 'download="xmas-cowboy-stamp.stl"' in generated.text
    assert 'data-auto-scroll-result="true"' in generated.text
    assert "Download STL" in generated.text
    assert "HX-Push-Url" not in generated.headers


def test_webapp_preview_error_preserves_upload_thumbnail() -> None:
    client = TestClient(app)

    with SAMPLE.open("rb") as image:
        response = client.post(
            "/preview",
            data={
                "width": "",
                "height": "",
                "border": "0.5",
                "base": "4",
                "relief": "1",
                "threshold": "190",
                "resolution": "0",
                "simplify": "0.08",
                "raised_border": "on",
                "invert": "",
            },
            files={"artwork": (SAMPLE.name, image, "image/jpeg")},
        )

    assert response.status_code == 200
    assert "Please enter at least one max dimension: width, height, or both." in response.text
    assert 'data-render-stage="preview"' in response.text
    assert 'data-artwork-url="data:image/jpeg;base64,' in response.text
    assert "Draft saved" in response.text
    assert "No trace yet" in response.text
    assert 'title="Prepare first"' in response.text
    assert 'data-submit-stage="result"' in response.text


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
    assert 'data-render-stage="preview"' in preview.text
    assert 'data-artwork-url="data:image/svg+xml;base64,' in preview.text
    assert "Generate STL" in preview.text
    assert "No trace yet" not in preview.text


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


def test_webapp_serves_sample_artwork() -> None:
    client = TestClient(app)

    sample = client.get(str(app.url_path_for("sample_artwork")))
    assert sample.status_code == 200
    assert sample.headers["content-type"].startswith("image/jpeg")
    assert sample.content == SAMPLE.read_bytes()

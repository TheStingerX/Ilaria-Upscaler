"""Ensure the requested browser branding and language survive updates."""
from fastapi.testclient import TestClient
import app as service

client = TestClient(service.app)


def test_chrome_tab_title_and_favicon():
    response = client.get("/")
    assert response.status_code == 200
    assert '<title>Ilaria Upscaler 2.0</title>' in response.text
    assert 'href="/static/favicon.png?v=2"' in response.text
    icon = client.get("/static/favicon.png")
    assert icon.status_code == 200
    assert icon.headers["content-type"].startswith("image/png")
    assert len(icon.content) > 1000


def test_api_errors_english():
    response = client.get("/api/jobs/not-a-real-job")
    assert response.status_code == 404
    assert response.json()["detail"] == "Job not found or expired."

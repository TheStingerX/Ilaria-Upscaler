"""FastAPI smoke tests with inference mocked (run without model weights)."""
from io import BytesIO
from PIL import Image
from fastapi.testclient import TestClient
import app as service

client = TestClient(service.app)


def _png() -> bytes:
    image = Image.new("RGB", (20, 15), (120, 70, 180))
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def test_frontend_and_status():
    assert client.get("/").status_code == 200
    body = client.get("/").text
    assert 'data-mode="fast"' in body and 'data-mode="quality"' in body and 'data-mode="ultra"' in body
    status = client.get("/api/status")
    assert status.status_code == 200
    assert set(status.json()["modes"]) == {"fast", "quality", "ultra"}


def test_reject_wrong_model():
    response = client.post("/api/jobs", files={"image": ("image.png", _png(), "image/png")},
                           data={"mode": "fast", "model": "seedvr2-3b", "scale": 2, "tile": 128})
    assert response.status_code == 400


def test_reject_malformed_upload():
    response = client.post("/api/jobs", files={"image": ("input.png", b"not a picture", "image/png")},
                           data={"mode": "fast", "model": "realesrgan-x4", "scale": 2, "tile": 128})
    assert response.status_code == 400


def test_job_processing_with_stub(monkeypatch):
    def fake_process(input_file, output_file, mode, model, scale, tile, progress):
        progress(35, "Testing")
        Image.open(input_file).resize((40, 30)).save(output_file, "PNG")
        progress(100, "Completed")
        return {"width": 40, "height": 30, "format": "PNG"}
    monkeypatch.setattr(service, "process", fake_process)
    response = client.post("/api/jobs", files={"image": ("input.png", _png(), "image/png")},
                           data={"mode": "fast", "model": "realesrgan-x4", "scale": 2, "tile": 128})
    assert response.status_code == 202
    job_id = response.json()["id"]
    future = service.EXECUTOR.submit(lambda: None)
    future.result(timeout=5)
    job = client.get(f"/api/jobs/{job_id}")
    assert job.json()["state"] == "done"
    result = client.get(f"/api/jobs/{job_id}/result")
    assert result.status_code == 200
    with Image.open(BytesIO(result.content)) as picture:
        assert picture.size == (40, 30)

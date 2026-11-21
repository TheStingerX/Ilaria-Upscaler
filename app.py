"""Ilaria Upscaler 2.0 — FastAPI server and isolated processing jobs."""
from __future__ import annotations

import io
import os
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from engines import model_catalog, process

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 18 * 1024 * 1024
MAX_INPUT_PIXELS = 12_000_000
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="upscaler")
# Each job uses its own private temp directory. No global image or output filename.
app = FastAPI(title="Ilaria Upscaler 2.0", version="2.0")
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")


@app.get("/")
def home():
    return FileResponse(ROOT / "web/index.html")


@app.get("/api/status")
def status():
    return model_catalog()


def _clean_expired() -> None:
    now = time.time()
    with LOCK:
        expired = [key for key, value in JOBS.items() if value.get("finished_at") and now - value["finished_at"] > 3600]
        for key in expired:
            job = JOBS.pop(key)
            shutil.rmtree(job["directory"], ignore_errors=True)


def _run(job_id: str, input_file: Path, mode: str, model: str, scale: float, tile: int) -> None:
    def update(percentage: int, message: str) -> None:
        with LOCK:
            JOBS[job_id].update(progress=percentage, message=message)
    with LOCK:
        JOBS[job_id].update(state="processing", message="Starting image processing")
    try:
        output = input_file.parent / "upscaled.png"
        metadata = process(input_file, output, mode, model, scale, tile, update)
        with LOCK:
            JOBS[job_id].update(state="done", progress=100, message="Image ready", metadata=metadata, finished_at=time.time())
    except Exception as exc:
        # Do not expose absolute file paths or internal tracebacks to the public API.
        error = str(exc).replace(str(ROOT), "[app]")[:600]
        with LOCK:
            JOBS[job_id].update(state="error", message=error, finished_at=time.time())


@app.post("/api/jobs", status_code=202)
async def submit(image: UploadFile = File(...), mode: str = Form("fast"), model: str = Form("realesrgan-x4"),
                 scale: float = Form(4), tile: int = Form(192)):
    _clean_expired()
    with LOCK:
        pending = sum(1 for job in JOBS.values() if job["state"] in ("queued", "processing"))
    if pending >= 12:
        raise HTTPException(429, "Server busy. Please try again when the queue has capacity.")
    catalog = model_catalog()["modes"]
    if mode not in catalog or not catalog[mode]["available"]:
        raise HTTPException(400, "This engine is not available on this server.")
    if model not in [item["id"] for item in catalog[mode]["models"]]:
        raise HTTPException(400, "The selected model is not valid for this engine.")
    if not 1 <= scale <= 4 or tile not in (128, 192, 256, 384):
        raise HTTPException(400, "Invalid processing parameters.")
    payload = await image.read(MAX_UPLOAD_BYTES + 1)
    if not payload or len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image missing or larger than 18 MB.")
    try:
        Image.MAX_IMAGE_PIXELS = MAX_INPUT_PIXELS
        with Image.open(io.BytesIO(payload)) as img:
            if img.format not in ("PNG", "JPEG", "WEBP"):
                raise ValueError("Unsupported image format.")
            if img.width * img.height > MAX_INPUT_PIXELS:
                raise ValueError("Image is too large (maximum 12 MP).")
            if max(round(img.width * scale), round(img.height * scale)) > 10000 or round(img.width * scale) * round(img.height * scale) > 75_000_000:
                raise ValueError("Output exceeds size limits. Reduce the upscale factor.")
            img.verify()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(400, f"Invalid image: {exc}") from exc
    directory = Path(tempfile.mkdtemp(prefix="ilaria-upscaler-"))
    input_file = directory / "input.png"
    try:
        # Transcode to a known input format; no user-controlled filenames are used.
        with Image.open(io.BytesIO(payload)) as img:
            from PIL import ImageOps
            img = ImageOps.exif_transpose(img)
            img.save(input_file, "PNG")
        job_id = uuid.uuid4().hex
        with LOCK:
            JOBS[job_id] = {"state": "queued", "progress": 0, "message": "Queued", "directory": directory, "created_at": time.time()}
        EXECUTOR.submit(_run, job_id, input_file, mode, model, scale, tile)
        return {"id": job_id, "state": "queued"}
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    _clean_expired()
    with LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found or expired.")
        return {key: value for key, value in job.items() if key not in ("directory", "created_at", "finished_at")}


@app.get("/api/jobs/{job_id}/result")
def get_result(job_id: str):
    _clean_expired()
    with LOCK:
        job = JOBS.get(job_id)
        if job is None or job["state"] != "done":
            raise HTTPException(404, "Result is not ready yet.")
        path = job["directory"] / "upscaled.png"
    return FileResponse(path, media_type="image/png", filename="ilaria-upscaled.png")


def _open_local_browser_when_ready(port: int) -> None:
    """Open one browser tab after the local server starts accepting requests."""
    import urllib.request
    import webbrowser

    url = f"http://127.0.0.1:{port}/"
    for _ in range(80):
        try:
            with urllib.request.urlopen(url, timeout=0.7):
                pass
            webbrowser.open(url)
            return
        except (OSError, TimeoutError):
            time.sleep(0.5)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    if os.getenv("ILARIA_OPEN_BROWSER") == "1":
        threading.Thread(target=_open_local_browser_when_ready, args=(port,), daemon=True).start()
    # Polling is frequent: keep access logs out of the user-facing console.
    # Exceptions and other serious server errors remain visible.
    uvicorn.run("app:app", host="0.0.0.0", port=port, access_log=False, log_level="warning")

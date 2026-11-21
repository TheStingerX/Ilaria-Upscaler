"""The three inference lanes. No fake AI processing or silent fallbacks."""
from __future__ import annotations

import math
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Callable

from PIL import Image
from bootstrap import nvidia_gpu

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "models"
QUALITY = WEIGHTS / "quality"
FAST_WEIGHTS = {
    "realesrgan-x4": ("RealESRGAN_x4plus.pth", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"),
    "realesrgan-x2": ("RealESRGAN_x2plus.pth", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"),
    "anime-x4": ("RealESRGAN_x4plus_anime_6B.pth", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"),
}
QUALITY_WEIGHTS = {
    "realesrnet-x4": ("RealESRNet_x4plus.pth", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"),
}
ALLOWED_EXTENSIONS = {".pth", ".pt", ".ckpt", ".safetensors"}


def _torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed. Install the project requirements and the correct PyTorch build for your hardware.") from exc


def hardware() -> dict:
    gpu = nvidia_gpu()
    try:
        torch = _torch()
        cuda = bool(torch.cuda.is_available())
        return {
            "cuda": cuda,
            "device": torch.cuda.get_device_name(0) if cuda else "CPU",
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if cuda else None,
            "torch_cuda": str(torch.version.cuda) if torch.version.cuda else None,
            "nvidia_detected": bool(gpu),
            "nvidia_name": gpu["name"] if gpu else None,
        }
    except Exception:
        return {"cuda": False, "device": "Not detected", "vram_gb": None,
                "torch_cuda": None, "nvidia_detected": bool(gpu),
                "nvidia_name": gpu["name"] if gpu else None}


def cli_path() -> Path | None:
    raw = os.environ.get("SEEDVR2_CLI", "").strip()
    if raw:
        path = Path(raw).expanduser().resolve()
        return path if path.is_file() and path.name == "inference_cli.py" else None
    bundled = ROOT / "vendor" / "seedvr2" / "inference_cli.py"
    return bundled if bundled.is_file() else None


def ultra_python() -> Path | None:
    raw = os.environ.get("SEEDVR2_PYTHON", "").strip()
    if raw:
        path = Path(raw).expanduser().resolve()
        return path if path.is_file() else None
    venv = ROOT / ".venv-ultra"
    bundled = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return bundled if bundled.is_file() else None


def ultra_models() -> list[dict]:
    folder = WEIGHTS / "ultra"
    models = [
        ("seedvr2-3b-q4", "SeedVR2 3B · Q4 · Lower VRAM", "seedvr2_ema_3b-Q4_K_M.gguf", 1_995_344_224),
        ("seedvr2-3b-fp8", "SeedVR2 3B · FP8 · Higher quality", "seedvr2_ema_3b_fp8_e4m3fn.safetensors", 3_391_544_696),
    ]
    result = []
    for key, label, filename, size in models:
        model = folder / filename
        if model.is_file() and model.stat().st_size == size:
            result.append({"id": key, "label": label, "filename": filename})
    return result


def model_catalog() -> dict:
    QUALITY.mkdir(parents=True, exist_ok=True)
    custom = []
    for path in sorted(QUALITY.iterdir()):
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS and path.stat().st_size:
            custom.append({"id": f"local:{path.name}", "label": f"{path.stem} · custom/HAT", "available": True})
    hw = hardware()
    cli = cli_path()
    python = ultra_python()
    weights = ultra_models()
    vae = WEIGHTS / "ultra" / "ema_vae_fp16.safetensors"
    vae_ok = vae.is_file() and vae.stat().st_size == 501_324_814
    ultra_ready = bool(hw["cuda"] and cli and python and weights and vae_ok)
    if not hw["cuda"]:
        ultra_reason = ("NVIDIA GPU detected, but CUDA is unavailable in PyTorch. Run Run.bat again and check your drivers."
                        if hw["nvidia_detected"] else "A CUDA-compatible NVIDIA GPU is required.")
    elif not cli or not python:
        ultra_reason = "SeedVR2 is not installed. Run Run.bat again to install it automatically."
    elif not weights or not vae_ok:
        ultra_reason = "SeedVR2 model files are missing. Run Run.bat again to finish setup."
    else:
        ultra_reason = "SeedVR2 and model files are installed. Full compatibility will be checked during the first run."
    return {
        "hardware": hw,
        "modes": {
            "fast": {"available": True, "reason": "Real-ESRGAN model files download on first use.", "models": [
                {"id": "realesrgan-x4", "label": "Real-ESRGAN · 4×"},
                {"id": "realesrgan-x2", "label": "Real-ESRGAN · 2×"},
                {"id": "anime-x4", "label": "Real-ESRGAN Anime · 4×"},
            ]},
            "quality": {"available": True, "reason": "Real-ESRNet downloads on first use. Add compatible HAT models to models/quality.", "models": [
                {"id": "realesrnet-x4", "label": "Real-ESRNet · 4×"}, *custom
            ]},
            "ultra": {"available": ultra_ready, "reason": ultra_reason,
                      "models": [{"id": x["id"], "label": x["label"]} for x in weights] or
                                [{"id": "seedvr2-3b-q4", "label": "SeedVR2 · 3B (installing)"}]}
        }
    }


def _download(name: str, url: str) -> Path:
    WEIGHTS.mkdir(exist_ok=True)
    path = WEIGHTS / name
    if path.is_file() and path.stat().st_size > 0:
        return path
    temp = path.with_suffix(".download")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Ilaria-Upscaler/2.0"})
        with urllib.request.urlopen(request, timeout=120) as response, temp.open("wb") as out:
            while True:
                data = response.read(1024 * 1024)
                if not data:
                    break
                out.write(data)
        if temp.stat().st_size < 100_000:
            raise RuntimeError("Model download is incomplete.")
        temp.replace(path)
        return path
    finally:
        temp.unlink(missing_ok=True)


def model_path(mode: str, model: str) -> Path:
    if mode == "fast" and model in FAST_WEIGHTS:
        return _download(*FAST_WEIGHTS[model])
    if mode == "quality" and model in QUALITY_WEIGHTS:
        return _download(*QUALITY_WEIGHTS[model])
    if mode == "quality" and model.startswith("local:"):
        name = model[6:]
        if Path(name).name != name or Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
            raise ValueError("Invalid model filename.")
        path = (QUALITY / name).resolve()
        if path.parent != QUALITY.resolve() or not path.is_file():
            raise ValueError("Local model was not found.")
        return path
    raise ValueError("This model is not available for the selected engine.")


def _upscale_spandrel(img: Image.Image, model_file: Path, scale: float, tile: int, progress: Callable[[int, str], None]) -> Image.Image:
    import numpy as np
    import torch
    from spandrel import ImageModelDescriptor, ModelLoader

    # Spandrel supports RRDB, HAT and many other super-resolution architectures.
    # Some additional model formats are only available with the optional package.
    try:
        import spandrel_extra_arches
        spandrel_extra_arches.install(ignore_duplicates=True)
    except ImportError:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    progress(17, "Loading model")
    descriptor = ModelLoader().load_from_file(str(model_file))
    if not isinstance(descriptor, ImageModelDescriptor):
        raise RuntimeError("This file does not contain an image-to-image model supported by Spandrel.")
    descriptor = descriptor.to(device).eval()
    native = int(descriptor.scale)
    if native < 1:
        raise RuntimeError("Invalid native model scale.")
    original = img.convert("RGBA")
    alpha = original.getchannel("A") if img.mode in ("RGBA", "LA", "P") else None
    rgb = np.asarray(original.convert("RGB"), dtype=np.float32) / 255.0
    h, w = rgb.shape[:2]
    source = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0)
    del rgb
    # Whole-image processing can run out of GPU memory; tiles bound peak memory.
    # For HAT-like window transformers, pad each tile to a multiple of 32.
    tile = int(tile)
    overlap = min(24, max(8, tile // 8))
    grid_x, grid_y = math.ceil(w / tile), math.ceil(h / tile)
    dest = np.empty((h * native, w * native, 3), dtype=np.uint8)
    total = grid_x * grid_y
    try:
        with torch.inference_mode():
            current = 0
            for y in range(0, h, tile):
                for x in range(0, w, tile):
                    x0, y0 = max(0, x - overlap), max(0, y - overlap)
                    x1, y1 = min(w, x + tile + overlap), min(h, y + tile + overlap)
                    patch = source[:, :, y0:y1, x0:x1].to(device)
                    ph, pw = patch.shape[-2:]
                    pad_h = (32 - ph % 32) % 32
                    pad_w = (32 - pw % 32) % 32
                    if pad_h or pad_w:
                        patch = torch.nn.functional.pad(patch, (0, pad_w, 0, pad_h), mode="replicate")
                    output = descriptor(patch).float().clamp_(0, 1)
                    output = output[:, :, :ph * native, :pw * native]
                    offset_x, offset_y = (x - x0) * native, (y - y0) * native
                    valid_w, valid_h = min(tile, w - x) * native, min(tile, h - y) * native
                    crop = output[0, :, offset_y:offset_y + valid_h, offset_x:offset_x + valid_w]
                    dest[y * native:y * native + valid_h, x * native:x * native + valid_w] = (
                        crop.permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
                    )
                    del patch, output, crop
                    current += 1
                    progress(22 + int(65 * current / total), f"Processing tile {current}/{total}")
    finally:
        del descriptor, source
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = Image.fromarray(dest, "RGB")
    target = (round(w * scale), round(h * scale))
    if result.size != target:
        result = result.resize(target, Image.Resampling.LANCZOS)
    if alpha:
        alpha = alpha.resize(target, Image.Resampling.LANCZOS)
        result.putalpha(alpha)
    return result


def _upscale_seedvr2(input_file: Path, output_file: Path, img: Image.Image, scale: float,
                     model: str, progress: Callable[[int, str], None]) -> Image.Image:
    cli = cli_path()
    interpreter = ultra_python()
    if not cli or not interpreter or not hardware()["cuda"]:
        raise RuntimeError("SeedVR2 is unavailable. Check CUDA and run Run.bat again.")
    choices = {item["id"]: item["filename"] for item in ultra_models()}
    if model not in choices:
        raise RuntimeError("The selected SeedVR2 model is not installed.")
    width, height = img.size
    short_side = min(4096, max(64, round(min(width, height) * scale)))
    output_dir = output_file.parent / "seedvr2"
    output_dir.mkdir(exist_ok=True)
    command = [str(interpreter), str(cli), str(input_file),
               "--output", str(output_dir),
               "--model_dir", str(WEIGHTS / "ultra"),
               "--dit_model", choices[model],
               "--resolution", str(short_side), "--batch_size", "1",
               "--vae_encode_tiled", "--vae_decode_tiled", "--color_correction", "lab"]
    # BlockSwap on small VRAM: more stable but slower.
    vram = hardware()["vram_gb"] or 0
    if vram < 12:
        command.extend(["--dit_offload_device", "cpu", "--vae_offload_device", "cpu",
                        "--blocks_to_swap", "32", "--swap_io_components"])
    progress(25, "SeedVR2 is processing your image")
    try:
        subprocess.run(command, cwd=str(cli.parent), check=True, timeout=3600,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("SeedVR2 failed: " + (exc.stderr or "")[-1200:]) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("SeedVR2 exceeded the 60-minute processing timeout.") from exc
    results = [p for p in output_dir.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp") and p.is_file()]
    if not results:
        raise RuntimeError("SeedVR2 did not produce an image. Check the CLI logs.")
    progress(91, "Preparing output")
    with Image.open(max(results, key=lambda p: p.stat().st_mtime)) as result:
        return result.convert("RGB").copy()


def process(input_file: Path, output_file: Path, mode: str, model: str, scale: float, tile: int, progress: Callable[[int, str], None]) -> dict:
    with Image.open(input_file) as source:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(source).copy()
    progress(9, "Image verified")
    if mode == "ultra":
        if model not in {x["id"] for x in ultra_models()}:
            raise ValueError("Unrecognized or missing Ultra model.")
        result = _upscale_seedvr2(input_file, output_file, img, scale, model, progress)
    else:
        path = model_path(mode, model)
        result = _upscale_spandrel(img, path, scale, tile, progress)
    if max(result.size) > 10000 or result.width * result.height > 75_000_000:
        raise RuntimeError("Output exceeds size limits. Reduce the upscale factor.")
    result.save(output_file, "PNG", optimize=True)
    progress(100, "Completed")
    return {"width": result.width, "height": result.height, "format": "PNG"}

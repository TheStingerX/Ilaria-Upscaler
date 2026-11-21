"""Logic tests: no third-party models or CUDA needed."""
import subprocess
from types import SimpleNamespace
from pathlib import Path
from PIL import Image
import bootstrap
import engines


def test_model_selection_by_vram():
    low = bootstrap.pick_model({"vram_mb": 6144})
    high = bootstrap.pick_model({"vram_mb": 16384})
    assert low[0] == "seedvr2_ema_3b-Q4_K_M.gguf"
    assert high[0] == "seedvr2_ema_3b_fp8_e4m3fn.safetensors"


def test_cpu_badge_can_identify_nvidia_with_cpu_only_torch(monkeypatch):
    monkeypatch.setattr(engines, "nvidia_gpu", lambda: {"name": "Example NVIDIA", "vram_mb": 8192})
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        version=SimpleNamespace(cuda=None),
    )
    monkeypatch.setattr(engines, "_torch", lambda: fake_torch)
    hw = engines.hardware()
    assert hw["cuda"] is False and hw["nvidia_detected"] is True
    assert hw["nvidia_name"] == "Example NVIDIA"


def test_cli_subprocess_uses_local_model_and_vram_optimizations(monkeypatch, tmp_path):
    cli = tmp_path / "vendor" / "seedvr2" / "inference_cli.py"
    cli.parent.mkdir(parents=True)
    cli.write_text("# placeholder")
    python = tmp_path / ".venv-ultra" / "python.exe"
    python.parent.mkdir()
    python.write_text("placeholder")
    monkeypatch.setattr(engines, "WEIGHTS", tmp_path / "models")
    monkeypatch.setattr(engines, "cli_path", lambda: cli)
    monkeypatch.setattr(engines, "ultra_python", lambda: python)
    monkeypatch.setattr(engines, "hardware", lambda: {"cuda": True, "vram_gb": 6})
    monkeypatch.setattr(engines, "ultra_models", lambda: [
        {"id": "seedvr2-3b-q4", "filename": "seedvr2_ema_3b-Q4_K_M.gguf"}])

    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    Image.new("RGB", (20, 16)).save(input_path)
    seen = []

    def simulate(args, **kwargs):
        seen.extend(args)
        dest = tmp_path / "seedvr2" / "generated.png"
        Image.new("RGB", (40, 32)).save(dest)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(engines.subprocess, "run", simulate)
    picture = engines._upscale_seedvr2(input_path, output_path, Image.open(input_path), 2,
                                       "seedvr2-3b-q4", lambda pct, message: None)
    assert picture.size == (40, 32)
    assert "--dit_model" in seen and "seedvr2_ema_3b-Q4_K_M.gguf" in seen
    assert "--model_dir" in seen and "--blocks_to_swap" in seen
    assert "--vae_encode_tiled" in seen and "--vae_decode_tiled" in seen

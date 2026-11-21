"""Windows-first, idempotent setup for Ilaria Upscaler 2.0.

Default behavior: install CPU/GPU Fast/Quality dependencies and, when a CUDA
capable NVIDIA GPU is available, install SeedVR2 + its matching checkpoints.

No dependency is downloaded at import time. The installer is called explicitly
by Run.bat, with --skip-ultra available for troubleshooting.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
MAIN_VENV = ROOT / ".venv"
ULTRA_VENV = ROOT / ".venv-ultra"
CLI_REPO = ROOT / "vendor" / "seedvr2"
WEIGHTS = ROOT / "models" / "ultra"
REPO_ZIP = "https://codeload.github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/zip/refs/heads/main"
LOG = ROOT / "logs" / "setup.log"


def say(msg: str) -> None:
    print(msg, flush=True)


def command(args: list[str], *, cwd: Path | None = None, allow_failure: bool = False) -> bool:
    """Run dependencies quietly while preserving complete setup logs on disk."""
    LOG.parent.mkdir(parents=True, exist_ok=True)
    # Keep the setup log small and retain the previous one for troubleshooting.
    if LOG.is_file() and LOG.stat().st_size > 5_000_000:
        LOG.replace(LOG.with_name("setup.previous.log"))
    with LOG.open("ab") as log:
        log.write(("\n$ " + " ".join(str(arg) for arg in args) + "\n").encode("utf-8", errors="replace"))
        log.flush()
        result = subprocess.run(args, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=False)
        log.write((f"\nExit code: {result.returncode}\n").encode("utf-8"))
    if result.returncode != 0 and not allow_failure:
        raise RuntimeError(f"Command failed (exit code {result.returncode}): {Path(str(args[0])).name}. Details: logs/setup.log")
    return result.returncode == 0


def env_python(env: Path) -> Path:
    return env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def pip(python: Path, *args: str) -> None:
    command([str(python), "-m", "pip", *args])


def make_venv(dest: Path, python_executable: str | Path) -> Path:
    interpreter = env_python(dest)
    if not interpreter.is_file():
        say(f"Creating Python environment: {dest.name}")
        command([str(python_executable), "-m", "venv", str(dest)])
    return interpreter


def nvidia_gpu() -> dict | None:
    """Hardware detection independent of any existing PyTorch wheel."""
    executables = ["nvidia-smi"]
    if os.name == "nt":
        executables.append(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe")
    for exe in executables:
        try:
            response = subprocess.run(
                [exe, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=False, timeout=15,
            )
            if response.returncode:
                continue
            first = response.stdout.splitlines()[0].strip()
            items = [x.strip() for x in first.split(",")]
            if len(items) < 3:
                continue
            return {"name": items[0], "driver": items[1], "vram_mb": int(float(items[2]))}
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            continue
    return None


def torch_info(python: Path) -> dict:
    code = (
        "import json,torch;"
        "print(json.dumps({'version':torch.__version__, 'cuda_build':torch.version.cuda, "
        "'available':torch.cuda.is_available()}))"
    )
    try:
        result = subprocess.run([str(python), "-c", code], capture_output=True, text=True,
                                check=False, timeout=45)
        if result.returncode:
            return {"error": (result.stderr or result.stdout).strip()[-700:]}
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def cuda_index(gpu: dict) -> str:
    try:
        major = float(".".join(gpu["driver"].split(".")[:2]))
    except (ValueError, KeyError):
        major = 999
    # NVIDIA minimum compatible driver branches: CUDA 12.8 / 12.6 / 11.8.
    return "cu128" if major >= 570 else "cu126" if major >= 560 else "cu118"


def ensure_gpu_torch(python: Path, gpu: dict) -> bool:
    info = torch_info(python)
    if info.get("available"):
        say("CUDA acceleration is ready.")
        return True
    if info.get("cuda_build"):
        say("WARNING: PyTorch includes CUDA, but GPU acceleration is unavailable. Check your NVIDIA drivers.")
        return False
    index = cuda_index(gpu)
    say(f"Detected NVIDIA {gpu['name']} ({gpu['vram_mb']} MB VRAM, driver {gpu['driver']}.")
    say(f"Installing PyTorch with CUDA {index} (this may download several gigabytes)...")
    pip(python, "install", "--upgrade", "torch", "torchvision", "--index-url", f"https://download.pytorch.org/whl/{index}")
    info = torch_info(python)
    if not info.get("available"):
        say(f"WARNING: CUDA is still unavailable after setup: {info}")
        return False
    say("CUDA acceleration is ready.")
    return True


def install_main(gpu: dict | None) -> tuple[Path, bool]:
    say("\n[1/3] Checking Fast and Quality engines")
    python = make_venv(MAIN_VENV, sys.executable)
    has_cuda = False
    if gpu:
        try:
            has_cuda = ensure_gpu_torch(python, gpu)
        except RuntimeError as exc:
            say(f"WARNING: CUDA installation failed ({exc}).")
            say("Continuing with the available PyTorch installation.")
    pip(python, "install", "-r", str(ROOT / "requirements.txt"))
    # CPU-only devices can use the ordinary PyPI package pulled in via spandrel.
    code = "import fastapi,uvicorn,torch,spandrel; print('Core dependencies are available')"
    command([str(python), "-c", code])
    if gpu:
        has_cuda = bool(torch_info(python).get("available"))
    return python, has_cuda


def fetch_seedvr2() -> None:
    cli = CLI_REPO / "inference_cli.py"
    if cli.is_file():
        return
    say("Downloading SeedVR2 source code...")
    CLI_REPO.parent.mkdir(parents=True, exist_ok=True)
    # Git is optional; use GitHub's source archive when Git is not installed.
    if shutil.which("git"):
        if command(["git", "clone", "--depth", "1", "--branch", "main",
                    "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git", str(CLI_REPO)], allow_failure=True):
            if cli.is_file():
                return
    request = urllib.request.Request(REPO_ZIP, headers={"User-Agent": "Ilaria-Upscaler/2.1"})
    with urllib.request.urlopen(request, timeout=90) as response:
        data = response.read(80_000_001)
        if len(data) > 80_000_000:
            raise RuntimeError("SeedVR2 source archive exceeded the permitted size.")
    with tempfile.TemporaryDirectory(prefix="seedvr2-src-") as temp:
        base = Path(temp)
        path = base / "source.zip"
        path.write_bytes(data)
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            for name in names:
                entry = Path(name)
                if name.startswith("/") or ".." in entry.parts:
                    raise RuntimeError("GitHub source archive contains an unsafe file path.")
            z.extractall(base / "unpacked")
        extracted = base / "unpacked" / "ComfyUI-SeedVR2_VideoUpscaler-main"
        if not (extracted / "inference_cli.py").is_file():
            raise RuntimeError("SeedVR2 CLI was not found in the downloaded archive.")
        if CLI_REPO.exists():
            shutil.rmtree(CLI_REPO)
        shutil.move(str(extracted), str(CLI_REPO))


def pick_model(gpu: dict) -> tuple[str, str, str]:
    """Pick conservatively for low VRAM, not a claim of measured performance."""
    if gpu["vram_mb"] < 12000:
        return "seedvr2_ema_3b-Q4_K_M.gguf", "AInVFX/SeedVR2_comfyUI", "low"
    return "seedvr2_ema_3b_fp8_e4m3fn.safetensors", "numz/SeedVR2_comfyUI", "standard"


def ensure_ultra(gpu: dict, main_cuda: bool) -> bool:
    say("\n[2/3] Checking Ultra engine (SeedVR2)")
    if not main_cuda:
        say("Ultra is unavailable: CUDA support is not active in the main environment.")
        return False
    if shutil.disk_usage(ROOT).free < 7 * 1024**3:
        say("Ultra was not installed: at least 7 GB of free disk space is recommended.")
        return False
    fetch_seedvr2()
    # Separate environment prevents dependency conflicts with Fast and Quality.
    # Python 3.12 is preferred if installed; 3.11 is compatible with CLI requirements.
    executable = sys.executable
    if os.name == "nt":
        choice = subprocess.run(["py", "-3.12", "-c", "import sys;print(sys.executable)"],
                                capture_output=True, text=True, check=False)
        if choice.returncode == 0:
            executable = choice.stdout.strip()
    ultra_python = make_venv(ULTRA_VENV, executable)
    ready_file = ULTRA_VENV / ".ilaria-ultra-ready.json"
    model, repo, profile = pick_model(gpu)
    if not ready_file.is_file():
        say("Installing SeedVR2 dependencies in an isolated environment...")
        pip(ultra_python, "install", "--upgrade", "pip")
        if not ensure_gpu_torch(ultra_python, gpu):
            say("Ultra is unavailable: CUDA could not be enabled in its environment.")
            return False
        pip(ultra_python, "install", "-r", str(CLI_REPO / "requirements.txt"))
        pip(ultra_python, "install", "huggingface_hub")
        # Validate actual imports, not merely the presence of a script file.
        command([str(ultra_python), str(CLI_REPO / "inference_cli.py"), "--help"], cwd=CLI_REPO)
        ready_file.write_text(json.dumps({"engine": "SeedVR2", "python": str(ultra_python)}), encoding="utf-8")
    else:
        say("SeedVR2 dependencies already installed.")
    say("\n[3/3] Checking Ultra models")
    say(f"Profile: {profile}; model: {model} (large download if missing).")
    command([str(ultra_python), str(ROOT / "prefetch_ultra.py"), "--model", model,
             "--repo", repo, "--model-dir", str(WEIGHTS)])
    say("Ultra engine is ready.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ultra", action="store_true", help="Skip SeedVR2 (troubleshooting)")
    args = parser.parse_args()
    gpu = nvidia_gpu()
    if gpu:
        say(f"GPU: {gpu['name']} | VRAM: {gpu['vram_mb']} MB | driver: {gpu['driver']}")
    else:
        say("No NVIDIA GPU detected. Fast and Quality can run in CPU mode.")
    try:
        _, cuda_ready = install_main(gpu)
    except Exception as exc:
        say(f"ERROR: Fast/Quality setup failed: {exc}")
        return 1
    if gpu and not args.skip_ultra:
        try:
            ensure_ultra(gpu, cuda_ready)
        except Exception as exc:
            say(f"WARNING: Ultra setup is incomplete: {exc}")
            say("Fast and Quality remain available. Run Run.bat again to retry Ultra setup.")
    elif not gpu:
        say("Ultra requires an NVIDIA CUDA GPU. Ultra setup skipped.")
    else:
        say("Ultra setup skipped (--skip-ultra).")
    say("\nEnvironment checked. Starting the local server...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

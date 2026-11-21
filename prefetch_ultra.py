"""Download and verify SeedVR2's selected DiT and VAE model files."""
from __future__ import annotations
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download

SIZES = {
    "seedvr2_ema_3b-Q4_K_M.gguf": 1_995_344_224,
    "seedvr2_ema_3b_fp8_e4m3fn.safetensors": 3_391_544_696,
    "ema_vae_fp16.safetensors": 501_324_814,
}

def ensure(repo: str, file_name: str, folder: Path) -> None:
    target = folder / file_name
    expected = SIZES[file_name]
    if target.is_file() and target.stat().st_size == expected:
        print(f"OK: {file_name} ({expected / 1_000_000_000:.2f} GB)", flush=True)
        return
    if target.exists() and target.stat().st_size != expected:
        # Keep incomplete/suspicious local files for inspection, rather than overwriting.
        raise RuntimeError(f"Existing file has an unexpected size: {target}. Delete it manually to download it again.")
    print(f"Downloading {file_name} ({expected / 1_000_000_000:.2f} GB)...", flush=True)
    downloaded = hf_hub_download(repo_id=repo, filename=file_name, local_dir=folder)
    if Path(downloaded).stat().st_size != expected:
        raise RuntimeError(f"Incomplete download of {file_name}: unexpected file size.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(SIZES)[:2])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    ensure(args.repo, args.model, args.model_dir)
    ensure("numz/SeedVR2_comfyUI", "ema_vae_fp16.safetensors", args.model_dir)
    print("SeedVR2 model files verified.", flush=True)


if __name__ == "__main__":
    main()

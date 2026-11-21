<div align="center">

<img src="web/favicon.png" alt="Ilaria Upscaler icon" width="112" />

# Ilaria Upscaler 2.0

**Bring your images into focus. Three AI engines. One beautiful studio.**

A modern, self-hosted AI image upscaler with a custom browser interface, flexible enhancement models, and an optional diffusion-powered restoration engine.

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Windows](https://img.shields.io/badge/Windows-10%20%2F%2011-0078D4?logo=windows&logoColor=white)](#2-installation)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![CUDA](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-76B900?logo=nvidia&logoColor=white)](#3-compatible-gpus)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/TheStinger/Ilaria_Upscaler)

[**Original GitHub project**](https://github.com/TheStingerX/Ilaria-Upscaler) · [**Hugging Face Space**](https://huggingface.co/spaces/TheStinger/Ilaria_Upscaler) · [**Support on Ko-fi**](https://ko-fi.com/ilariaowo)

</div>

---

## Contents

1. [What is Ilaria Upscaler?](#1-what-is-ilaria-upscaler)
   - [The three engines](#the-three-engines)
   - [Features and workflow](#features-and-workflow)
   - [Image limits and output](#image-limits-and-output)
2. [Installation](#2-installation)
   - [Windows: quick installation](#windows-quick-installation)
   - [Running the application](#running-the-application)
   - [Manual and CPU-only setup](#manual-and-cpu-only-setup)
   - [Updating without losing models](#updating-without-losing-models)
   - [Troubleshooting](#troubleshooting)
   - [Docker and Hugging Face Spaces](#docker-and-hugging-face-spaces)
3. [Compatible GPUs](#3-compatible-gpus)
   - [Hardware compatibility](#hardware-compatibility)
   - [VRAM recommendations](#vram-recommendations)
   - [Check your GPU and CUDA](#check-your-gpu-and-cuda)

---

# 1. What is Ilaria Upscaler?

**Ilaria Upscaler 2.0** is an AI-powered image enhancement application designed to make upscaling accessible without requiring users to build a machine-learning workflow by hand. Upload an image, select the engine that fits your content, customize the output size, and download the result from your browser.

Version 2.0 is a new application rather than a reskin of the original Gradio tool. It uses **FastAPI** for local processing and a custom **HTML, CSS, and JavaScript** interface. All three engines are selectable in the same studio, with the available models and controls changing according to the chosen engine.

It is designed primarily for **Windows 10/11**, with local NVIDIA GPU acceleration where supported. Fast and Quality can also operate on the CPU. The included Dockerfile provides a CPU-oriented deployment path for those engines; running Ultra in Docker requires a separate CUDA-enabled setup.

### The three engines

| Engine | Technology | Best suited for | Speed / requirements |
|:--|:--|:--|:--|
| ⚡ **Fast** | **Real-ESRGAN** via Spandrel | Everyday photos, illustrations, screenshots, and anime | Usually the quickest option; CPU or CUDA |
| ✦ **Quality** | **Real-ESRNet**, plus compatible user-provided **HAT / ESRGAN** checkpoints via Spandrel | Model-specific restoration and greater control over the enhancement approach | Depends on the selected model; CPU or CUDA |
| ✨ **Ultra** | **SeedVR2 3B** diffusion upscaling through its standalone CLI | Complex or heavily degraded images that benefit from generative reconstruction | GPU-intensive; requires NVIDIA CUDA in this Windows build |

**Fast — Real-ESRGAN**

Fast is the default, general-purpose image enhancement engine. It includes three downloadable model options:

- **Real-ESRGAN x4** — general image upscaling.
- **Real-ESRGAN x2** — a native 2× model.
- **Real-ESRGAN Anime x4** — tailored for anime-style illustrations and similar artwork.

Weights download **on first use** and are reused on later launches. A tiled inference pipeline limits memory usage for large images. Start here when you want a reliable, comparatively lightweight option.

**Quality — Real-ESRNet and custom checkpoints**

Quality gives you a different restoration model and the ability to expand your local model library:

- **Real-ESRNet x4** downloads automatically on first use.
- Additional **Spandrel-compatible image-to-image checkpoints**, including supported HAT/ESRGAN models, can be placed in `models/quality/`.
- Local checkpoint formats recognized by the app: `.pth`, `.pt`, `.ckpt`, `.safetensors`.

A checkpoint must actually be compatible with Spandrel; a matching file extension alone does not guarantee support. **HAT weights are not bundled or automatically downloaded.** Quality is about *model choice*, not a promise that every image will look better than Fast.

**Ultra — SeedVR2 diffusion restoration**

Ultra integrates the standalone CLI from [ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler). Its generative approach can recover convincing visual structures from degraded sources, but may also **invent fine details or change lettering, logos, and faces**. For clean anime artwork or posters containing readable text, compare it against Fast or Quality before choosing a result.

The Windows launcher attempts to install Ultra automatically on a compatible NVIDIA CUDA system. It uses a separate Python environment to reduce conflicts with the other engines and chooses one of these **SeedVR2 3B** variants based on detected VRAM:

| Automatically selected model | GPU threshold used by installer | Model download (approx.) |
|:--|:--|--:|
| **Q4_K_M (GGUF)** — lower-VRAM profile | Below **12,000 MB** reported VRAM | 2.0 GB + 0.5 GB VAE |
| **FP8 (safetensors)** — standard profile | **12,000 MB or more** reported VRAM | 3.4 GB + 0.5 GB VAE |

The selected model and shared VAE are downloaded during setup, not packed into this repository. The low-VRAM path also requests CPU offloading, BlockSwap, and VAE tiling; these can trade speed for lower GPU memory use. Neither profile is guaranteed to fit every image or every GPU. The Ultra integration currently processes **images, not videos**, even though upstream SeedVR2 has additional video capabilities.

### Features and workflow

- **Original web interface:** three engine cards, a consistent design, and model-specific controls.
- **Simple image import:** drag and drop or select PNG, JPG/JPEG, or WebP files.
- **Adjustable output scale:** 1×–4× in 0.5× increments; Fast/Quality use model-native upscaling and resize to your selected final dimensions when necessary.
- **Memory controls:** select Fast/Quality tile sizes of 128, 192, 256, or 384 pixels. Ultra manages tiling through its own VAE settings instead.
- **Before/after preview:** compare the uploaded image with the processed output using the comparison slider.
- **Status and download:** see job status and export the finished image as PNG.
- **Hardware awareness:** identifies CUDA availability and distinguishes an undetected GPU from an NVIDIA GPU whose CUDA installation is not working.
- **Local processing:** the server runs on your machine and opens in your browser at `http://127.0.0.1:7860/`.
- **Cleaner startup:** English console messages, suppressed repetitive HTTP polling, and installation details saved in `logs/setup.log`.
- **Isolated jobs:** each job uses its own temporary directory; jobs are processed one at a time in the current server implementation.

**Typical workflow**

1. Launch `Run.bat` and let the setup finish.
2. Open the studio at `http://127.0.0.1:7860/` (the launcher normally opens it for you).
3. Choose **Fast**, **Quality**, or **Ultra**.
4. Upload an image and select an available model.
5. Choose an output scale; adjust tile size if using Fast or Quality.
6. Click **Upscale image** and follow the processing status.
7. Compare the result with the original and click **Download PNG**.

> [!NOTE]
> While SeedVR2 is running, the app may remain at **25%** for an extended period. That percentage represents the external CLI processing stage, **not** an exact measurement of SeedVR2's internal progress. High GPU activity and VRAM usage can be normal during inference; they do not prove that every job will complete.

### Image limits and output

| Parameter | Current limit |
|:--|:--|
| Accepted input formats | PNG, JPEG, WebP |
| Maximum upload size | **18 MB** |
| Maximum input resolution | **12 megapixels** |
| Upscale setting | **1×–4×**, in 0.5× steps |
| Maximum output dimension | **10,000 pixels** on either side |
| Maximum output area | **75 megapixels** |
| Download format | **PNG** |

For example, a 2048 × 2048 image at 2× targets approximately **4096 × 4096 pixels**. The maximum dimension and megapixel limits both apply. An output that exceeds them is rejected; the current build does not automatically reduce your chosen scale.

As with any AI upscaler, a larger image is **not** guaranteed to contain more faithful information. Results depend on the source, model, scale, and hardware.

---

# 2. Installation

## Windows: quick installation

### Prerequisites

- **Windows 10 or Windows 11, 64-bit** (Windows 11 is the primary target).
- **Python 3.11 (64-bit)**, installed and available through the Windows `py` launcher.
- An internet connection for Python packages, AI models, and (when supported) the SeedVR2 source code.
- Free disk space: **at least 7 GB** is checked before installing Ultra; significantly more is recommended for PyTorch, downloaded model weights, and package caches.
- **NVIDIA drivers** if you want CUDA acceleration or Ultra; a dedicated NVIDIA GPU is **not required** for CPU-only Fast/Quality.
- Git is optional: the installer can download the SeedVR2 source archive without it.

Download Python 3.11 from [python.org](https://www.python.org/downloads/) or install it from PowerShell:

```powershell
winget install --id Python.Python.3.11 --exact --source winget
```

Close and reopen PowerShell, then verify:

```powershell
py -3.11 --version
```

The output should begin with `Python 3.11`. Installing 3.11 **does not require uninstalling another Python version**.

### Download and extract

1. Download the **full Ilaria Upscaler 2.0 project archive** from its release/distribution page.
2. Extract the archive to a location you can write to, for example:

   ```text
   C:\Ilaria-Upscaler-2.0\
   ```

3. Open the extracted `Ilaria-Upscaler-2.0` folder. Make sure `Run.bat`, `app.py`, `bootstrap.py`, and `requirements.txt` are visible together.

### Running the application

**Double-click `Run.bat`.** There is no EXE to install.

On the first launch, `Run.bat` runs `bootstrap.py` and checks the necessary environments:

1. Creates or reuses **`.venv/`** for the application.
2. Detects NVIDIA hardware with `nvidia-smi` and attempts to install a CUDA-enabled PyTorch wheel when appropriate.
3. Installs the packages used by **Fast** and **Quality**.
4. If CUDA is ready, downloads the **SeedVR2** standalone source into `vendor/seedvr2/`.
5. Creates **`.venv-ultra/`** for SeedVR2 dependencies; this environment may use Python 3.12 if you already have it, otherwise Python 3.11.
6. Downloads the chosen **3B SeedVR2 checkpoint** and the shared VAE to `models/ultra/`.
7. Starts the server and opens `http://127.0.0.1:7860/` in your default browser.

Installation runs **automatically on launch**, and completed steps are reused. Large PyTorch and model downloads may make the first launch substantially slower than later launches. If optional Ultra setup fails, the launcher should still start Fast and Quality; consult the logs to see why Ultra is unavailable.

The console is intentionally simple:

```text
  ILARIA UPSCALER 2.0
  --------------------------------------------------
  Local AI Image Studio  |  Fast  /  Quality  /  Ultra

  Checking your environment...
  Setup details: logs\setup.log

  [1/3] Checking Fast and Quality engines
  [2/3] Checking Ultra engine (SeedVR2)
  [3/3] Checking Ultra models

  --------------------------------------------------
  Studio ready: http://127.0.0.1:7860/
```

> [!IMPORTANT]
> The displayed *Studio ready* line means the launcher is starting the web server; the browser opens once the server responds. You must keep the console window open while using the application. Use **Ctrl+C** to stop the server. The setup logs are stored in `logs/setup.log`, and server errors remain visible in the console.

**Prefer PowerShell?** Launch it from the project directory:

```powershell
cd "C:\Ilaria-Upscaler-2.0"
.\Run.bat
```

### Manual and CPU-only setup

If your NVIDIA installation is not ready, or you only want to use Fast and Quality, you can explicitly skip Ultra:

```powershell
cd "C:\Ilaria-Upscaler-2.0"
py -3.11 bootstrap.py --skip-ultra
.\.venv\Scripts\python.exe app.py
```

Alternatively, if you want to set up the primary environment yourself, choose a [PyTorch build appropriate for your hardware](https://pytorch.org/get-started/locally/) and then install the project requirements:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

The commands above install **CPU PyTorch** deliberately. They are useful for troubleshooting and do **not** enable Ultra. If you need CUDA, follow the official PyTorch selector instead of the CPU-specific install command.

### Updating without losing models

To update an existing installation:

1. Stop the running server with **Ctrl+C**.
2. Extract the update archive **into your existing project folder** and replace the matching source files.
3. **Do not remove** `.venv/`, `.venv-ultra/`, `vendor/`, `models/`, or `logs/` unless you are intentionally reinstalling or troubleshooting them.
4. Relaunch with **`Run.bat`**.

The important directories are:

```text
Ilaria-Upscaler-2.0/
├── Run.bat                 # Windows startup launcher
├── app.py                  # FastAPI application and jobs
├── engines.py              # Fast / Quality / Ultra inference
├── bootstrap.py            # Automatic setup and CUDA detection
├── prefetch_ultra.py       # SeedVR2 checkpoint downloads
├── requirements.txt        # Fast / Quality Python packages
├── web/                    # Browser UI, favicon and styles
│   ├── index.html
│   ├── app.js
│   ├── style.css
│   └── favicon.png
├── models/                 # Downloaded model files (created at runtime)
│   ├── quality/            # Your optional Spandrel/HAT checkpoints
│   └── ultra/              # SeedVR2 DiT and VAE weights
├── vendor/seedvr2/         # Downloaded upstream CLI (runtime)
├── .venv/                  # Main Python environment (runtime)
├── .venv-ultra/            # SeedVR2 Python environment (runtime)
└── logs/setup.log          # Installation log (runtime)
```

The runtime folders are not included in the small source ZIP. That is why a new installation requires downloads. To preserve your installation, replace source files rather than deleting the directory and starting from scratch.

### Troubleshooting

<details>
<summary><strong>Windows says Python 3.11 was not found</strong></summary>

Run `py -0p` to list installed interpreters. Install Python 3.11 from [python.org](https://www.python.org/downloads/) and reopen the terminal. Verify using `py -3.11 --version`; `Run.bat` explicitly uses 3.11.

</details>

<details>
<summary><strong>The app shows CPU MODE, but I have an NVIDIA GPU</strong></summary>

Run `nvidia-smi` and check the exact Python environment used by the application:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('Torch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

If `CUDA available: False`, check your NVIDIA driver and the PyTorch CUDA build. Re-running `Run.bat` attempts to fix a CPU-only PyTorch installation. A compatible GPU driver is still required.

</details>

<details>
<summary><strong>Ultra is unavailable or setup is incomplete</strong></summary>

Confirm that CUDA is available in `.venv/`, then inspect `logs/setup.log`. Ultra also needs the separate `.venv-ultra/` interpreter, `vendor/seedvr2/inference_cli.py`, the selected DiT weight, and `ema_vae_fp16.safetensors`. Running `Run.bat` again retries incomplete setup. An unsupported GPU, failed download, or Python dependency conflict can still prevent the model from loading.

</details>

<details>
<summary><strong>Ultra appears stuck at 25%</strong></summary>

This is usually a **stage indicator**, not a continuous percentage from the SeedVR2 CLI. Check GPU activity using `nvidia-smi` and the process status. SeedVR2 can use almost all available VRAM. A lengthy stage is not by itself proof of a crash; if the job fails, inspect the displayed error. The current Ultra subprocess has a **60-minute timeout** and does not stream detailed progress into the web UI.

</details>

<details>
<summary><strong>CUDA out of memory, or the image is too large</strong></summary>

Reduce the output scale (try **2×**) and/or input resolution. Fast and Quality have a tile selector: try **128 px** to lower peak memory requirements. Ultra already uses tiled VAE processing; the Q4 low-memory path is selected automatically only for GPUs below the 12,000 MB reported-VRAM threshold. You can also close GPU-intensive programs before processing. The application's output limits are 10,000 pixels per side and 75 MP in total.

</details>

<details>
<summary><strong>Fast/Quality reports “Duplicate architecture: SRFormer”</strong></summary>

The shipped engine integration uses `spandrel_extra_arches.install(ignore_duplicates=True)` to avoid a duplicate architecture registration. Make sure your `engines.py` is up to date. If the error persists, record the full traceback and the versions of `spandrel` and `spandrel-extra-arches` rather than repeatedly modifying the installed library.

</details>

<details>
<summary><strong>Installation or first launch failed</strong></summary>

Check `logs/setup.log` for the full command output. Common causes include interrupted downloads, insufficient disk space, incompatible Python/PyTorch builds, antivirus interference, and upstream dependency changes. The CLI is installed from its upstream repository at launch time, so a future upstream update may need additional integration work.

</details>

### Docker and Hugging Face Spaces

The repository includes a **CPU-only reference `Dockerfile`** using Python 3.11 and port `7860`. It installs CPU PyTorch and starts the FastAPI server. You can use it locally:

```bash
docker build -t ilaria-upscaler .
docker run --rm -p 7860:7860 ilaria-upscaler
```

Visit `http://127.0.0.1:7860/` after startup. The Dockerfile is suitable for **CPU Fast/Quality**, not GPU-enabled Ultra. Running Ultra in a container or a Hugging Face GPU Space requires a separate CUDA base image, suitable drivers/runtime, and an explicit SeedVR2 installation. A standard CPU Hugging Face Space does not provide the CUDA GPU that Ultra needs.

The YAML header at the top of this README is included for **Hugging Face Docker Spaces** metadata; the Windows `Run.bat` launcher is not used by the Dockerfile.

---

# 3. Compatible GPUs

**Hardware support depends on the selected engine, the available VRAM, the NVIDIA driver, and the installed PyTorch CUDA build.** The examples below describe the intended support of this Windows integration, **not** an exhaustive, GPU-by-GPU test matrix.

### Hardware compatibility

| GPU or platform | Fast / Quality | Ultra (SeedVR2) | Notes |
|:--|:--|:--|:--|
| **NVIDIA GeForce RTX 50 series** | CUDA expected with appropriate PyTorch build | **Potentially supported** | Requires a compatible PyTorch/CUDA wheel and drivers; not verified on every model |
| **NVIDIA GeForce RTX 40 series** | CUDA expected | **Potentially supported** | 12 GB+ models offer more room; 8 GB cards are more constrained |
| **NVIDIA GeForce RTX 30 series** | CUDA expected | **Potentially supported** | VRAM varies widely; 6–8 GB models may need aggressive memory optimizations |
| **NVIDIA GeForce RTX 20 series** | CUDA possible | **Experimental / memory-dependent** | Available VRAM and modern PyTorch support are key limitations |
| **NVIDIA GTX 16 / 10 series and older** | CPU or CUDA with a compatible older build | **Not a recommended target** | CUDA architecture support varies; modern SeedVR2 packages may not work |
| **AMD Radeon** | **CPU fallback** in this Windows build | **Not supported by this integration** | ROCm/DirectML backends are not configured here |
| **Intel Arc / integrated graphics** | **CPU fallback** | **Not supported by this integration** | Intel-specific GPU acceleration is not configured |
| **No dedicated GPU** | **CPU** | **Unavailable** | Small Fast/Quality tasks only; expect slower processing |
| **Apple Silicon / macOS** | Manual CPU path may be possible | **Not integrated** | No MPS configuration is included in this Windows-first project |

> [!IMPORTANT]
> Seeing a GPU in `nvidia-smi` is **not sufficient** to guarantee working CUDA inference. `torch.cuda.is_available()` must also return `True` in the relevant Python environment, and the model must be able to fit its runtime memory requirements.

### VRAM recommendations

| VRAM | Fast / Quality | Ultra guidance |
|:--|:--|:--|
| **Less than 6 GB** | Use small tiles or CPU for large images | **Not recommended** |
| **6–7 GB** | Generally more practical with reduced tile sizes | **Experimental**; Q4/offloading may still fail |
| **8–11 GB** | Good starting point for tiled inference | **Q4 low-VRAM profile**; slower, outcome not guaranteed |
| **12–15 GB** | Well suited to many image tasks | **FP8 profile by default**; large inputs can still exhaust VRAM |
| **16–23 GB** | More headroom | **Better candidate for Ultra**; performance varies by checkpoint and image |
| **24 GB or more** | Ample headroom for most ordinary image sizes | **Preferred for demanding Ultra workloads**, subject to specific model requirements |

These are **practical guidance bands, not official hard minimums or measured benchmarks for this application**. SeedVR2's upstream optimizations (quantized models, VAE tiling, and CPU offloading) improve the range of hardware it can attempt to run on, but there is no universal minimum VRAM that guarantees success for every image. For current upstream guidance, see the [SeedVR2 integration](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) and [PyTorch installation guide](https://pytorch.org/get-started/locally/).

For example, a **GeForce RTX 4070 12 GB** is a reasonable GPU on which to *attempt* Ultra 3B. This does **not** mean that every 4K/large-image task, checkpoint, or driver/PyTorch combination has been validated.

### Check your GPU and CUDA

**Check your NVIDIA card and VRAM** in PowerShell:

```powershell
nvidia-smi
```

**Check whether the main environment can use CUDA:**

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('CUDA:', torch.cuda.is_available()); print('PyTorch CUDA:', torch.version.cuda)"
```

**Check Ultra's separate interpreter:**

```powershell
.\.venv-ultra\Scripts\python.exe -c "import torch; print('Ultra CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Run these from the application folder. If the Ultra interpreter does not exist, its setup has not completed. If CUDA reports `False`, update your drivers or use the [official PyTorch selector](https://pytorch.org/get-started/locally/) to verify the correct wheel for your GPU.

---

## Project notes, credits and limitations

- **Ilaria Upscaler 2.0** provides the application interface, local orchestration, installation flow, and integration of upstream enhancement engines. The underlying ML models and libraries are maintained by their respective authors.
- **Real-ESRGAN / Real-ESRNet:** [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).
- **Spandrel and extra architectures:** [chaiNNer-org/spandrel](https://github.com/chaiNNer-org/spandrel).
- **SeedVR2 integration:** [numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler).
- **Hugging Face model weights:** [numz/SeedVR2_comfyUI](https://huggingface.co/numz/SeedVR2_comfyUI) and [AInVFX/SeedVR2_comfyUI](https://huggingface.co/AInVFX/SeedVR2_comfyUI).
- **Interface/server:** FastAPI, Uvicorn, HTML, CSS, and JavaScript.
- **Branding:** Ilaria Upscaler and its provided character artwork.

**Licensing:** Check the license terms of this project and of each upstream codebase and model before redistribution, commercial use, or hosting. They do not all necessarily share the same license.

**Testing:** API and interface checks can run without GPU inference, but real end-to-end SeedVR2 CUDA inference, all third-party weights, every supported model architecture, and hardware-specific installation have **not** been fully validated across Windows GPUs. Treat GPU compatibility and memory recommendations as guidance, not certifications.

**Security:** Designed primarily for local use. Before exposing the server publicly, add authentication, upload and request limits, rate limiting, safer job persistence, and appropriate file retention policies. Installing Ultra downloads and executes third-party source and weights; review upstream dependencies as appropriate.

<div align="center">

---

**Ilaria Upscaler 2.0** · Built with care by Ilaria 💖

[GitHub](https://github.com/TheStingerX/Ilaria-Upscaler) · [Ko-fi](https://ko-fi.com/ilariaowo)

</div>

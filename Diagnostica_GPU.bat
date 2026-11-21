@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title Ilaria Upscaler 2.0 - GPU Diagnostics
cls
echo.
echo   ILARIA UPSCALER 2.0 - GPU DIAGNOSTICS
echo   --------------------------------------------------
echo.
echo   [NVIDIA GPU]
nvidia-smi
if errorlevel 1 echo   [WARNING] NVIDIA driver or nvidia-smi is unavailable.
echo.
echo   [MAIN ENVIRONMENT]
if exist ".venv\Scripts\python.exe" (
 ".venv\Scripts\python.exe" -c "import torch;print('PyTorch:',torch.__version__);print('CUDA build:',torch.version.cuda);print('CUDA available:',torch.cuda.is_available());print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Unavailable')"
) else (
 echo   Environment .venv was not found.
)
echo.
echo   [ULTRA ENVIRONMENT]
if exist ".venv-ultra\Scripts\python.exe" (
 ".venv-ultra\Scripts\python.exe" -c "import torch;print('PyTorch:',torch.__version__);print('CUDA build:',torch.version.cuda);print('CUDA available:',torch.cuda.is_available());print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Unavailable')"
) else (
 echo   Ultra environment has not been installed yet.
)
echo.
pause

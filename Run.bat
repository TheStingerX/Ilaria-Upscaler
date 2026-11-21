@echo off
setlocal
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "ILARIA_OPEN_BROWSER=1"
set "PYTHONUNBUFFERED=1"
cd /d "%~dp0"
title Ilaria Upscaler 2.0
cls
echo.
echo   ILARIA UPSCALER 2.0
echo   --------------------------------------------------
echo   Local AI Image Studio  ^|  Fast  /  Quality  /  Ultra
echo.

if not exist "app.py" (
  echo   [ERROR] app.py was not found in this directory.
  goto :error
)
if not exist "bootstrap.py" (
  echo   [ERROR] bootstrap.py is missing. Extract the full update.
  goto :error
)
py -3.11 --version >nul 2>&1
if errorlevel 1 (
  echo   [ERROR] Python 3.11 was not found.
  echo   Install it from https://www.python.org/downloads/
  goto :error
)

echo   Checking your environment...
echo   Setup details: logs\setup.log
echo.
py -3.11 "bootstrap.py"
if errorlevel 1 goto :setup_error
if not exist ".venv\Scripts\python.exe" goto :setup_error

echo.
echo   --------------------------------------------------
echo   Studio ready: http://127.0.0.1:7860/
echo   Opening your browser when the server is ready...
echo   Press Ctrl+C to stop the app.
echo.
".venv\Scripts\python.exe" "app.py"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
  echo.
  echo   [ERROR] Server exited with code %EXIT_CODE%.
  echo   See the error above for details.
) else (
  echo.
  echo   Server stopped.
)
goto :end

:setup_error
echo.
echo   [ERROR] Setup failed. Check logs\setup.log.
goto :end

:error
echo.
echo   Launch could not continue.

:end
echo.
pause
endlocal

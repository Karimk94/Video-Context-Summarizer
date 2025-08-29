@echo off
echo ===================================================
echo  Setting up the Video Summarizer Web App...
echo ===================================================

:: Check if Python is installed and in PATH using a more reliable method
python -c "import sys" >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not found in your PATH.
    echo Please install Python 3.8+ and ensure it's added to your PATH.
    pause
    exit /b 1
)

:: Create a virtual environment
echo [1/5] Creating Python virtual environment in 'venv'...
python -m venv venv
if %errorlevel% neq 0 (
    echo Error: Failed to create virtual environment.
    pause
    exit /b 1
)

echo [2/5] Activating environment...
call venv\Scripts\activate.bat

echo [3/5] Upgrading pip installer...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo Error: Failed to upgrade pip.
    pause
    exit /b 1
)

echo [4/5] Installing core build tools (setuptools)...
pip install setuptools
if %errorlevel% neq 0 (
    echo Error: Failed to install setuptools.
    pause
    exit /b 1
)

:: Install all other packages from requirements.txt
echo [5/5] Installing all other dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install required packages.
    echo Please check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo ===================================================
echo  Setup complete!
echo ===================================================
echo To run the application, double-click on 'run.bat'.
echo.
pause

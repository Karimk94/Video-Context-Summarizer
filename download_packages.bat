@echo off
REM ============================================================================
REM  download_packages.bat
REM
REM  Run this script on an INTERNET-CONNECTED machine to download all required
REM  Python wheel files into the 'packages' folder.
REM
REM  The resulting 'packages' folder is included in the deployment archive so
REM  the target server can install everything OFFLINE via install_packages.bat.
REM
REM  Requirements:
REM  - Python and pip must be installed on this machine.
REM  - Run from the project root.
REM ============================================================================

@echo off
setlocal

echo Changing directory to the script's location...
cd /d "%~dp0"

echo.
echo Cleaning old packages...
if exist packages rmdir /s /q packages
mkdir packages

echo.
echo Downloading wheels for all packages in requirements.txt...
echo.

pip wheel -r requirements.txt --wheel-dir=packages

echo.
echo Creating a temporary requirements list for strict binary downloads (excluding moviepy)...
findstr /v /i "moviepy" requirements.txt > req_bin.txt

echo.
echo Downloading all requirements specifically for Python 3.12 (64-bit) to ensure compatibility...
pip download -r req_bin.txt --only-binary=:all: --platform win_amd64 --python-version 3.12 -d packages

echo.
echo Downloading all requirements specifically for Python 3.12 (32-bit) in case the server is x86...
pip download -r req_bin.txt --only-binary=:all: --platform win32 --python-version 3.12 -d packages

del req_bin.txt

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to download one or more packages.
    pause
    exit /b 1
)

echo.
echo ===================================================================
echo  Download complete!
echo ===================================================================
echo  All wheels are saved in: packages\
echo  Now run 'create_archive.bat' to bundle everything for deployment.
echo ===================================================================
echo.
pause
endlocal

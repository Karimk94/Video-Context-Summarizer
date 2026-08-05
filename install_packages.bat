@echo off
REM ============================================================================
REM  install_packages.bat
REM
REM  Creates a new virtual environment, creates a logs directory,
REM  and installs Python packages from the local 'packages' folder — NO internet.
REM ============================================================================

@echo off
setlocal

echo Changing directory to the script's location...
cd /d "%~dp0"

echo.
echo Creating virtual environment (venv)...
if exist venv rmdir /s /q venv
python -m venv venv

echo.
echo Creating logs directory...
if not exist logs mkdir logs

if not exist packages (
    echo ERROR: 'packages' folder not found.
    echo        Copy the 'packages' folder from the deployment archive.
    pause
    exit /b 1
)

echo.
echo Installing packages OFFLINE from 'packages\' folder...
call venv\Scripts\activate.bat
pip install --no-index --find-links=packages -r requirements.txt

if %errorlevel% neq 0 (
    echo ERROR: Failed to install packages.
    pause
    exit /b 1
)

echo.
echo Packages installed successfully.
echo.
pause
endlocal

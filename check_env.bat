@echo off
echo ===================================================
echo  Checking Python Environment...
echo ===================================================

:: Check if venv exists
if not exist "venv" (
    echo Error: Virtual environment 'venv' not found.
    echo Please run 'install_packages.bat' first.
    pause
    exit /b 1
)

echo [1/3] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [2/3] Checking for fastapi with pip...
pip list | findstr "fastapi"
if %errorlevel% neq 0 (
    echo.
    echo ERROR: "fastapi" is NOT found in the virtual environment.
    echo Please run "install_packages.bat" again.
) else (
    echo.
    echo SUCCESS: "fastapi" is installed.
)

echo.
echo [3/3] Attempting to import fastapi with Python...
python -c "import fastapi; print('SUCCESS: Python can import fastapi')"
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Python failed to import fastapi. The installation might be corrupt.
    echo Try deleting the 'venv' folder and running 'install_packages.bat' again.
)

echo.
echo ===================================================
echo Check complete.
echo ===================================================
pause

@echo off
echo ===================================================
echo  Checking Python Environment...
echo ===================================================

:: Check if venv exists
if not exist "venv" (
    echo Error: Virtual environment 'venv' not found.
    echo Please run 'install.bat' first.
    pause
    exit /b 1
)

echo [1/3] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [2/3] Checking for moviepy with pip...
pip list | findstr "moviepy"
if %errorlevel% neq 0 (
    echo.
    echo ERROR: "moviepy" is NOT found in the virtual environment.
    echo Please run "install.bat" again.
) else (
    echo.
    echo SUCCESS: "moviepy" is installed.
)

echo.
echo [3/3] Attempting to import moviepy with Python...
python -c "import moviepy.editor; print('SUCCESS: Python can import moviepy.editor')"
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Python failed to import moviepy. The installation might be corrupt.
    echo Try deleting the 'venv' folder and running 'install.bat' again.
)

echo.
echo ===================================================
echo Check complete.
echo ===================================================
pause

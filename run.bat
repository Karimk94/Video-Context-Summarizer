@echo off
echo ===================================================
echo  Launching the Video Summarizer Web Server...
echo ===================================================

:: Check if the virtual environment exists
if not exist "venv" (
    echo Error: Virtual environment 'venv' not found.
    echo Please run 'install_packages.bat' first to set up the environment.
    pause
    exit /b 1
)

echo.
echo [INSTRUCTIONS]
echo 1. Keep this window open. This is your local server.
echo 2. Open your web browser and go to: http://localhost:5008/
echo.
echo To stop the server, close this window or press CTRL+C.
echo.
echo ===================================================
echo Starting server...
echo.

:: Activate the virtual environment and run the FastAPI server
call venv\Scripts\activate.bat
uvicorn app:app --host 0.0.0.0 --port 5008 --workers 4

echo Server has been stopped.
pause

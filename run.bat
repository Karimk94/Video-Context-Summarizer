@echo off
echo ===================================================
echo  Launching the Video Summarizer Web Server...
echo ===================================================

:: Check if the virtual environment exists
if not exist "venv" (
    echo Error: Virtual environment 'venv' not found.
    echo Please run 'install.bat' first to set up the environment.
    pause
    exit /b 1
)

echo.
echo [INSTRUCTIONS]
echo 1. Keep this window open. This is your local server.
echo 2. Open your web browser and go to: http://172.20.10.5:5008/
echo.
echo To stop the server, close this window or press CTRL+C.
echo.
echo ===================================================
echo Starting server...
echo.

:: Activate the virtual environment and run the Flask server
call venv\Scripts\activate.bat && python app.py

echo Server has been stopped.
pause

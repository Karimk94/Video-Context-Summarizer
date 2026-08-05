@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  Creating deployment archive (includes offline packages)
echo ============================================================
echo.

REM ── Generate timestamp-based zip filename ────────────────────
set "foldername=Video_Context_Summarizer"
for /f "tokens=1-6 delims=.:/ " %%a in ("%DATE% %TIME%") do (
    set "ts=%%c-%%a-%%b_%%d-%%e-%%f"
)
set "ts=%ts: =0%"
set "zipname=%foldername%_%ts%.zip"
set "destPath=%~dp0..\%zipname%"

REM ── Call the PowerShell helper script ────────────────────────
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0archive_helper.ps1" -ZipDest "%destPath%"
set "exitcode=%ERRORLEVEL%"

if %exitcode% EQU 0 (
    echo.
    echo ============================================================
    echo  Archive created successfully!
    echo  Location: ..\%zipname%
    echo ============================================================
) else (
    echo.
    echo Failed to create archive.
    pause
    exit /b 1
)

echo.
pause
endlocal

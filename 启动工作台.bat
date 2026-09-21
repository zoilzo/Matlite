@echo off
REM ============================================================
REM  MatLite launcher (ASCII-only to avoid cmd GBK mis-parse)
REM  Launches app.py without a flashing console window.
REM ============================================================
setlocal
cd /d "%~dp0"

set "ROOT=%LOCALAPPDATA%\Programs\Python"
set "PYW="

REM 1) Prefer a real pythonw.exe under the per-user Programs\Python folder
for /d %%d in ("%ROOT%\Python3*") do (
    if exist "%%~d\pythonw.exe" set "PYW=%%~d\pythonw.exe"
)

REM 2) Fallback: resolve pythonw next to any python.exe on PATH
if not defined PYW (
    for /f "delims=" %%p in ('where python') do (
        if not defined PYW (
            set "CAND=%%~dppythonw.exe"
            if exist "%%~dppythonw.exe" set "PYW=%%~dppythonw.exe"
        )
    )
)

REM 3) Last resort: use py launcher with pythonw
if not defined PYW (
    where py >nul 2>nul
    if not errorlevel 1 set "PYW=pyw"
)

if not defined PYW (
    echo Python not found. Install Python 3 and tick "Add python.exe to PATH".
    pause
    exit /b 1
)

if "%PYW%"=="pyw" (
    start "" pyw app.py
) else (
    start "" "%PYW%" app.py
)
exit /b 0

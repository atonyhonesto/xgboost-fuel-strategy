@echo off
rem Builds (or rebuilds) the Python environment for this computer.
rem Run it once on any new PC after plugging in the drive. Needs Python 3.12 installed
rem (python.org, with "py launcher" ticked), because .venv can't move between computers.
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo Python's "py" launcher wasn't found. Install Python 3.12 from python.org first.
    pause
    exit /b 1
)

if exist ".venv" (
    echo Removing the old .venv, which was built on another computer...
    rmdir /s /q ".venv"
)

echo Creating .venv with Python 3.12...
py -3.12 -m venv .venv || (echo Couldn't create .venv. Is Python 3.12 installed? & pause & exit /b 1)
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt || (echo Package install failed. & pause & exit /b 1)

echo.
echo Done. Double-click run_app.bat to open the app.
pause

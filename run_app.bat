@echo off
rem Double-click to open the Fuel Strategy Simulator in your browser.
cd /d "%~dp0"

rem .venv only works on the computer that built it. If it's missing or broken here, rebuild it.
".venv\Scripts\python.exe" -c "import streamlit" >nul 2>nul
if errorlevel 1 (
    echo The Python environment isn't set up on this computer yet. Running setup.bat...
    call setup.bat
)

rem Started through python.exe rather than streamlit.exe: the .exe launchers remember the folder
rem path they were installed under and break silently if the folder is renamed or the drive letter changes.
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 pause

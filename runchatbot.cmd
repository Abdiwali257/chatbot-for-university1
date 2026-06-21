@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
  echo Project Python was not found. Recreate the virtual environment first.
  exit /b 1
)

"venv\Scripts\python.exe" chatbot.py %*

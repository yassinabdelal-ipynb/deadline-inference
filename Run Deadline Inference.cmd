@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Project Python is missing. Run scripts\setup.ps1 first.
  pause
  exit /b 1
)
start "Deadline inference" ".venv\Scripts\pythonw.exe" "-m" "deadline_engine" "desktop"
endlocal

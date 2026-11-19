@echo off
cd /d "%~dp0"
echo Starting AI Face Recognition Server...
powershell -ExecutionPolicy Bypass -NoExit -File "serve.ps1"
pause

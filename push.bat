@echo off
setlocal enabledelayedexpansion
title AI Sentry - GitHub Auto Push

echo ========================================================
echo        AI SENTRY PRO - GITHUB REPO SYNC UTILITY
echo ========================================================
echo.

set "GIT_CMD=C:\Users\srian\AppData\Local\GitHubDesktop\app-3.6.10\resources\app\git\cmd\git.exe"
if not exist "%GIT_CMD%" set "GIT_CMD=git"

cd /d "%~dp0"

echo [1/4] Checking Git Status...
"%GIT_CMD%" status -s

echo.
echo [2/4] Staging all project changes...
"%GIT_CMD%" add -A

set "COMMIT_MSG=%~1"
if "%COMMIT_MSG%"=="" (
    set /p "COMMIT_MSG=Enter commit message (Press ENTER for auto-timestamp): "
)

if "%COMMIT_MSG%"=="" (
    for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set "dt=%%I"
    set "TIMESTAMP=!dt:~0,4!-!dt:~4,2!-!dt:~6,2! !dt:~8,2!:!dt:~10,2!"
    set "COMMIT_MSG=Auto-update: !TIMESTAMP! [Multi-face + UI polish]"
)

echo [3/4] Committing: "!COMMIT_MSG!"
"%GIT_CMD%" commit -m "!COMMIT_MSG!"

echo.
echo [4/4] Pushing to GitHub (origin/main)...
"%GIT_CMD%" push origin main

if %ERRORLEVEL% equ 0 (
    echo.
    echo ========================================================
    echo  SUCCESS: All updates pushed to GitHub successfully!
    echo  Repo: https://github.com/revanth1029/AI-Face-Recognition-Sentry
    echo ========================================================
) else (
    echo.
    echo [ERROR] Push failed. Check network connection or credentials.
)

echo.
if "%~1"=="" pause

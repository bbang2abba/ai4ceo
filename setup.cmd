@echo off
REM Same as setup.bat: runs setup.ps1. In PowerShell you can type: .\setup
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
exit /b %ERRORLEVEL%

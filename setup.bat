@echo off
REM Thin wrapper: CMD resolves "setup" / ".\setup" to setup.bat before .cmd.
REM Same as setup.cmd — runs setup.ps1 in PowerShell.
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
exit /b %ERRORLEVEL%

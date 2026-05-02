#Requires -Version 5.1
<#
.SYNOPSIS
  ai4ceo environment setup for Windows — run this script in PowerShell only.

.DESCRIPTION
  Installs uv via the official install.ps1, installs Python with uv, then runs uv sync.
  Do not run setup.sh on Windows; use this file (or setup.cmd from Explorer/cmd).

.EXAMPLE
  # From the ai4ceo folder in PowerShell (extension optional):
  .\setup

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
  .\setup.ps1
  .\setup.cmd
#>

$ErrorActionPreference = "Stop"

$PYTHON_VER = "3.13.1"
$UV_INSTALL_URI = "https://astral.sh/uv/install.ps1"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "============================================================"
    Write-Host $Message
    Write-Host "============================================================"
}

if ($PSScriptRoot) {
    Set-Location -LiteralPath $PSScriptRoot
}

try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
} catch { }

function Get-UvCandidateDirs {
    return @(
        (Join-Path $env:USERPROFILE ".local\bin"),
        (Join-Path $env:USERPROFILE ".cargo\bin")
    )
}

function Add-UvDirsToProcessPath {
    foreach ($dir in Get-UvCandidateDirs) {
        if ((Test-Path -LiteralPath $dir) -and ($env:Path -notlike "*${dir}*")) {
            $env:Path = "${dir};${env:Path}"
        }
    }
}

function Get-UvExePath {
    Add-UvDirsToProcessPath
    foreach ($dir in Get-UvCandidateDirs) {
        $exe = Join-Path $dir "uv.exe"
        if (Test-Path -LiteralPath $exe) {
            return $exe
        }
    }
    $cmd = Get-Command "uv.exe" -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) {
        return $cmd.Source
    }
    return $null
}

function Ensure-UserPathHasUvBin {
    $uvBin = Join-Path $env:USERPROFILE ".local\bin"
    if (-not (Test-Path -LiteralPath $uvBin)) {
        return
    }
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($userPath -and ($userPath -like "*${uvBin}*")) {
        return
    }
    if ($userPath) {
        [Environment]::SetEnvironmentVariable("Path", "${userPath};${uvBin}", "User")
    } else {
        [Environment]::SetEnvironmentVariable("Path", $uvBin, "User")
    }
    Write-Host ("Added to user PATH: " + $uvBin)
    if ($env:Path -notlike "*${uvBin}*") {
        $env:Path = "${uvBin};${env:Path}"
    }
}

function Refresh-PathFromRegistry {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "${machine};${user}"
    Add-UvDirsToProcessPath
}

# uv writes progress to stderr; calling via Start-Process avoids NativeCommandError under $ErrorActionPreference Stop.
function Invoke-UvProcess {
    param(
        [Parameter(Mandatory)][string]$UvExe,
        [Parameter(Mandatory)][string[]]$Arguments
    )
    $p = Start-Process -FilePath $UvExe -ArgumentList $Arguments -WorkingDirectory $PWD.Path -NoNewWindow -Wait -PassThru
    if ($p.ExitCode -ne 0) {
        return $p.ExitCode
    }
    return 0
}

# In double-quoted strings, [ ] is wildcard syntax; use single quotes for step labels.
Write-Step '[0/6] Prerequisites (no system Python / pip / uv required)'
if (-not (Get-Command "Invoke-RestMethod" -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Invoke-RestMethod not available. PowerShell 5.1 or later is required."
    exit 1
}
Write-Host "OK: Invoke-RestMethod available for HTTPS download."
Add-UvDirsToProcessPath

Write-Step '[1/6] uv package manager'
$uvExe = Get-UvExePath
if (-not $uvExe) {
    Write-Host "uv not found. Installing via official install.ps1 (Python not required)..."
    try {
        $installScript = Invoke-RestMethod -Uri $UV_INSTALL_URI -UseBasicParsing
        Invoke-Expression $installScript
    } catch {
        Write-Host ("ERROR: uv install failed: " + $_.Exception.Message)
        Write-Host "Check network / firewall / proxy, or run:"
        Write-Host '  powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"'
        exit 1
    }
    Refresh-PathFromRegistry
    Ensure-UserPathHasUvBin
    $uvExe = Get-UvExePath
}
if (-not $uvExe) {
    Write-Host "ERROR: uv.exe not found after install. Close this window, open a new PowerShell, and run this script again from the project root."
    exit 1
}
Write-Host ("uv: " + $uvExe)

Write-Step '[2/6] Ensure user PATH includes uv directory'
Ensure-UserPathHasUvBin

Write-Step ('[3/6] Install Python ' + $PYTHON_VER + ' via uv (no system Python needed)')
$exitInstall = Invoke-UvProcess -UvExe $uvExe -Arguments @("python", "install", $PYTHON_VER)
if ($exitInstall -ne 0) {
    Write-Host ("ERROR: Failed to install Python " + $PYTHON_VER)
    exit $exitInstall
}

Write-Step '[4/6] Check pyproject.toml and remove old .venv'
if (-not (Test-Path -LiteralPath "pyproject.toml")) {
    Write-Host "ERROR: pyproject.toml not found. Run this script from the project root."
    exit 1
}

if (Test-Path -LiteralPath ".venv") {
    Write-Host "Removing existing .venv..."
    Get-Process -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -and ($_.Path -like "*\.venv\*") } |
        ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 2
    try {
        Remove-Item -LiteralPath ".venv" -Recurse -Force -ErrorAction Stop
        Write-Host "Removed .venv."
    } catch {
        Write-Host "ERROR: Could not delete .venv. Delete the folder manually, then run this script again."
        exit 1
    }
}

Write-Step '[5/6] uv sync (may take 1-3 minutes)'
$exitSync = Invoke-UvProcess -UvExe $uvExe -Arguments @("sync", "--python", $PYTHON_VER)
if ($exitSync -ne 0) {
    Write-Host "ERROR: uv sync failed."
    exit $exitSync
}

Write-Step '[6/6] Verification'
$prevEa = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $uvExe pip list 2>$null | Select-String -Pattern "langchain|openai|streamlit|python" -CaseSensitive:$false
$ErrorActionPreference = $prevEa

Write-Host ""
Write-Host "============================================================"
Write-Host ("Setup complete. Python " + $PYTHON_VER)
Write-Host "============================================================"
Write-Host ""
Write-Host "Activate venv (PowerShell):"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "If script execution is blocked, run once (CurrentUser):"
Write-Host "  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned"
Write-Host ""

# Runs the application using the local venv_win environment.
$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "venv_win\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Could not find virtual environment python at '$python'." -ForegroundColor Red
    Write-Host "Create it with: python -m venv package_win\venv_win"
    exit 1
}

# main.py lives in the project root (the parent of this folder); run from there so
# any working-directory-relative paths resolve against the project root.
$root = Split-Path -Parent $PSScriptRoot
Set-Location -Path $root
& $python "main.py" @args
exit $LASTEXITCODE

<#
.SYNOPSIS
    Runs local containerized Act CI per Synaptic Antibody AB-35.
.DESCRIPTION
    Executes GitHub Actions workflows locally using nektos/act and Docker.
    Passes ACT=true and CI=true environment variables to satisfy workflow guards.
#>

[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ActArgs
)

$ProgressPreference = 'SilentlyContinue'
$env:CI = 'true'
$env:ACT = 'true'
$env:PAGER = 'cat'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Push-Location $ProjectRoot

try {
    Write-Host "==> [Act CI Sentry] Checking local containerized CI prerequisites..." -ForegroundColor Cyan

    if (-not (Get-Command "act" -ErrorAction SilentlyContinue)) {
        Write-Error "==> [Act CI Sentry] 'act' CLI is not installed or not in PATH. Install via: choco install gh-act or winget install nektos.act"
        exit 1
    }

    if (-not (Get-Command "docker" -ErrorAction SilentlyContinue)) {
        Write-Error "==> [Act CI Sentry] 'docker' is not installed or not in PATH."
        exit 1
    }

    # Verify Docker daemon is running
    docker info > $null 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Error "==> [Act CI Sentry] Docker daemon is not running. Please start Docker Desktop or the dockerd service."
        exit 1
    }

    $envFile = Join-Path $ProjectRoot ".env.ci"
    $actrcFile = Join-Path $ProjectRoot ".actrc"

    $cmdArgs = @()
    if (Test-Path $envFile) {
        $cmdArgs += @("--env-file", $envFile)
    } else {
        $cmdArgs += @("--env", "ACT=true", "--env", "CI=true")
    }

    if (Test-Path $actrcFile) {
        $cmdArgs += @("--action-offline-mode=false")
    }

    if ($ActArgs -and $ActArgs.Count -gt 0) {
        $cmdArgs += $ActArgs
    } else {
        $cmdArgs += @("push")
    }

    Write-Host "==> [Act CI Sentry] Executing: act $($cmdArgs -join ' ')" -ForegroundColor Green
    & act @cmdArgs
    $actExitCode = $LASTEXITCODE

    if ($actExitCode -eq 0) {
        Write-Host "==> [Act CI Sentry] Local Act CI passed successfully (Exit Code 0)." -ForegroundColor Green
        exit 0
    } else {
        Write-Error "==> [Act CI Sentry] Local Act CI failed with exit code $actExitCode."
        exit $actExitCode
    }
} finally {
    Pop-Location
}

param(
    [string]$Python = $env:SODLAB_PYTHON
)

$ErrorActionPreference = "Stop"

if (-not $Python) {
    $Python = "E:\anaconda\envs\dl\python.exe"
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python executable not found: $Python. Set SODLAB_PYTHON or pass -Python."
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
Set-Location $ProjectRoot

& $Python -m sodlab.cli generate --output data/synthetic
if ($LASTEXITCODE -ne 0) { throw "Dataset generation failed." }

& $Python -m sodlab.cli validate --data data/synthetic/data.yaml
if ($LASTEXITCODE -ne 0) { throw "Dataset validation failed." }

& $Python -m sodlab.cli train --config configs/smoke.yaml
if ($LASTEXITCODE -ne 0) { throw "Smoke training failed." }

Write-Host "Smoke pipeline completed."

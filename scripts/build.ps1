param([switch]$SourceOnly, [switch]$SkipBuild)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$buildPython = Join-Path $projectRoot '.build-venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $buildPython)) { throw 'Create a CPU build environment first. See docs\PORTABILITY.md.' }
Push-Location -LiteralPath $projectRoot
try {
    $arguments = @((Join-Path $projectRoot 'scripts\build_release.py'))
    if ($SourceOnly) { $arguments += '--source-only' }
    if ($SkipBuild) { $arguments += '--skip-build' }
    & $buildPython @arguments
    if ($LASTEXITCODE -ne 0) { throw 'Release build failed.' }
} finally { Pop-Location }

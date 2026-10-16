param([switch]$Hardware, [switch]$Model)
. (Join-Path $PSScriptRoot 'common.ps1')
Push-Location -LiteralPath $projectRoot
$previousGpuTests = $env:RUN_GPU_TESTS
$previousModelTests = $env:RUN_MODEL_TESTS
try {
    $env:RUN_GPU_TESTS = if ($Hardware) { '1' } else { '0' }
    $env:RUN_MODEL_TESTS = if ($Model -or $Hardware) { '1' } else { '0' }
    & $projectPython -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
} finally {
    $env:RUN_GPU_TESTS = $previousGpuTests
    $env:RUN_MODEL_TESTS = $previousModelTests
    Pop-Location
}

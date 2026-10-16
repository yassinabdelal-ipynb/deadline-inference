. (Join-Path $PSScriptRoot 'common.ps1')
Push-Location -LiteralPath $projectRoot
try {
    & $projectPython -m deadline_engine doctor --output (Join-Path $projectRoot 'results\environment.json')
    if ($LASTEXITCODE -ne 0) { throw 'Doctor failed.' }
} finally { Pop-Location }

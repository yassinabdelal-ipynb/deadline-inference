. (Join-Path $PSScriptRoot 'common.ps1')
Push-Location -LiteralPath $projectRoot
try {
    & $projectPython -m deadline_engine desktop
    if ($LASTEXITCODE -ne 0) { throw 'Application stopped with an error.' }
} finally { Pop-Location }

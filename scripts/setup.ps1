param(
    [switch]$DownloadWeights,
    [ValidateSet('cuda','cpu')][string]$Backend = 'cpu',
    [string]$Python = '',
    [switch]$ReplaceBackend,
    [string]$Environment = ''
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    $existingPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if ($Python) { $pythonCommand = $Python; $pythonArgs = @() }
    elseif (Test-Path -LiteralPath $existingPython) { $pythonCommand = $existingPython; $pythonArgs = @() }
    elseif (Get-Command py -ErrorAction SilentlyContinue) { $pythonCommand = 'py'; $pythonArgs = @('-3.14') }
    elseif (Get-Command python -ErrorAction SilentlyContinue) { $pythonCommand = 'python'; $pythonArgs = @() }
    else { throw 'Install official 64-bit Python 3.14 with Tcl/Tk from https://www.python.org/downloads/windows/, then rerun. No system settings were changed.' }
    $bootstrapArgs = @((Join-Path $PSScriptRoot 'bootstrap.py'), '--backend', $Backend)
    if ($DownloadWeights) { $bootstrapArgs += '--download-weights' }
    if ($ReplaceBackend) { $bootstrapArgs += '--replace-backend' }
    if ($Environment) { $bootstrapArgs += @('--environment', $Environment) }
    & $pythonCommand @pythonArgs @bootstrapArgs
    if ($LASTEXITCODE -ne 0) { throw 'Setup stopped. Read the explanation above; existing backends are preserved unless -ReplaceBackend is supplied.' }
} finally { Pop-Location }

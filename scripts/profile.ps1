param([ValidateSet('cuda','cpu')][string]$Device = 'cpu', [int]$Samples = 10, [int]$Batch = 4, [string]$Output = 'results/profile', [string]$Model = 'mobilenet_v3_small', [string]$Weights = '', [int]$Classes = 1000, [int]$ImageSize = 224)
. (Join-Path $PSScriptRoot 'common.ps1')
Push-Location -LiteralPath $projectRoot
try {
    $modelOptions = @('--model', $Model, '--classes', $Classes, '--image-size', $ImageSize)
    if ($Weights) { $modelOptions += @('--weights', $Weights) }
    & $projectPython -m deadline_engine profile --device $Device --samples $Samples --batch $Batch --output $Output @modelOptions
    if ($LASTEXITCODE -ne 0) { throw 'Device test failed; read the error above.' }
} finally { Pop-Location }

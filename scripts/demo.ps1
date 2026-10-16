param([ValidateSet('cuda','cpu')][string]$Device = 'cpu', [double]$Duration = 4, [double]$Rate = 100, [string]$Output = 'results/demo', [string]$Model = 'mobilenet_v3_small', [string]$Weights = '', [int]$Classes = 1000, [int]$ImageSize = 224)
. (Join-Path $PSScriptRoot 'common.ps1')
Push-Location -LiteralPath $projectRoot
try {
    $modelOptions = @('--model', $Model, '--classes', $Classes, '--image-size', $ImageSize)
    if ($Weights) { $modelOptions += @('--weights', $Weights) }
    & $projectPython -m deadline_engine demo --device $Device --duration $Duration --rate $Rate --output $Output @modelOptions
    if ($LASTEXITCODE -ne 0) { throw 'Demo failed; read the error above.' }
} finally { Pop-Location }

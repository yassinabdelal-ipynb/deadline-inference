param([ValidateSet('cuda','cpu')][string]$Device = 'cpu', [int]$Repeats = 3, [double]$Duration = 2, [double]$Rate = 150, [string]$Output = 'results/benchmark', [string]$Model = 'mobilenet_v3_small', [string]$Weights = '', [int]$Classes = 1000, [int]$ImageSize = 224)
. (Join-Path $PSScriptRoot 'common.ps1')
Push-Location -LiteralPath $projectRoot
try {
    $modelOptions = @('--model', $Model, '--classes', $Classes, '--image-size', $ImageSize)
    if ($Weights) { $modelOptions += @('--weights', $Weights) }
    & $projectPython -m deadline_engine benchmark --device $Device --repeats $Repeats --duration $Duration --rate $Rate --output $Output @modelOptions
    if ($LASTEXITCODE -ne 0) { throw 'Benchmark failed; read the error above.' }
} finally { Pop-Location }

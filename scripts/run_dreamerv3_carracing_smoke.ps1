$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$UpstreamRoot = Join-Path $ProjectRoot "external\dreamer_v3"
if (-not (Test-Path (Join-Path $UpstreamRoot "main.py"))) {
    throw "DreamerV3 checkout not found at $UpstreamRoot"
}

$OldLocation = Get-Location
$OldPythonPath = $env:PYTHONPATH
$OldEnvName = $env:env_name
$OldOverride = $env:override_config
$OldRunName = $env:run_name
$OldPrecision = $env:DREAMER_PRECISION

try {
    $env:PYTHONPATH = "$ProjectRoot;$UpstreamRoot;$OldPythonPath"
    $env:env_name = "dmc-CarRacing-v3"
    $env:run_name = "carracing-smoke"
    $env:DREAMER_PRECISION = "float32"
    Remove-Item Env:override_config -ErrorAction SilentlyContinue

    Push-Location $UpstreamRoot

    Write-Host "Phase 1/2: construct original DreamerV3 and test the wrapped environment"
    python (Join-Path $ProjectRoot "scripts\smoke_dreamerv3_carracing.py") --steps 5
    if ($LASTEXITCODE -ne 0) { throw "DreamerV3 environment smoke test failed." }

    Write-Host "Phase 2/2: one short online training epoch (prefill + one World Model/Actor/Critic update)"
    python main.py --config_file integrations/dreamerv3_carracing/config.py --steps_per_epoch 1 --saving_period_epoch 1 --eval_period_epoch 999999 --log_figure_period_epoch 999999 --step_log_period 1 --verbose_progress_bar 2
    if ($LASTEXITCODE -ne 0) { throw "DreamerV3 one-update training smoke test failed." }
}
finally {
    if ((Get-Location).Path -eq $UpstreamRoot) { Pop-Location }
    Set-Location $OldLocation.Path
    $env:PYTHONPATH = $OldPythonPath
    $env:env_name = $OldEnvName
    $env:override_config = $OldOverride
    $env:run_name = $OldRunName
    $env:DREAMER_PRECISION = $OldPrecision
}

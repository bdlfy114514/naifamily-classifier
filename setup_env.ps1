param(
    [ValidateSet('xpu', 'cuda', 'cpu')][string]$Backend = 'xpu',
    [string]$Python = '',
    [ValidateSet('cu126', 'cu128', 'cu130')][string]$CudaWheel = 'cu128'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskTemp = Join-Path $PSScriptRoot '.tmp'
New-Item -ItemType Directory -Force -Path $taskTemp | Out-Null
$previousTemp = $env:TEMP
$previousTmp = $env:TMP
$env:TEMP = $taskTemp
$env:TMP = $taskTemp
function Check-Exit([string]$Step) {
    if ($LASTEXITCODE -ne 0) { throw "$Step failed (exit $LASTEXITCODE)" }
}
try {
    $venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $venvPython)) {
        if ($Python) {
            & $Python -m venv .venv
        } else {
            # 使用常见且兼容这些固定依赖的 Python 3.12。
            & py -3.12 -m venv .venv
        }
        Check-Exit 'Create venv'
    }
    & $venvPython -m ensurepip --upgrade
    Check-Exit 'Bootstrap pip'
    & $venvPython -m pip install -r requirements.txt --cache-dir .tmp\pip-cache
    Check-Exit 'Install common dependencies'
    $wheel = if ($Backend -eq 'cuda') { $CudaWheel } else { $Backend }
    # 同一个环境选择一个 GPU wheel；切换后端时显式替换，避免 pip 保留旧后端。
    & $venvPython -m pip install --force-reinstall --no-deps torch==2.9.1 torchvision==0.24.1 --index-url "https://download.pytorch.org/whl/$wheel" --cache-dir .tmp\pip-cache
    Check-Exit 'Install backend wheels'
    # 安装/核对后端依赖（XPU wheel 包含所需的运行时依赖声明）。
    & $venvPython -m pip install torch==2.9.1 torchvision==0.24.1 --index-url "https://download.pytorch.org/whl/$wheel" --cache-dir .tmp\pip-cache
    Check-Exit 'Install backend dependencies'
    & $venvPython -m pip check
    Check-Exit 'Check dependencies'
    & $venvPython -c "import torch; print('PyTorch:',torch.__version__); print('CUDA:',torch.cuda.is_available()); print('XPU:',hasattr(torch,'xpu') and torch.xpu.is_available())"
    Check-Exit 'Check import'
    & $venvPython -m pip freeze | Set-Content -Encoding utf8 requirements-lock.txt
    Write-Host "Environment ready: $venvPython"
} finally {
    $env:TEMP = $previousTemp
    $env:TMP = $previousTmp
}

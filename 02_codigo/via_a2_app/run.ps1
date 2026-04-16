#Requires -Version 5.1
# Inicia o app Via A2 localmente.
# Uso:
#   .\run.ps1            # sobe apenas local
#   .\run.ps1 -Ngrok     # sobe local + tunel ngrok
#   .\run.ps1 -Port 8095 -Ngrok

param(
    [int]$Port = 8095,
    [switch]$Ngrok
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "==> Instalando dependencias (se necessario)..." -ForegroundColor Cyan
python -m pip install --quiet -r requirements.txt

if ($Ngrok) {
    Write-Host "==> Subindo tunel ngrok em segundo plano..." -ForegroundColor Cyan
    Start-Process -NoNewWindow python -ArgumentList "ngrok_tunnel.py","--port",$Port
    Start-Sleep -Seconds 2
}

Write-Host "==> Iniciando servidor na porta $Port..." -ForegroundColor Cyan
Write-Host "    Local: http://localhost:$Port"
Write-Host ""
python -m uvicorn app:app --host 0.0.0.0 --port $Port

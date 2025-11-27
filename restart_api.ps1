# PowerShell script to restart FastAPI server and test the scaling fix

Write-Host "=== Stopping FastAPI Server ===" -ForegroundColor Yellow
# Find and stop the process on port 8000
$process = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
if ($process) {
    Stop-Process -Id $process -Force
    Write-Host "Stopped process with PID: $process" -ForegroundColor Green
    Start-Sleep -Seconds 2
} else {
    Write-Host "No process found on port 8000" -ForegroundColor Yellow
}

Write-Host "`n=== Starting FastAPI Server ===" -ForegroundColor Yellow
# Navigate to api directory and start server
Set-Location api
Start-Process python -ArgumentList "-m", "uvicorn", "app:app", "--reload" -WindowStyle Minimized
Set-Location ..
Write-Host "FastAPI server starting on http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "Waiting for server to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

Write-Host "`n=== Testing API Connection ===" -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:8000/transactions" -Method GET -TimeoutSec 5
    Write-Host "✓ API is responding!" -ForegroundColor Green
} catch {
    Write-Host "✗ API not responding yet. Wait a few more seconds and try again." -ForegroundColor Red
}

Write-Host "`n=== Server Ready ===" -ForegroundColor Green
Write-Host "FastAPI server is running at: http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host "API docs available at: http://127.0.0.1:8000/docs" -ForegroundColor Cyan


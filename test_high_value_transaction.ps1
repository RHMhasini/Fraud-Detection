# PowerShell script to test high-value transaction with scaling fix

Write-Host "=== Testing High-Value Transaction (Scaling Fix) ===" -ForegroundColor Yellow

# Test transaction with extremely high purchase value
$testTransaction = @{
    signup_time = "2025-11-22T10:00:00"
    purchase_time = "2025-11-22T10:05:00"
    purchase_value = 100000000.00
    age = 30.0
    device_id_count = 1.0
    ip_address_count = 1.0
    hashed_user_id = "test_user_high_value"
    hashed_device_id = "test_device_high_value"
    hashed_ip_address = "192.168.1.100"
    source_Ads = 0
    source_Direct = 1
    source_SEO = 0
    browser_Chrome = 1
    browser_FireFox = 0
    browser_IE = 0
    browser_Opera = 0
    browser_Safari = 0
    sex_F = 0
    sex_M = 1
} | ConvertTo-Json

Write-Host "`nSubmitting transaction with purchase_value = 100,000,000..." -ForegroundColor Cyan
Write-Host "Expected: High fraud score (ensemble > 0.8 for 'deny' status)" -ForegroundColor Cyan

try {
    $response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/transaction" -Method POST -Body $testTransaction -ContentType "application/json"
    
    Write-Host "`n=== Results ===" -ForegroundColor Green
    Write-Host "Transaction ID: $($response.transaction_id)" -ForegroundColor White
    Write-Host "RF Score: $([math]::Round($response.RF_score, 4))" -ForegroundColor $(if ($response.RF_score -gt 0.5) { "Red" } else { "Yellow" })
    Write-Host "XGB Score: $([math]::Round($response.XGB_score, 4))" -ForegroundColor $(if ($response.XGB_score -gt 0.5) { "Red" } else { "Yellow" })
    Write-Host "Ensemble Score: $([math]::Round($response.ensemble_score, 4))" -ForegroundColor $(if ($response.ensemble_score -gt 0.8) { "Red" } elseif ($response.ensemble_score -gt 0.2) { "Yellow" } else { "Green" })
    
    $status = if ($response.ensemble_score -lt 0.2) { "PASS" } elseif ($response.ensemble_score -lt 0.8) { "FLAG" } else { "DENY" }
    Write-Host "Status: $status" -ForegroundColor $(if ($status -eq "DENY") { "Red" } elseif ($status -eq "FLAG") { "Yellow" } else { "Green" })
    
    if ($response.ensemble_score -gt 0.8) {
        Write-Host "`n✓ SUCCESS: High-value transaction correctly flagged as DENY!" -ForegroundColor Green
    } elseif ($response.ensemble_score -gt 0.2) {
        Write-Host "`n⚠ WARNING: Transaction flagged but score may be lower than expected" -ForegroundColor Yellow
    } else {
        Write-Host "`n✗ ISSUE: Score is too low. Scaling may not be working correctly." -ForegroundColor Red
    }
    
} catch {
    Write-Host "`n✗ Error: $($_.Exception.Message)" -ForegroundColor Red
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $responseBody = $reader.ReadToEnd()
        Write-Host "Response: $responseBody" -ForegroundColor Red
    }
}


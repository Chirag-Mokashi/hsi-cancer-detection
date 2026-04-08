$RC = '.\rclone-v1.73.3-windows-amd64\rclone.exe'

Write-Host '=== gdrive:HSI/ top-level folders ==='
& $RC lsd gdrive:HSI/ 2>&1

Write-Host ''
Write-Host '=== gdrive:HSI/band_selection/ ==='
& $RC lsf gdrive:HSI/band_selection/ 2>&1 | Select-Object -First 30

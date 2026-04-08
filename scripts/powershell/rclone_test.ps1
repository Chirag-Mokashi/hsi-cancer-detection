$rc = '.\rclone-v1.73.3-windows-amd64\rclone.exe'
'test' | Out-File "$env:TEMP\ping.txt" -Encoding ascii
& $rc copyto "$env:TEMP\ping.txt" gdrive:HSI/preprocessed/ping.txt
if ($LASTEXITCODE -eq 0) {
    Write-Host 'WRITE TEST: OK'
    & $rc deletefile gdrive:HSI/preprocessed/ping.txt
} else {
    Write-Host ('WRITE TEST: FAILED (exit=' + $LASTEXITCODE + ')')
}

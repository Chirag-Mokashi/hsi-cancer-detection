$rc = '.\rclone-v1.73.3-windows-amd64\rclone.exe'
while ($true) {
    $n = & $rc ls gdrive:HSI/preprocessed | Measure-Object -Line | Select-Object -ExpandProperty Lines
    $free = [math]::Round((Get-PSDrive C).Free / 1GB, 1)
    $ts = Get-Date -Format 'HH:mm:ss'
    Write-Host ($ts + '  Drive: ' + $n + '/135  C: free: ' + $free + ' GB')
    Start-Sleep 600
}

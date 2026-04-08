$RC  = '.\rclone-v1.73.3-windows-amd64\rclone.exe'
$SRC = 'C:\Users\mokas\OneDrive\Desktop\HSI\preprocessed'
$DST = 'gdrive:HSI/preprocessed'

$localFiles = Get-ChildItem $SRC -Filter '*.h5' -ErrorAction SilentlyContinue |
              Select-Object -ExpandProperty Name | Sort-Object

$driveFiles = & $RC lsf $DST --include '*.h5' 2>$null |
              ForEach-Object { $_.TrimEnd('/') } | Sort-Object

Write-Host '========================================================'
Write-Host '  EXTRA FILES ON DRIVE (not in local)'
Write-Host '========================================================'
$extra = $driveFiles | Where-Object { $localFiles -notcontains $_ }
if ($extra.Count -eq 0) {
    Write-Host '  None.'
} else {
    $extra | ForEach-Object { Write-Host "  $_" }
}

Write-Host ''
Write-Host '========================================================'
Write-Host '  DETAILED DRIVE FILE LIST WITH SIZES'
Write-Host '========================================================'

$driveJson = & $RC lsjson $DST --include '*.h5' 2>$null | ConvertFrom-Json
$totalSize = 0

$driveJson | Sort-Object Name | ForEach-Object {
    $sizeMB = [math]::Round($_.Size / 1MB, 2)
    $totalSize += $_.Size
    $flag = if ($localFiles -notcontains $_.Name) { ' [EXTRA]' } else { '' }
    Write-Host ("  {0,-55} {1,10} MB{2}" -f $_.Name, $sizeMB, $flag)
}

$totalGB = [math]::Round($totalSize / 1GB, 3)
Write-Host ''
Write-Host ("  Total Drive size : $totalGB GB across $($driveJson.Count) files")

Write-Host ''
Write-Host '========================================================'
Write-Host '  LOCAL FILE SIZE SUMMARY'
Write-Host '========================================================'
$localTotal = 0
Get-ChildItem $SRC -Filter '*.h5' | Sort-Object Name | ForEach-Object {
    $sizeMB = [math]::Round($_.Length / 1MB, 2)
    $localTotal += $_.Length
    Write-Host ("  {0,-55} {1,10} MB" -f $_.Name, $sizeMB)
}
$localGB = [math]::Round($localTotal / 1GB, 3)
Write-Host ''
Write-Host ("  Total local size : $localGB GB across $($localFiles.Count) files")

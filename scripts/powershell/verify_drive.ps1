param([switch]$SizeCheck)

$RC   = '.\rclone-v1.73.3-windows-amd64\rclone.exe'
$SRC  = 'C:\Users\mokas\OneDrive\Desktop\HSI\preprocessed'
$DST  = 'gdrive:HSI/preprocessed'

Write-Host '========================================================'
Write-Host '  DRIVE UPLOAD VERIFICATION'
Write-Host '========================================================'
Write-Host ''

# Phase 1: Count check
Write-Host '--- Phase 1: File count ---'

$localFiles = Get-ChildItem $SRC -Filter '*.h5' -ErrorAction SilentlyContinue |
              Select-Object -ExpandProperty Name | Sort-Object

$driveFiles = & $RC lsf $DST --include '*.h5' 2>$null |
              ForEach-Object { $_.TrimEnd('/') } | Sort-Object

$localCount = $localFiles.Count
$driveCount = $driveFiles.Count

Write-Host ("  Local  : $localCount files")
Write-Host ("  Drive  : $driveCount files")
Write-Host ("  Target : 135 files")
Write-Host ''

$missingOnDrive = $localFiles | Where-Object { $driveFiles -notcontains $_ }
$extraOnDrive   = $driveFiles | Where-Object { $localFiles -notcontains $_ }

if ($missingOnDrive.Count -eq 0) {
    Write-Host '  [OK] All local files are present on Drive.'
} else {
    Write-Warning ("  MISSING from Drive (" + $missingOnDrive.Count + "):")
    $missingOnDrive | ForEach-Object { Write-Warning ("    - $_") }
}

if ($extraOnDrive.Count -gt 0) {
    Write-Warning ("  Extra on Drive (not in local, " + $extraOnDrive.Count + "):")
    $extraOnDrive | ForEach-Object { Write-Warning ("    + $_") }
}

Write-Host ''
Write-Host '--- Count summary ---'
if ($driveCount -eq 135 -and $missingOnDrive.Count -eq 0) {
    Write-Host '  STATUS: PASS - 135/135 files on Drive.'
} else {
    Write-Host ("  STATUS: INCOMPLETE - $driveCount/135 on Drive, $($missingOnDrive.Count) missing.")
}

Write-Host ''
Write-Host '========================================================'
Write-Host '  Phase 2 (size check) -- run with -SizeCheck flag'
Write-Host '========================================================'

if (-not $SizeCheck) {
    Write-Host '  Skipped. Re-run as:  .\verify_drive.ps1 -SizeCheck'
    exit 0
}

Write-Host ''
Write-Host '--- Phase 2: Size comparison (local vs Drive) ---'

$pass    = 0
$fail    = @()
$skipped = @()

foreach ($f in $localFiles) {
    $localPath = Join-Path $SRC $f
    $drivePath = "$DST/$f"

    $item  = Get-Item $localPath -Force -ErrorAction SilentlyContinue
    $attr  = if ($item) { [int]$item.Attributes } else { 0 }
    $isOD  = ($attr -band 0x80000) -ne 0

    $localSizeBytes = $item.Length

    $driveSizeLine = & $RC size $drivePath 2>$null | Select-String 'Total size'
    if (-not $driveSizeLine) {
        $fail += "$f  (not found on Drive)"
        continue
    }

    if ($driveSizeLine -match '\((\d+)\s+Bytes\)') {
        $driveSizeBytes = [long]$Matches[1]
    } else {
        $skipped += "$f  (could not parse Drive size)"
        continue
    }

    if ($localSizeBytes -eq $driveSizeBytes) {
        $pass++
    } else {
        $diff = $driveSizeBytes - $localSizeBytes
        $fail += "$f  local=$localSizeBytes  drive=$driveSizeBytes  diff=$diff"
    }
}

Write-Host ("  Pass   : $pass")
Write-Host ("  Fail   : " + $fail.Count)
Write-Host ("  Skipped: " + $skipped.Count)
Write-Host ''

if ($fail.Count -gt 0) {
    Write-Warning 'SIZE MISMATCHES:'
    $fail | ForEach-Object { Write-Warning "  $_" }
}
if ($skipped.Count -gt 0) {
    Write-Warning 'SKIPPED (parse error):'
    $skipped | ForEach-Object { Write-Warning "  $_" }
}

if ($fail.Count -eq 0 -and $skipped.Count -eq 0) {
    Write-Host "  STATUS: PASS - all $pass files match size on Drive."
} else {
    Write-Host '  STATUS: ISSUES FOUND - check warnings above.'
}

$RC  = '.\rclone-v1.73.3-windows-amd64\rclone.exe'
$SRC = 'C:\Users\mokas\OneDrive\Desktop\HSI\preprocessed'
$DST = 'gdrive:HSI/preprocessed'

$missing = @(
    'samples.h5',
    'top-level_ROI_01_C10_T.h5',
    'top-level_ROI_01_C11_T.h5',
    'top-level_ROI_01_C12_T.h5',
    'top-level_ROI_02_C10_NT.h5',
    'top-level_ROI_02_C11_NT.h5',
    'top-level_ROI_02_C12_NT.h5'
)

foreach ($f in $missing) {
    Write-Host "Uploading $f ..."
    & $RC copy "$SRC\$f" $DST --progress 2>&1
    Write-Host "Done: $f"
    Write-Host ''
}

Write-Host 'All 7 files uploaded.'

$SRC   = 'C:\Users\mokas\OneDrive\Desktop\HSI\preprocessed'
$DST   = 'gdrive:HSI/preprocessed'
$RC    = '.\rclone-v1.73.3-windows-amd64\rclone.exe'
$MIN_FREE_GB = 5

# ── Pre-flight ───────────────────────────────────────────────────────────────
Write-Host '=== PRE-FLIGHT CHECK ==='
Write-Host ('OneDrive process : ' + ((Get-Process OneDrive -ErrorAction SilentlyContinue | Measure-Object).Count) + ' running')
Write-Host ('C: free GB       : ' + [math]::Round((Get-PSDrive C).Free / 1GB, 1))

$driveCount = & $RC ls $DST --max-depth 1 2>$null | Measure-Object -Line | Select-Object -ExpandProperty Lines
Write-Host ('Drive h5 count   : ' + $driveCount + ' of 135')

'ok' | Out-File "$env:TEMP\gdping.txt" -Encoding ascii
& $RC copyto "$env:TEMP\gdping.txt" "$DST/gdping.txt" 2>$null
if ($LASTEXITCODE -eq 0) {
    & $RC deletefile "$DST/gdping.txt" 2>$null
    Write-Host 'rclone write test: OK'
} else {
    Write-Warning 'rclone write test FAILED - check rclone config and Drive auth.'
    exit 1
}
Write-Host ''

# ── Build work list: files in SRC not yet on Drive ───────────────────────────
$allSrc   = Get-ChildItem $SRC -Filter *.h5 -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty Name
$onDrive  = & $RC lsf $DST --include '*.h5' 2>$null | ForEach-Object { $_.TrimEnd('/') }
$MISSING  = $allSrc | Where-Object { $onDrive -notcontains $_ }
Write-Host ('Work list: ' + $MISSING.Count + ' to upload, ' + ($allSrc.Count - $MISSING.Count) + ' already on Drive')
Write-Host ''

# ── Functions ────────────────────────────────────────────────────────────────
function FreeSpace-File {
    param($filePath)
    $invoked = $false
    try {
        $shell  = New-Object -ComObject Shell.Application
        $folder = $shell.NameSpace((Split-Path $filePath))
        $item   = $folder.ParseName((Split-Path $filePath -Leaf))
        if ($item) {
            $verbs = $item.Verbs()
            for ($i = 0; $i -lt $verbs.Count; $i++) {
                if ($verbs.Item($i).Name -eq 'Free up space') {
                    $verbs.Item($i).DoIt()
                    $invoked = $true
                    break
                }
            }
        }
    } catch {}
    if (-not $invoked) {
        & attrib +U $filePath 2>$null
    }
}

function WaitForDehydration {
    param($filePath, $TimeoutSec = 300)
    $sw = [Diagnostics.Stopwatch]::StartNew()
    while ($sw.Elapsed.TotalSeconds -lt $TimeoutSec) {
        $attr = [int](Get-Item $filePath -Force -ErrorAction SilentlyContinue).Attributes
        if (($attr -band 0x80000) -ne 0) {
            Write-Host ('    Dehydrated in ' + [math]::Round($sw.Elapsed.TotalSeconds) + 's')
            return $true
        }
        Start-Sleep 5
    }
    Write-Warning '    Dehydration timed out (file may still be local)'
    return $false
}

function HydrateFile {
    param($filePath, $TimeoutSec = 1800)
    $name   = Split-Path $filePath -Leaf
    $sizeMB = [math]::Round((Get-Item $filePath -Force).Length / 1MB, 1)
    $ts     = Get-Date -Format 'HH:mm:ss'

    $attr = [int](Get-Item $filePath -Force).Attributes
    if (($attr -band 0x80000) -eq 0) {
        Write-Host ('  ' + $ts + ' Already local: ' + $name + ' (' + $sizeMB + ' MB)')
        return $true
    }

    Write-Host ('  ' + $ts + ' Hydrating ' + $name + ' (' + $sizeMB + ' MB)...')
    $freeBefore = (Get-PSDrive C).Free
    $sw = [Diagnostics.Stopwatch]::StartNew()

    $job = Start-Job -ScriptBlock {
        param($p)
        $buf = [byte[]]::new(4MB)
        $s   = [IO.FileStream]::new($p, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
        $tot = 0
        while (($r = $s.Read($buf, 0, $buf.Length)) -gt 0) { $tot += $r }
        $s.Close()
        return $tot
    } -ArgumentList $filePath

    $completed = Wait-Job $job -Timeout $TimeoutSec
    if ($completed -eq $null) {
        Stop-Job $job; Remove-Job $job -Force; $sw.Stop()
        Write-Warning ('    Hydrate TIMED OUT after ' + $TimeoutSec + 's')
        return $false
    }
    $bytesRead = Receive-Job $job
    $jobState  = $job.State
    Remove-Job $job -Force
    $sw.Stop()

    if ($jobState -eq 'Failed') {
        Write-Warning '    Hydrate FAILED'
        return $false
    }

    $freeAfter    = (Get-PSDrive C).Free
    $downloadedGB = [math]::Round(($freeBefore - $freeAfter) / 1GB, 2)
    Write-Host ('    Read ' + [math]::Round($bytesRead/1MB,1) + ' MB in ' +
                [math]::Round($sw.Elapsed.TotalSeconds,1) + 's  (C: used +' + $downloadedGB + ' GB)')
    return $true
}

# ── Main loop ────────────────────────────────────────────────────────────────
$total   = $MISSING.Count
$done    = 0
$skipped = 0
$failed  = @()

Write-Host ('Processing ' + $total + ' files one at a time.')
Write-Host ''

foreach ($f in $MISSING) {
    $srcPath = Join-Path $SRC $f
    $dstPath = "$DST/$f"

    $done++
    $ts = Get-Date -Format 'HH:mm:ss'
    Write-Host ($ts + ' [' + $done + '/' + $total + '] ' + $f)

    # Free space check
    $freeGB = [math]::Round((Get-PSDrive C).Free / 1GB, 1)
    if ($freeGB -lt $MIN_FREE_GB) {
        Write-Warning ('  Low C: space (' + $freeGB + ' GB). Stopping. Free up space and re-run.')
        break
    }

    # Step 1: Hydrate
    $ok = HydrateFile $srcPath
    if (-not $ok) { $failed += $f; continue }

    # Step 2: Upload via rclone
    Write-Host '  Uploading to Drive...'
    & $RC copyto $srcPath $dstPath --progress 2>&1 | Where-Object { $_ -match 'Transferred|ERROR' }
    $rc_exit = $LASTEXITCODE

    # Step 3: Verify
    $sizeOnDrive = & $RC size $dstPath 2>$null | Select-String 'Total size' | ForEach-Object { $_.Line }
    if ($rc_exit -eq 0 -and $sizeOnDrive) {
        Write-Host ('  Uploaded OK  ' + $sizeOnDrive)
    } else {
        Write-Warning ('  Upload FAILED (rclone exit=' + $rc_exit + ')')
        $failed += $f
        continue
    }

    # Step 4: Dehydrate source
    FreeSpace-File $srcPath
    $null = WaitForDehydration $srcPath

    $dc    = & $RC ls $DST --max-depth 1 2>$null | Measure-Object -Line | Select-Object -ExpandProperty Lines
    $freeA = [math]::Round((Get-PSDrive C).Free / 1GB, 1)
    Write-Host ('  Drive: ' + $dc + '/135  |  C: free: ' + $freeA + ' GB')
    Write-Host ''
}

# ── Summary ──────────────────────────────────────────────────────────────────
$fc = & $RC ls $DST --max-depth 1 2>$null | Measure-Object -Line | Select-Object -ExpandProperty Lines
Write-Host '=== DONE ==='
Write-Host ('Final Drive count : ' + $fc + ' of 135')
Write-Host ('Skipped           : ' + $skipped)
if ($failed.Count -gt 0) {
    Write-Warning ('Failed (' + $failed.Count + '):')
    $failed | ForEach-Object { Write-Warning ('  ' + $_) }
}

# watch_reasoner.ps1 - DeepSeek-Reasoner experiment progress monitor
# Usage (in E:\论文\sci_redo):
#   powershell -ExecutionPolicy Bypass -File watch_reasoner.ps1
#   powershell -ExecutionPolicy Bypass -File watch_reasoner.ps1 -Interval 3
param([int]$Interval = 5)

$res  = "E:\论文\sci_redo\results"
$f    = "$res\llm_deepseek_deepseek-reasoner_t0.3_s1_partial.csv"
$total = 120

while ($true) {
    Clear-Host
    $proc = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
            Where-Object { $_.CommandLine -match 'run_llm.py.*deepseek-reasoner' } | Select-Object -First 1
    $procRunning = $null -ne $proc
    $startTime = if ($procRunning) { $proc.CreationDate } else { $null }

    Write-Host "=== DeepSeek-Reasoner experiment progress ===" -ForegroundColor Cyan
    Write-Host ("now: " + (Get-Date).ToString("HH:mm:ss"))

    if ($procRunning) {
        $elapsed = (Get-Date) - $startTime
        Write-Host ("process: RUNNING | elapsed " + [math]::Round($elapsed.TotalMinutes, 1) + " min")
    } else {
        Write-Host "process: FINISHED/STOPPED" -ForegroundColor Yellow
    }

    $done = 0
    if (Test-Path $f) {
        $done = (Get-Content $f | Measure-Object -Line).Lines - 1
        if ($done -lt 0) { $done = 0 }
    }

    $shown = if ($done -gt 0) { $done } elseif ($procRunning) { 0 } else { $done }
    $barLen = 40
    $filled = [math]::Floor($shown / $total * $barLen)
    $bar = ("#" * $filled) + ("-" * ($barLen - $filled))
    Write-Host ("progress: [{0}] {1}/{2} matches" -f $bar, $shown, $total)

    if ($done -gt 0 -and $procRunning) {
        $elapsed = (Get-Date) - $startTime
        $avgPer = $elapsed.TotalSeconds / $done
        $remain = ($total - $done) * $avgPer
        Write-Host ("checkpoint shows {0} done (updates every 50 matches)" -f $done)
        Write-Host ("estimate: ~{0} s/match | ETA {1} min" -f [math]::Round($avgPer, 1), [math]::Round($remain / 60, 1))
    } elseif ($procRunning) {
        Write-Host "no checkpoint yet (first at 50 matches). If >30 min with no checkpoint, check network/API." -ForegroundColor DarkYellow
    }

    if (-not $procRunning) {
        Write-Host ""
        Write-Host "Experiment finished. See results:" -ForegroundColor Green
        Write-Host "  Get-Content results\llm_deepseek_deepseek-reasoner_t0.3.json"
        break
    }

    Start-Sleep -Seconds $Interval
}

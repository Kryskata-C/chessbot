# Build dist\ChessVision\ChessVision.exe and dist\ChessVision-<ver>-windows-x64.zip.
#
#   powershell -ExecutionPolicy Bypass -File packaging\build_win.ps1
#
# Needs: Python 3.11+ on PATH with `pip install -r requirements.txt pyinstaller`
# (plus requirements-win-ocr.txt if you want the opponent-rating OCR).
# Stockfish: set CV_STOCKFISH to a stockfish .exe, or it is downloaded from the
# official GitHub release. Since Stockfish 19 the Windows download is one "universal" zip that
# picks the fastest code path for the CPU at runtime.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$Version = if ($env:CV_VERSION) { $env:CV_VERSION } else { (Select-String -Path "version.py" -Pattern '__version__\s*=\s*"([^"]+)"').Matches[0].Groups[1].Value }
$Py = if ($env:CV_PYTHON) { $env:CV_PYTHON } else { "python" }
$SfBuild = if ($env:CV_STOCKFISH_BUILD) { $env:CV_STOCKFISH_BUILD } else { "universal" }

# 1. Stage Stockfish.
New-Item -ItemType Directory -Force -Path packaging\stage | Out-Null
$Stage = "packaging\stage\stockfish.exe"
if ($env:CV_STOCKFISH -and (Test-Path $env:CV_STOCKFISH)) {
    Copy-Item -Force $env:CV_STOCKFISH $Stage
    Write-Host "Stockfish: $($env:CV_STOCKFISH)"
} elseif (-not (Test-Path $Stage)) {
    $Asset = "stockfish-windows-x86-64-$SfBuild.zip"
    $Url = "https://github.com/official-stockfish/Stockfish/releases/latest/download/$Asset"
    $Tmp = Join-Path $env:TEMP "cv-stockfish"
    Remove-Item -Recurse -Force $Tmp -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $Tmp | Out-Null
    Write-Host "Downloading $Url"
    Invoke-WebRequest -Uri $Url -OutFile (Join-Path $Tmp $Asset)
    Expand-Archive -Path (Join-Path $Tmp $Asset) -DestinationPath $Tmp -Force
    $Exe = Get-ChildItem -Path $Tmp -Recurse -Filter "stockfish*.exe" | Select-Object -First 1
    if (-not $Exe) { throw "no stockfish .exe inside $Asset" }
    Copy-Item -Force $Exe.FullName $Stage
    Write-Host "Stockfish: $($Exe.Name)"
} else {
    Write-Host "Stockfish: $Stage (already staged)"
}
# Smoke-test the engine binary the way the app drives it (a subprocess
# with piped stdin); print what it said so a broken binary is diagnosable.
$Check = @'
import subprocess, sys
exe = sys.argv[1]
try:
    p = subprocess.run([exe], input=b"uci\nquit\n", capture_output=True, timeout=60)
except Exception as e:
    print("engine failed to start:", e); sys.exit(1)
out = p.stdout.decode(errors="replace"); err = p.stderr.decode(errors="replace")
print("exit", p.returncode, "| stdout tail:", out[-300:].strip(), "| stderr:", err.strip()[:300])
sys.exit(0 if "uciok" in out else 1)
'@
$CheckFile = Join-Path $env:TEMP "cv-uci-check.py"
Set-Content -Path $CheckFile -Value $Check
& $Py $CheckFile $Stage
if ($LASTEXITCODE -ne 0) { throw "staged Stockfish does not answer uci" }

# 2. Icon (only if missing; assets/ChessVision.ico is committed).
if (-not (Test-Path assets\ChessVision.ico)) {
    $env:QT_QPA_PLATFORM = "offscreen"
    & $Py packaging\make_icon.py
}

# 3. Build.
Remove-Item -Recurse -Force build\ChessVision, dist\ChessVision -ErrorAction SilentlyContinue
$env:CV_VERSION = $Version
& $Py -m PyInstaller --noconfirm --clean ChessVision-win.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }
if (-not (Test-Path dist\ChessVision\ChessVision.exe)) { throw "dist\ChessVision\ChessVision.exe missing" }
if (-not (Test-Path dist\ChessVision\_internal\bin\stockfish.exe) -and -not (Test-Path dist\ChessVision\bin\stockfish.exe)) {
    throw "Stockfish was not bundled"
}

# 4. Zip for distribution.
$Zip = "dist\ChessVision-$Version-windows-x64.zip"
Remove-Item -Force $Zip -ErrorAction SilentlyContinue
Compress-Archive -Path dist\ChessVision -DestinationPath $Zip
Write-Host "built: dist\ChessVision\ChessVision.exe"
Write-Host "zip:   $Zip ($([math]::Round((Get-Item $Zip).Length / 1MB)) MB)"

# 파일명 예: setup_deepface.ps1
# 위치: C:\Users\dhyang\deepface_age_server\server\setup_deepface.ps1

$ErrorActionPreference = "Stop"

$serverDir = "C:\Users\dhyang\deepface_age_server\server"
if (-not (Test-Path $serverDir)) {
  Write-Host "폴더가 없습니다: $serverDir"
  exit 1
}
Set-Location $serverDir

Write-Host ">>> Python 찾는 중..."

# 1) python 명령어 먼저 시도
$python = $null
try {
  $cmd = Get-Command python -ErrorAction Stop
  $python = $cmd.Path
} catch {
  $python = $null
}

# 2) 안 나오면 디스크에서 python.exe 검색
if (-not $python) {
  $searchRoots = @("$env:LOCALAPPDATA","C:\Program Files","C:\Program Files (x86)")
  $candidate = Get-ChildItem -Path $searchRoots -Recurse -Filter python.exe -ErrorAction SilentlyContinue |
    Select-Object -First 1 -ExpandProperty FullName

  if ($candidate) {
    $python = $candidate
  }
}

if (-not $python) {
  Write-Host ""
  Write-Host "### Python을 찾지 못했습니다. ###"
  Write-Host "1) python.org 또는 Microsoft Store에서 Python 3.11 이상 설치"
  Write-Host "2) 설치 후 이 스크립트를 다시 실행"
  exit 1
}

Write-Host ">>> 사용할 Python 경로:" $python

# venv 안의 python 경로
$venvPython = Join-Path $serverDir "venv\Scripts\python.exe"

# 3) venv 없으면 생성
if (-not (Test-Path $venvPython)) {
  Write-Host ">>> 가상환경(venv) 생성 중..."
  & $python -m venv venv
} else {
  Write-Host ">>> 기존 가상환경(venv) 사용"
}

# 4) pip 업그레이드 + requirements 설치
Write-Host ">>> pip 업그레이드..."
& $venvPython -m pip install --upgrade pip

Write-Host ">>> requirements 설치..."
if (-not (Test-Path (Join-Path $serverDir "requirements.txt"))) {
@"
deepface
flask
opencv-python
numpy
"@ | Out-File -FilePath (Join-Path $serverDir "requirements.txt") -Encoding UTF8
}

& $venvPython -m pip install -r requirements.txt

Write-Host ">>> DeepFace 서버 실행 시작 (포트 5000)..."
Write-Host "창을 끄지 말고 그대로 두면 됨."
Write-Host ""

& $venvPython app.py

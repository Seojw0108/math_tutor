#!/usr/bin/env bash
# 학교 GPU 서버(도커 컨테이너, 리눅스) 첫 설정 + 점검.
#
# 0) 먼저 "컨테이너를 다시 만들어도 남는 폴더"를 찾는다:  df -h
#    Filesystem 이 overlay 인 곳(/ 등)은 사라질 수 있다. /workspace, /data, /home/<이름> 처럼
#    따로 잡힌 경로가 유지되는 볼륨일 가능성이 높다. 모르면 관리자에게 묻는다.
# 1) 그 폴더 안에서 clone:   cd <유지 폴더> && git clone https://github.com/Seojw0108/math_tutor.git
# 2) 실행:
#    bash math_tutor/scripts/server_setup.sh             # 점검 + env.sh + venv + pip 설치 + 테스트
#    bash math_tutor/scripts/server_setup.sh --check     # 점검만 (아무것도 설치하지 않음)
#    bash math_tutor/scripts/server_setup.sh --download  # 위 + Qwen3-4B 다운로드 (약 8GB)
#    bash math_tutor/scripts/server_setup.sh --claude    # 위 + Claude Code 설치
#    옵션은 같이 써도 된다. --persist DIR 로 유지 폴더를 직접 정할 수 있다 (기본: 레포의 상위 폴더).
# 3) 다음 접속부터는:  source <유지 폴더>/env.sh
#
# 결과 요약은 <유지 폴더>/server_check.txt 에도 남는다.

set -u
export PYTHONIOENCODING=utf-8

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERSIST="$(dirname "$REPO")"
DO_INSTALL=1; DO_DOWNLOAD=0; DO_CLAUDE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --check) DO_INSTALL=0 ;;
    --download) DO_DOWNLOAD=1 ;;
    --claude) DO_CLAUDE=1 ;;
    --persist) shift; PERSIST="$(cd "$1" && pwd)" ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "모르는 옵션: $1"; exit 2 ;;
  esac
  shift
done

REPORT="$PERSIST/server_check.txt"
WARN=()
say()  { echo "$*" | tee -a "$REPORT"; }
warn() { WARN+=("$*"); say "  [주의] $*"; }
head_() { say ""; say "== $* =="; }
: > "$REPORT"

say "math_tutor 서버 점검  $(date '+%Y-%m-%d %H:%M')"
say "레포:      $REPO"
say "유지 폴더: $PERSIST"

# ---------------------------------------------------------------- 1. 저장 공간
head_ "1. 저장 공간 (파일이 남는 곳인가)"
fstype() { df -PT "$1" 2>/dev/null | awk 'NR==2{print $2}'; }
for d in "$PERSIST" "$HOME"; do
  t="$(fstype "$d")"
  say "  $d -> ${t:-?}"
  if [ "$t" = "overlay" ]; then
    warn "$d 는 overlay(컨테이너 층)다. 컨테이너를 다시 만들면 사라질 수 있다."
  fi
done
say "  -- df -h (tmpfs 제외)"
df -h 2>/dev/null | grep -vE '^(tmpfs|devtmpfs|shm|udev)' | sed 's/^/  /' | tee -a "$REPORT"
free_gb="$(df -Pk "$PERSIST" 2>/dev/null | awk 'NR==2{printf "%d", $4/1048576}')"
say "  유지 폴더 남은 공간: ${free_gb:-?}GB"
if [ -n "${free_gb:-}" ] && [ "$free_gb" -lt 30 ]; then
  warn "남은 공간 ${free_gb}GB. 모델 8GB + 병합 모델(모드당 8GB) + 체크포인트로 30GB 이상 권장."
fi
shm="$(df -Pk /dev/shm 2>/dev/null | awk 'NR==2{printf "%d", $2/1024}')"
say "  /dev/shm: ${shm:-?}MB"
if [ -n "${shm:-}" ] && [ "$shm" -lt 1024 ]; then
  say "  (작다. 학습 스크립트는 DataLoader 워커 0개라 보통 괜찮다. 'Bus error'가 나면 이것 때문.)"
fi

# ---------------------------------------------------------------- 2. GPU
head_ "2. GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv,noheader \
    | sed 's/^/  /' | tee -a "$REPORT"
  cuda_drv="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9.]+' | head -1)"
  say "  드라이버가 지원하는 ${cuda_drv:-CUDA Version: ?}"
else
  warn "nvidia-smi 없음. GPU가 컨테이너에 연결되지 않았을 수 있다 (docker --gpus)."
fi

# ---------------------------------------------------------------- 3. Python / torch
head_ "3. Python / torch"
PY=""
for c in python3 python; do  # 실제로 실행되는 첫 번째 것
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys' >/dev/null 2>&1; then PY="$(command -v "$c")"; break; fi
done
if [ -z "$PY" ]; then
  warn "python 없음. 설치가 필요하다."
else
  say "  $PY  ($("$PY" --version 2>&1))"
  if command -v conda >/dev/null 2>&1; then say "  conda 있음: $(conda --version 2>&1)"; fi
  TORCH_LINE="$("$PY" - <<'EOF' 2>/dev/null
import torch
print(f"torch {torch.__version__} | CUDA 빌드 {torch.version.cuda} | cuda 사용 {torch.cuda.is_available()}"
      + (f" | bf16 {torch.cuda.is_bf16_supported()}" if torch.cuda.is_available() else ""))
EOF
)"
  if [ -n "$TORCH_LINE" ]; then
    say "  $TORCH_LINE"
    HAS_TORCH=1
  else
    say "  torch 없음 (설치 단계에서 pip 기본 휠을 깐다)"
    HAS_TORCH=0
  fi
fi
[ "$(id -u)" = "0" ] && say "  root 권한 있음 (apt 사용 가능)" || say "  root 아님"

# ---------------------------------------------------------------- 4. 인터넷
head_ "4. 바깥 인터넷"
for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy; do
  [ -n "${!v:-}" ] && say "  프록시 설정 $v=${!v}"
done
probe() {  # HTTP 코드가 오면(404 포함) 연결됨, 000 이면 막힘
  if command -v curl >/dev/null 2>&1; then
    curl -s -o /dev/null -w '%{http_code}' --max-time 8 "$1" 2>/dev/null || true
  elif [ -n "$PY" ]; then
    "$PY" - "$1" <<'EOF' 2>/dev/null
import sys, urllib.request, urllib.error
try:
    print(urllib.request.urlopen(sys.argv[1], timeout=8).status)
except urllib.error.HTTPError as e:
    print(e.code)
except Exception:
    print("000")
EOF
  else
    echo "000"
  fi
}
NET_ANTHROPIC=0; NET_HF=0; NET_PYPI=0
for pair in "api.anthropic.com|Claude Code" "huggingface.co|모델 다운로드" "pypi.org|pip 설치" "github.com|git pull/push"; do
  host="${pair%%|*}"; why="${pair#*|}"
  code="$(probe "https://$host")"
  if [ -n "$code" ] && [ "$code" != "000" ]; then
    say "  OK    $host  ($why, HTTP $code)"
    case "$host" in api.anthropic.com) NET_ANTHROPIC=1;; huggingface.co) NET_HF=1;; pypi.org) NET_PYPI=1;; esac
  else
    say "  막힘  $host  ($why)"
  fi
done
[ $NET_ANTHROPIC = 0 ] && warn "api.anthropic.com 막힘 -> Claude Code는 학교 PC에 깔고 SSH로 이 서버에 연결해서 쓴다."
[ $NET_HF = 0 ] && warn "huggingface.co 막힘 -> 모델을 다른 곳에서 받아 scp 로 $REPO/models/Qwen3-4B 에 옮긴다."
[ $NET_PYPI = 0 ] && warn "pypi.org 막힘 -> pip 설치 불가. 관리자에게 미러/프록시를 묻는다."

# ---------------------------------------------------------------- 5. env.sh
ENV_SH="$PERSIST/env.sh"
if [ $DO_INSTALL = 1 ]; then
  head_ "5. env.sh 쓰기"
  cat > "$ENV_SH" <<EOF
# math_tutor 서버 환경. 접속할 때마다:  source $ENV_SH
export MT_PERSIST="$PERSIST"
export MT_REPO="$REPO"
export HF_HOME="\$MT_PERSIST/hf_cache"            # 허깅페이스 캐시도 유지 폴더에
export PIP_CACHE_DIR="\$MT_PERSIST/pip_cache"
export CLAUDE_CONFIG_DIR="\$MT_PERSIST/.claude"   # Claude Code 로그인·설정 유지
export PATH="\$HOME/.local/bin:\$PATH"
[ -f "\$MT_PERSIST/venv/bin/activate" ] && . "\$MT_PERSIST/venv/bin/activate"
cd "\$MT_REPO"
command -v claude >/dev/null 2>&1 || echo "(claude 없음. 컨테이너가 새로 만들어졌다면: curl -fsSL https://claude.ai/install.sh | bash)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null
EOF
  say "  $ENV_SH 작성"
  # shellcheck disable=SC1090
  . "$ENV_SH" >/dev/null
fi

# ---------------------------------------------------------------- 6. venv + pip
if [ $DO_INSTALL = 1 ] && [ -n "$PY" ]; then
  head_ "6. 가상환경 + 패키지"
  VENV="$PERSIST/venv"
  if [ ! -f "$VENV/bin/activate" ]; then
    flag=""; [ "${HAS_TORCH:-0}" = 1 ] && flag="--system-site-packages"   # 컨테이너의 CUDA torch 재사용
    "$PY" -m venv $flag "$VENV" 2>/dev/null \
      || { "$PY" -m pip install -q --user virtualenv && "$PY" -m virtualenv $flag "$VENV"; } \
      || { warn "venv 생성 실패. root면: apt-get update && apt-get install -y python3-venv"; }
  fi
  if [ -f "$VENV/bin/activate" ]; then
    . "$VENV/bin/activate"
    say "  venv: $VENV  (system torch 재사용: $([ "${HAS_TORCH:-0}" = 1 ] && echo 예 || echo 아니오))"
    if [ $NET_PYPI = 1 ]; then
      python -m pip install -q --upgrade pip
      if [ "${HAS_TORCH:-0}" = 0 ]; then
        say "  torch 설치 (pip 기본 CUDA 휠, 약 2~3GB). 드라이버가 너무 오래되면 cuda 사용 False가 나온다."
        python -m pip install -q torch || warn "torch 설치 실패"
      fi
      if python -m pip install -q -r "$REPO/requirements.txt"; then
        say "  requirements.txt 설치 OK"
      else
        warn "전체 설치 실패 -> 선택 패키지(bitsandbytes, gradio) 빼고 다시 시도"
        grep -vE '^(bitsandbytes|gradio)' "$REPO/requirements.txt" > "$PERSIST/.req_core.txt"
        python -m pip install -q -r "$PERSIST/.req_core.txt" && say "  필수 패키지 설치 OK" || warn "필수 패키지 설치 실패 (위 pip 오류 확인)"
      fi
    fi
    say "  -- check_env.py"
    (cd "$REPO" && python scripts/check_env.py 2>&1 | sed 's/^/  /' | tee -a "$REPORT")
    say "  -- pytest"
    (cd "$REPO" && python -m pytest -q 2>&1 | tail -3 | sed 's/^/  /' | tee -a "$REPORT")
  fi
fi

# ---------------------------------------------------------------- 7. 모델 다운로드
if [ $DO_DOWNLOAD = 1 ]; then
  head_ "7. Qwen3-4B 다운로드"
  if [ -f "$REPO/models/Qwen3-4B/config.json" ] && ls "$REPO"/models/Qwen3-4B/*.safetensors >/dev/null 2>&1; then
    say "  이미 있음: $REPO/models/Qwen3-4B"
  elif [ $NET_HF = 1 ]; then
    (cd "$REPO" && python scripts/download_model.py 2>&1 | tail -2 | sed 's/^/  /' | tee -a "$REPORT")
  else
    warn "huggingface.co 막힘 -> 다운로드 건너뜀"
  fi
fi

# ---------------------------------------------------------------- 8. Claude Code
if [ $DO_CLAUDE = 1 ]; then
  head_ "8. Claude Code"
  if command -v claude >/dev/null 2>&1; then
    say "  이미 있음: $(claude --version 2>/dev/null)"
  elif [ $NET_ANTHROPIC = 1 ]; then
    curl -fsSL https://claude.ai/install.sh | bash && say "  설치 OK. 'claude' 실행 후 나오는 URL을 학교 PC 브라우저에서 열어 로그인한다."
  else
    warn "api.anthropic.com 막힘 -> 이 서버에서는 Claude Code를 쓸 수 없다. 학교 PC + SSH 방식으로."
  fi
fi

# ---------------------------------------------------------------- 요약
head_ "요약"
if [ ${#WARN[@]} -eq 0 ]; then say "  주의할 점 없음"; else for w in "${WARN[@]}"; do say "  - $w"; done; fi
say ""
if [ $DO_INSTALL = 1 ]; then say "다음 접속부터:  source $ENV_SH"; else say "설치까지 하려면 --check 없이 다시 실행"; fi
say "다음 단계 (README ①②):"
say "  python scripts/check_masking.py --model models/Qwen3-4B"
say "  python scripts/base_measure.py --model models/Qwen3-4B --n_samples 3"
say "점검 결과 파일: $REPORT"

#!/usr/bin/env bash
# =============================================================================
# setup.sh — 재테크 AI Agent 개발 환경 설치 스크립트
#
# 실행 방법:
#   bash setup.sh
#
# 수행 작업:
#   1. Python 3.9+ 확인
#   2. 가상환경(.venv) 생성
#   3. 패키지 설치 (requirements.txt)
#   4. .env 파일 초기화 안내
#
# 참고:
#   - ChromaDB 벡터 DB와 KRX 주가 캐시는 앱 첫 실행 시 자동 생성됩니다.
#   - sentence-transformers 한국어 모델(~400MB)도 첫 실행 시 자동 다운로드됩니다.
# =============================================================================
set -euo pipefail

# ── 색상 출력 헬퍼 ──────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*" >&2; }
info() { echo -e "${CYAN}[-]${NC} $*"; }
step() { echo -e "\n${BOLD}$*${NC}"; }

# ── 스크립트 위치 기준으로 실행 ─────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BOLD}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " 재테크 AI Agent — 환경 설치"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${NC}"

# ── Step 1: Python 3.9+ 탐색 ────────────────────────────────────────────────
step "Step 1/4  Python 버전 확인"

PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3.9 python3; do
    if command -v "$cmd" &>/dev/null; then
        is_ok=$("$cmd" -c \
            "import sys; print('ok' if sys.version_info >= (3,9) else 'old')" 2>/dev/null || true)
        if [ "$is_ok" = "ok" ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    err "Python 3.9 이상을 찾을 수 없습니다."
    err "https://www.python.org/downloads/ 에서 설치 후 다시 실행해 주세요."
    exit 1
fi

PY_VERSION=$("$PYTHON" --version 2>&1)
ok "$PY_VERSION 확인 → $PYTHON"

# ── Step 2: 가상환경 생성 ────────────────────────────────────────────────────
step "Step 2/4  가상환경(.venv) 설정"

if [ -d ".venv" ]; then
    ok "가상환경 이미 존재 — 재사용합니다."
else
    info "가상환경 생성 중…"
    "$PYTHON" -m venv .venv
    ok "가상환경 생성 완료 (.venv/)"
fi

# 가상환경 활성화
# shellcheck disable=SC1091
source .venv/bin/activate

# pip 최신화 (경고 억제)
info "pip 업데이트 중…"
pip install --upgrade pip --quiet

ok "가상환경 활성화 완료"

# ── Step 3: 패키지 설치 ──────────────────────────────────────────────────────
step "Step 3/4  패키지 설치 (requirements.txt)"
warn "sentence-transformers, torch 등 대용량 패키지가 포함되어 시간이 걸릴 수 있습니다."
echo ""

pip install -r requirements.txt

ok "모든 패키지 설치 완료"

# ── Step 4: .env 파일 초기화 ────────────────────────────────────────────────
step "Step 4/4  .env 설정"

if [ -f ".env" ]; then
    ok ".env 파일 이미 존재 — 건너뜁니다."
else
    cp .env.example .env
    ok ".env 파일 생성 완료 (.env.example 복사)"
    echo ""
    warn "API 키를 입력해야 앱이 동작합니다. 지금 바로 .env 를 열어 입력하세요:"
    echo ""
    echo "     ANTHROPIC_API_KEY=sk-ant-api03-..."
    echo "     NEWS_API_KEY=xxxxxxxxxxxxxxxx"
    echo ""
    echo "  - Anthropic API 키: https://console.anthropic.com/"
    echo "  - NewsAPI 키:       https://newsapi.org/ (무료 플랜 가능)"
fi

# ── 완료 안내 ────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}${BOLD} 설치 완료!${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  다음 단계:"
echo ""
echo -e "  ${CYAN}1.${NC} .env 파일에 API 키 입력 (위 안내 참고)"
echo ""
echo -e "  ${CYAN}2.${NC} 앱 실행:"
echo -e "       ${BOLD}bash run.sh${NC}"
echo "     또는"
echo -e "       ${BOLD}source .venv/bin/activate && streamlit run app.py${NC}"
echo ""
echo "  ※ 첫 실행 시 한국어 임베딩 모델(~400MB)과 KRX 주가 데이터를"
echo "    자동으로 다운로드합니다. 인터넷 연결을 유지해 주세요."
echo ""

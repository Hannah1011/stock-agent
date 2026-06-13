#!/usr/bin/env bash
# =============================================================================
# run.sh — 재테크 AI Agent 실행 스크립트
#
# 실행 방법:
#   bash run.sh
#
# 참고:
#   setup.sh를 먼저 실행하고 .env에 API 키를 입력해야 합니다.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 사전 조건 확인 ───────────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "[✗] 가상환경이 없습니다. 먼저 setup.sh를 실행해 주세요:"
    echo "      bash setup.sh"
    exit 1
fi

if [ ! -f ".env" ]; then
    echo "[!] .env 파일이 없습니다. .env.example을 복사해 API 키를 입력해 주세요:"
    echo "      cp .env.example .env"
    exit 1
fi

# .env 로드 (주석 라인·공백 안전하게 처리)
# shellcheck disable=SC1091
source .env 2>/dev/null || true

# ── ANTHROPIC_API_KEY 확인 ───────────────────────────────────────────────────
if [ -z "${ANTHROPIC_API_KEY:-}" ] || [ "${ANTHROPIC_API_KEY}" = "your_anthropic_api_key_here" ]; then
    echo "[!] .env 파일의 ANTHROPIC_API_KEY가 설정되지 않았습니다."
    echo "    https://console.anthropic.com/ 에서 키를 발급받아 .env에 입력해 주세요."
    exit 1
fi

# ── EMBEDDING_PROVIDER=openai 일 때 OPENAI_API_KEY 확인 ─────────────────────
if [ "${EMBEDDING_PROVIDER:-local}" = "openai" ]; then
    if [ -z "${OPENAI_API_KEY:-}" ] || [ "${OPENAI_API_KEY}" = "your_openai_api_key_here" ]; then
        echo "[!] .env의 EMBEDDING_PROVIDER=openai 이지만 OPENAI_API_KEY가 설정되지 않았습니다."
        echo "    .env에 OPENAI_API_KEY를 추가하거나 EMBEDDING_PROVIDER=local 로 변경하세요."
        exit 1
    fi
fi

# ── 가상환경 활성화 ──────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source .venv/bin/activate

# ── 앱 실행 ─────────────────────────────────────────────────────────────────
echo "재테크 AI Agent 시작 중…"
echo "브라우저에서 http://localhost:8501 로 접속하세요."
echo "종료: Ctrl+C"
echo ""

streamlit run app.py \
    --server.port 8501 \
    --server.address localhost \
    --browser.gatherUsageStats false

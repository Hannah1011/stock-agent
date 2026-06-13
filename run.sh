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

# ── 고정 포트 확인 ───────────────────────────────────────────────────────────
PORT=8501

port_is_available() {
    python -c '
import socket
import sys

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
except OSError:
    sys.exit(1)
finally:
    sock.close()
' "$1"
}

streamlit_is_running() {
    python -c '
import sys
import urllib.request

try:
    with urllib.request.urlopen("http://localhost:8501/_stcore/health", timeout=1) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception:
    sys.exit(1)
'
}

if ! port_is_available "$PORT"; then
    if streamlit_is_running; then
        echo "재테크 AI Agent가 이미 실행 중입니다."
        echo "브라우저에서 http://localhost:8501 로 접속하세요."
        exit 0
    fi

    # 같은 프로젝트에서 실행한 Streamlit이 멈춘 경우에만 종료 후 재시작한다.
    EXISTING_PID=""
    if command -v lsof &>/dev/null; then
        EXISTING_PID="$(lsof -tiTCP:${PORT} -sTCP:LISTEN 2>/dev/null | head -n 1 || true)"
    fi

    if [ -n "$EXISTING_PID" ]; then
        EXISTING_CWD="$(lsof -a -p "$EXISTING_PID" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -n 1)"
        EXISTING_COMMAND="$(ps -p "$EXISTING_PID" -o command= 2>/dev/null || true)"

        if [ "$EXISTING_CWD" = "$SCRIPT_DIR" ] && [[ "$EXISTING_COMMAND" == *"streamlit run app.py"* ]]; then
            echo "응답하지 않는 기존 앱을 종료하고 8501 포트에서 다시 시작합니다."
            kill "$EXISTING_PID"
            for _ in {1..20}; do
                port_is_available "$PORT" && break
                sleep 0.25
            done

            if ! port_is_available "$PORT" && kill -0 "$EXISTING_PID" 2>/dev/null; then
                echo "기존 앱이 종료되지 않아 강제 종료합니다."
                kill -9 "$EXISTING_PID"
                for _ in {1..20}; do
                    port_is_available "$PORT" && break
                    sleep 0.25
                done
            fi
        fi
    fi

    if ! port_is_available "$PORT"; then
        echo "[!] 포트 8501을 다른 프로그램이 사용 중입니다."
        echo "    해당 프로그램을 종료한 뒤 다시 실행해 주세요."
        exit 1
    fi
fi

# ── 앱 실행 ─────────────────────────────────────────────────────────────────
echo "재테크 AI Agent 시작 중…"
echo "브라우저에서 http://localhost:${PORT} 로 접속하세요."
echo "종료: Ctrl+C"
echo ""

streamlit run app.py \
    --server.port "$PORT" \
    --server.address localhost \
    --browser.gatherUsageStats false

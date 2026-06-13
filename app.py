"""
app.py

재테크 AI 어시스턴트 — Streamlit UI

디자인: 글래스모피즘(Glassmorphism) + Toss/CLOVA X 디자인 참고
  - 딥 네이비 그라디언트 배경 위에 유리 카드 레이어
  - 빈 화면: CLOVA X 스타일 — 제목 + 2×2 제안 카드
  - 채팅 화면: 사용자 우측 파랑 말풍선 / AI 유리 카드
"""
from __future__ import annotations

import json
import logging
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from dotenv import load_dotenv

from graph.workflow import StockAgentWorkflow, WorkflowResult
from schemas.models import ClarificationCandidate, FinalReport
from utils.monitoring import MonitoringService

load_dotenv()
logging.basicConfig(level=logging.WARNING)


# ── 상수 ────────────────────────────────────────────────────────────────────────

PORTFOLIO_PATH = os.getenv(
    "PORTFOLIO_PATH", os.path.join(PROJECT_ROOT, "portfolio.json")
)

# 빈 화면 제안 카드 (아이콘 · 카테고리 · 실제 쿼리)
SUGGESTIONS = [
    ("종목 분석",  "삼성전자 오늘 왜 떨어졌어?"),
    ("시장 동향",  "오늘 증시 분위기는 어때?"),
    ("포트폴리오", "내 포트폴리오 위험도 어때?"),
    ("금융 용어",  "RSI랑 볼린저밴드가 뭐야?"),
]


# ── 페이지 설정 ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="재테크 AI",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── 글래스모피즘 CSS ────────────────────────────────────────────────────────────

_CSS = """
<style>
/* ── 폰트: 2종만 사용 ── */
/* 1. 본문 — 시스템 한국어 산세리프 */
* {
    font-family: 'Apple SD Gothic Neo', 'Noto Sans KR', -apple-system,
                 BlinkMacSystemFont, 'Malgun Gothic', sans-serif !important;
}
/* 2. 로그 전용 — 등폭 (본문과 크게 다르지 않은 크기) */
[data-testid="stExpander"] .stMarkdown p {
    font-family: 'SF Mono', 'Fira Code', 'Courier New', monospace !important;
    font-size: 12px !important;
    line-height: 1.85 !important;
    color: rgba(200, 220, 255, 0.68) !important;
}

/* ── 앱 전체 배경 ── */
.stApp {
    background: linear-gradient(160deg, #0D1B3E 0%, #1A3A6B 55%, #0D2B52 100%) !important;
    background-attachment: fixed !important;
}

/* ── 상단 헤더 완전 숨김 ── */
.stAppHeader, header[data-testid="stHeader"] {
    display: none !important;
}

/* ── 메인 컨테이너 여백 (헤더 숨겼으므로 상단 여백 줄임) ── */
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 6rem !important;
    max-width: 820px !important;
}

/* ── 하단 입력 영역 흰색 완전 제거 ── */
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] {
    background: #0D1B3E !important;
}

/* ── Streamlit 푸터 숨김 ── */
footer, [data-testid="stStatusWidget"] { display: none !important; }

/* ── 사이드바 — 유리 패널 ── */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.04) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.09) !important;
}
[data-testid="stSidebarContent"] p,
[data-testid="stSidebarContent"] span,
[data-testid="stSidebarContent"] label {
    color: rgba(255, 255, 255, 0.82) !important;
}

/* ── 전역 텍스트 색상 ── */
.stMarkdown p, .stMarkdown li,
.stMarkdown strong, .stMarkdown em {
    color: rgba(255, 255, 255, 0.88) !important;
}

/* ── 리포트 헤딩: H1~H3 모두 작게 (본문과 비슷한 크기, bold로만 구분) ── */
[data-testid="stChatMessage"] .stMarkdown h1,
[data-testid="stChatMessage"] .stMarkdown h2,
[data-testid="stChatMessage"] .stMarkdown h3 {
    font-size: 15px !important;
    font-weight: 700 !important;
    color: rgba(255, 255, 255, 0.85) !important;
    margin: 0.9rem 0 0.25rem 0 !important;
    border-bottom: none !important;
    letter-spacing: -0.01em !important;
}

/* ── AI 채팅 말풍선 — 유리 카드 ── */
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.06) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border: 1px solid rgba(255, 255, 255, 0.11) !important;
    border-radius: 16px !important;
    margin-bottom: 1rem !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.28) !important;
    overflow: visible !important;
}
[data-testid="stChatMessageContent"] {
    overflow: visible !important;
    word-break: keep-all !important;
}

/* ── 채팅 입력창 — 유리 필드 ── */
[data-testid="stChatInput"] > div {
    background: rgba(255, 255, 255, 0.07) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: rgba(255, 255, 255, 0.9) !important;
    caret-color: #3182F6 !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: rgba(255, 255, 255, 0.32) !important;
}

/* ── expander — 유리 패널 ── */
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
}
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span {
    color: rgba(255, 255, 255, 0.48) !important;
    font-size: 12px !important;
}

/* ── 버튼 ── */
.stButton > button {
    background: rgba(255, 255, 255, 0.07) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px !important;
    color: rgba(255, 255, 255, 0.78) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: all 0.18s ease !important;
    white-space: pre-line !important;
    text-align: left !important;
    line-height: 1.5 !important;
    padding: 0.55rem 1rem !important;
}
.stButton > button:hover {
    background: rgba(49, 130, 246, 0.28) !important;
    border-color: rgba(49, 130, 246, 0.5) !important;
    color: white !important;
    box-shadow: 0 0 18px rgba(49, 130, 246, 0.24) !important;
}
.stButton > button:active { transform: scale(0.97) !important; }

/* ── 구분선 ── */
hr { border-color: rgba(255, 255, 255, 0.07) !important; }

/* ── 인라인 코드 ── */
code {
    background: rgba(255, 255, 255, 0.1) !important;
    color: rgba(170, 210, 255, 0.9) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
    font-size: 11px !important;
}

/* ── 링크 ── */
a { color: #60A5FA !important; }
a:hover { color: #93C5FD !important; text-decoration: underline; }

/* ── 마크다운 표 ── */
table { border-collapse: collapse; width: 100%; }
thead th {
    background: rgba(255, 255, 255, 0.07) !important;
    color: rgba(255, 255, 255, 0.52) !important;
    font-size: 13px !important;
    padding: 8px 12px !important;
    border: none !important;
}
tbody td {
    border-bottom: 1px solid rgba(255, 255, 255, 0.06) !important;
    color: rgba(255, 255, 255, 0.78) !important;
    padding: 8px 12px !important;
    font-size: 14px !important;
}

/* ── 스크롤바 ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.16); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.3); }

/* ── spinner ── */
[data-testid="stSpinner"] p { color: rgba(255,255,255,0.55) !important; }
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)


# ── 워크플로우 싱글턴 ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_workflow() -> StockAgentWorkflow:
    return StockAgentWorkflow()


# ── 세션 초기화 ─────────────────────────────────────────────────────────────────

def _init_session() -> None:
    defaults: dict = {
        "messages":              [],
        "pending_clarification": None,
        "processing":            False,
        "_pending_query":        None,
        "_confirmed_ticker":     None,
        "_confirmed_name":       None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── HTML 헬퍼 ─────────────────────────────────────────────────────────────────

def _glass_div(
    inner: str,
    *,
    bg: str = "rgba(255,255,255,0.07)",
    border: str = "rgba(255,255,255,0.12)",
    radius: str = "14px",
    padding: str = "1rem 1.3rem",
    extra: str = "",
) -> str:
    return (
        f'<div style="background:{bg};backdrop-filter:blur(20px);'
        f'-webkit-backdrop-filter:blur(20px);border:1px solid {border};'
        f'border-radius:{radius};padding:{padding};margin-bottom:0.75rem;'
        f'box-shadow:0 6px 28px rgba(0,0,0,0.26);{extra}">'
        f'{inner}</div>'
    )


def _user_bubble(text: str) -> str:
    return (
        '<div style="display:flex;justify-content:flex-end;margin-bottom:1rem;">'
        '<div style="'
        'background:rgba(49,130,246,0.62);'
        'backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);'
        'border:1px solid rgba(49,130,246,0.44);'
        'border-radius:16px 16px 4px 16px;'
        'padding:0.75rem 1.15rem;max-width:72%;'
        'color:white;font-size:15px;line-height:1.55;'
        'box-shadow:0 4px 20px rgba(49,130,246,0.26);'
        f'">{text}</div></div>'
    )


def _label(text: str, color: str = "rgba(255,255,255,0.4)") -> str:
    return (
        f'<p style="margin:0 0 0.45rem 0;font-size:10px;font-weight:700;'
        f'letter-spacing:0.12em;text-transform:uppercase;color:{color};">'
        f'{text}</p>'
    )


# ── 리포트 렌더링 ─────────────────────────────────────────────────────────────

def _render_report(report: FinalReport) -> None:
    # 한 줄 요약
    st.markdown(
        _glass_div(
            _label("한 줄 요약", "rgba(100,170,255,0.75)")
            + f'<p style="margin:0;font-size:17px;font-weight:600;'
            f'color:white;line-height:1.55;">{report.one_line_summary}</p>',
            bg="rgba(49,130,246,0.12)",
            border="rgba(49,130,246,0.28)",
        ),
        unsafe_allow_html=True,
    )

    # 주의 알림
    if report.risk_alerts:
        alerts_html = "".join(
            f'<p style="margin:0.25rem 0;font-size:14px;'
            f'color:rgba(255,205,80,0.92);">{a}</p>'
            for a in report.risk_alerts
        )
        st.markdown(
            _glass_div(
                _label("주의 알림", "rgba(255,170,50,0.75)") + alerts_html,
                bg="rgba(255,145,30,0.09)",
                border="rgba(255,145,30,0.24)",
            ),
            unsafe_allow_html=True,
        )

    # 풀 마크다운 리포트
    if report.full_report_md:
        st.markdown("---")
        st.markdown(report.full_report_md)

    # 금융 용어 사전
    if report.terms_glossary:
        terms_html = "".join(
            f'<div style="margin-bottom:0.5rem;">'
            f'<span style="font-weight:700;color:rgba(130,200,255,0.92);font-size:14px;">'
            f'{t.term}</span>'
            f'<span style="color:rgba(255,255,255,0.62);font-size:14px;"> — {t.explanation}</span>'
            f'</div>'
            for t in report.terms_glossary
        )
        st.markdown(
            _glass_div(_label("오늘의 금융 용어") + terms_html),
            unsafe_allow_html=True,
        )

    # 면책 고지
    st.markdown(
        f'<p style="font-size:11px;color:rgba(255,255,255,0.25);margin-top:0.5rem;">'
        f'{report.disclaimer}</p>',
        unsafe_allow_html=True,
    )


# ── 메시지 렌더링 ─────────────────────────────────────────────────────────────

def _render_messages() -> None:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(_user_bubble(msg["content"]), unsafe_allow_html=True)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                _render_assistant_body(msg)


def _render_assistant_body(msg: dict) -> None:
    """assistant 메시지 1건의 본문을 렌더링한다."""
    if msg.get("log_text"):
        label = f"에이전트 사고 과정  ·  {msg.get('log_summary', '')}"
        with st.expander(label, expanded=False):
            st.markdown(msg["log_text"])

    msg_type = msg.get("type", "text")

    if msg_type == "report":
        _render_report(msg["report"])

    elif msg_type == "clarification":
        st.markdown(
            _glass_div(
                _label("종목 확인")
                + f'<p style="margin:0;font-size:15px;color:rgba(255,255,255,0.85);">'
                f'{msg.get("clarification_message", "어떤 종목을 말씀하시나요?")}</p>',
            ),
            unsafe_allow_html=True,
        )
        for cand in msg.get("candidates", []):
            st.markdown(
                f'`{cand.name}` ({cand.ticker}) — {cand.market} · {cand.sector}'
            )

    elif msg_type == "rejection":
        st.markdown(
            _glass_div(
                _label("안내", "rgba(255,120,120,0.75)")
                + f'<p style="margin:0;font-size:15px;color:rgba(255,255,255,0.82);">'
                f'{msg["content"]}</p>'
                + '<p style="margin:0.65rem 0 0 0;font-size:13px;'
                f'color:rgba(255,255,255,0.38);">'
                '예시: "삼성전자 분석해줘" &nbsp;·&nbsp; "RSI가 뭐야?" &nbsp;·&nbsp; "내 포트폴리오 분석"'
                '</p>',
                bg="rgba(240,68,82,0.09)",
                border="rgba(240,68,82,0.22)",
            ),
            unsafe_allow_html=True,
        )

    else:
        st.markdown(msg.get("content", ""))


# ── 빈 화면 — CLOVA X 스타일 제안 카드 ────────────────────────────────────────

def _render_empty_state() -> None:
    """
    대화가 없을 때 타이틀 + 1×4 제안 카드를 표시한다.
    제안 카드 클릭 시 해당 쿼리를 자동 제출한다.
    """
    st.markdown(
        '<div style="margin-bottom:2rem;">'
        '<p style="margin:0;font-size:24px;font-weight:800;color:white;'
        'letter-spacing:-0.025em;line-height:1.2;">재테크 AI 어시스턴트</p>'
        '<p style="margin:0.45rem 0 0 0;font-size:14px;color:rgba(255,255,255,0.44);">'
        '주식 · 시장 동향 · 금융 용어 · 포트폴리오를 AI가 쉽게 설명해 드려요</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    for category, query in SUGGESTIONS:
        btn_label = f"{category}\n{query}"
        if st.button(btn_label, key=f"sug_{query[:8]}", use_container_width=True):
            _submit(query)


# ── 종목 후보 선택 UI ─────────────────────────────────────────────────────────

def _render_clarification_ui() -> None:
    """pending_clarification이 있으면 종목 선택 카드를 보여준다."""
    pending = st.session_state.pending_clarification
    if not pending:
        return

    candidates: list[ClarificationCandidate] = pending["candidates"]
    original_query: str                       = pending["original_query"]

    st.markdown(
        _glass_div(
            _label("종목 확인 필요")
            + '<p style="margin:0;font-size:15px;font-weight:600;color:white;">'
            '어떤 종목을 말씀하시는 건가요?</p>',
        ),
        unsafe_allow_html=True,
    )

    cols = st.columns(min(len(candidates), 3), gap="small")
    for i, cand in enumerate(candidates):
        with cols[i % 3]:
            btn_label = (
                f"{cand.name}\n{cand.ticker}\n{cand.market} · {cand.sector}"
            )
            if st.button(btn_label, key=f"cand_{cand.ticker}_{i}", use_container_width=True):
                st.session_state.pending_clarification = None
                st.session_state.processing            = True
                st.session_state["_pending_query"]     = original_query
                st.session_state["_confirmed_ticker"]  = cand.ticker
                st.session_state["_confirmed_name"]    = cand.name
                st.rerun()


# ── 사이드바 ──────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<p style="font-size:19px;font-weight:700;color:white;'
            'margin-bottom:0.1rem;letter-spacing:-0.01em;">내 포트폴리오</p>'
            '<p style="font-size:12px;color:rgba(255,255,255,0.36);'
            'margin-bottom:1.1rem;">portfolio.json 기준</p>',
            unsafe_allow_html=True,
        )

        holdings = _load_portfolio()

        if holdings is None:
            st.markdown(
                '<p style="font-size:13px;color:rgba(255,255,255,0.32);">'
                'portfolio.json 없음<br>'
                'PORTFOLIO_PATH 환경변수로 경로를 지정하세요.</p>',
                unsafe_allow_html=True,
            )
        elif not holdings:
            st.markdown(
                '<p style="font-size:13px;color:rgba(255,255,255,0.32);">종목 없음</p>',
                unsafe_allow_html=True,
            )
        else:
            for h in holdings:
                name   = h.get("name", h.get("ticker", "?"))
                shares = h.get("shares", 0)
                avg    = h.get("avg_price", 0)
                st.markdown(
                    _glass_div(
                        f'<span style="font-size:14px;font-weight:600;'
                        f'color:rgba(255,255,255,0.88);">{name}</span>'
                        f'<span style="font-size:12px;color:rgba(255,255,255,0.34);'
                        f'margin-left:6px;">{shares}주</span><br>'
                        f'<span style="font-size:12px;color:rgba(255,255,255,0.48);">'
                        f'평균 {avg:,.0f}원</span>',
                        bg="rgba(255,255,255,0.05)",
                        border="rgba(255,255,255,0.08)",
                        radius="10px",
                        padding="0.5rem 0.85rem",
                        extra="margin-bottom:0.35rem;",
                    ),
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            if st.button("포트폴리오 전체 분석", use_container_width=True):
                _submit("내 포트폴리오 전체 분석해줘")

        st.markdown("---")
        if st.button("대화 초기화", use_container_width=True):
            st.session_state.messages              = []
            st.session_state.pending_clarification = None
            st.rerun()


def _load_portfolio() -> list[dict] | None:
    if not os.path.exists(PORTFOLIO_PATH):
        return None
    try:
        with open(PORTFOLIO_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, list) else raw.get("holdings", [])
    except Exception:
        return []


# ── 쿼리 제출 ────────────────────────────────────────────────────────────────

def _submit(query: str) -> None:
    if not query.strip() or st.session_state.get("processing"):
        return
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.processing        = True
    st.session_state["_pending_query"] = query
    st.rerun()


# ── 워크플로우 실행 ────────────────────────────────────────────────────────────

def _handle_processing() -> bool:
    """
    session_state.processing이 True이면 워크플로우를 실행하고 결과를 렌더링한다.

    Returns:
        True: 처리를 시작했음 (내부에서 st.rerun() 호출 → 이후 코드 실행 안 됨)
        False: 처리할 작업 없음
    """
    if not st.session_state.get("processing"):
        return False

    query    = st.session_state.pop("_pending_query", "")
    ticker   = st.session_state.pop("_confirmed_ticker", None)
    name_val = st.session_state.pop("_confirmed_name", None)

    if not query:
        st.session_state.processing = False
        return False

    # 스피너 이전에 기존 히스토리(사용자 메시지 포함) 먼저 표시
    _render_messages()

    workflow = _get_workflow()
    monitor  = MonitoringService()

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("에이전트가 분석 중입니다..."):
            try:
                if ticker:
                    result: WorkflowResult = workflow.run_with_confirmed_ticker(
                        query, ticker, name_val, on_log=monitor.add
                    )
                else:
                    result = workflow.run(query, on_log=monitor.add)
            except Exception as e:
                st.error(f"예상치 못한 오류: {e}")
                st.session_state.processing = False
                return True

        log_text = monitor.format_for_expander()
        summary  = monitor.format_summary()
        result_msg = _build_result_msg(result, log_text, summary, query)

        # 결과를 즉시 렌더링
        _render_assistant_body(result_msg)

    st.session_state.messages.append(result_msg)
    st.session_state.processing = False
    st.rerun()
    return True  # st.rerun()이 중단시키므로 실제 도달하지 않음


def _build_result_msg(
    result: WorkflowResult,
    log_text: str,
    summary: str,
    query: str,
) -> dict:
    base = {
        "role":        "assistant",
        "log_text":    log_text,
        "log_summary": summary,
    }
    if result.is_success:
        return {**base, "type": "report", "report": result.final_report}
    if result.needs_clarification:
        plan = result.plan
        st.session_state.pending_clarification = {
            "candidates":     plan.clarification_candidates,
            "original_query": query,
        }
        return {
            **base,
            "type":                  "clarification",
            "candidates":            plan.clarification_candidates,
            "clarification_message": plan.clarification_message or "어떤 종목을 말씀하시나요?",
            "original_query":        query,
        }
    return {
        **base,
        "type":    "rejection",
        "content": result.plan.rejection_message or "답변하기 어려운 질문이에요.",
    }


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    _init_session()
    _render_sidebar()

    # 처리 중: 히스토리 + 스피너 + 결과 렌더링 후 st.rerun() 호출
    if _handle_processing():
        return  # st.rerun()이 이미 호출되었으므로 여기에 도달하지 않음

    if not st.session_state.messages:
        # 빈 화면: CLOVA X 스타일 제안 카드
        _render_empty_state()
    else:
        # 채팅 화면: 메시지 히스토리 + 종목 선택 UI(pending 시)
        _render_messages()
        _render_clarification_ui()

    # 채팅 입력 (항상 하단 고정)
    query = st.chat_input(
        "궁금한 것을 자유롭게 물어보세요... (예: 삼성전자 분석해줘, RSI가 뭐야?)",
        disabled=st.session_state.get("processing", False),
    )
    if query:
        st.session_state.pending_clarification = None
        _submit(query)


if __name__ == "__main__":
    main()

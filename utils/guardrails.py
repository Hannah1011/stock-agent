import re

ALLOWED_TOPICS = [
    "주식", "채권", "ETF", "펀드", "코스피", "코스닥", "환율",
    "금리", "인플레이션", "포트폴리오", "종목", "배당", "시가총액",
    "거래량", "이평선", "RSI", "재테크", "투자", "증시", "경제지표",
    "볼린저", "MACD", "PER", "PBR", "EPS", "섹터", "시장", "선물",
    "옵션", "공매도", "상장", "상폐", "분기", "실적", "반도체",
    "2차전지", "바이오", "자동차", "철강", "화학", "에너지", "IT",
]

BLOCKED_PHRASES = [
    "매수 추천", "지금 사세요", "반드시 오릅니다", "확실한 수익",
    "무조건 매수", "손실 없음", "보장된 수익", "투자하세요",
    "사면 됩니다", "팔면 됩니다", "지금 팔아",
]

DISCLAIMER = "본 정보는 투자 판단을 위한 참고 자료이며, 투자 권유가 아닙니다."

FINANCE_KEYWORDS = re.compile(
    "|".join(re.escape(kw) for kw in ALLOWED_TOPICS),
    re.IGNORECASE,
)


def is_in_scope(user_input: str) -> bool:
    """재테크·투자 관련 질문인지 확인한다."""
    return bool(FINANCE_KEYWORDS.search(user_input))


def filter_investment_advice(text: str) -> str:
    """투자 권유성 문구를 중립 표현으로 교체한다."""
    for phrase in BLOCKED_PHRASES:
        text = text.replace(phrase, "[참고 정보]")
    return text


def ensure_disclaimer(text: str) -> str:
    """리포트 말미에 면책 문구가 없으면 추가한다."""
    if DISCLAIMER not in text:
        text = text.rstrip() + f"\n\n---\n{DISCLAIMER}"
    return text


def validate_scope(user_input: str) -> tuple[bool, str]:
    """
    범위 체크 결과와 거절 메시지를 함께 반환한다.
    Returns (is_allowed, rejection_message).
    """
    if is_in_scope(user_input):
        return True, ""
    rejection = (
        "죄송합니다. 저는 주식·재테크 관련 질문만 답변할 수 있습니다. "
        "종목 분석, 포트폴리오, 시장 동향 등에 대해 질문해 주세요."
    )
    return False, rejection

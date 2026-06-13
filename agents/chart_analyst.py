"""
agents/chart_analyst.py

차트 분석 에이전트.

실행 흐름:
  1. ExecutionPlan에서 ticker 확인 (없으면 None 반환, 예외 없음)
  2. tools/stock_api.py의 get_technical_indicators()로 기술지표 수집
     - 6개월 OHLCV 기반: MA5/20/60, RSI(14일), 볼린저밴드, 거래량 신호
  3. 지표로 ChartData 구성 (추세 방향 자동 판별)
  4. Claude Haiku로 초보 투자자 친화적인 한국어 해설 생성
     - 모든 전문 용어에 쉬운 설명 병기
     - "매수/매도" 등 직접 투자 권유 표현 금지
  5. ChartAnalystOutput 반환
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import anthropic
from anthropic import APIConnectionError, APIError, APITimeoutError, RateLimitError
from dotenv import load_dotenv

from schemas.models import (
    ChartAnalystOutput,
    ChartData,
    ExecutionPlan,
    ModelTier,
)
from tools.stock_api import get_technical_indicators

load_dotenv()
logger = logging.getLogger(__name__)

_MODEL = ModelTier.LIGHT.value   # claude-haiku: 해설 텍스트 생성


class ChartAnalystError(Exception):
    """차트 분석 중 복구 불가능한 오류."""


class ChartAnalystAgent:
    """
    기술적 지표를 수집하고 초보자 언어로 해설한 ChartAnalystOutput을 반환한다.
    ticker가 없거나 데이터 부족 시 None을 반환하며 예외를 발생시키지 않는다.
    """

    def __init__(self) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ChartAnalystError("ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")
        self._client = anthropic.Anthropic(api_key=api_key)

    # ── 공개 메서드 ──────────────────────────────────────────────────────────

    def run(self, plan: ExecutionPlan) -> Optional[ChartAnalystOutput]:
        """
        기술지표를 분석하고 초보자 친화적 해설과 함께 ChartAnalystOutput을 반환한다.

        Returns:
            ChartAnalystOutput — 분석 성공 시
            None              — ticker 없음 또는 6개월치 데이터 부족 시
        """
        ticker = plan.orchestrator_output.ticker
        if not ticker:
            logger.info("[ChartAnalyst] ticker 없음. 분석 건너뜀.")
            return None

        indicators = self._fetch_indicators(ticker)
        if indicators is None:
            return None

        chart_data  = self._build_chart_data(ticker, indicators)
        explanation = self._explain_in_plain_korean(chart_data, indicators)

        logger.info("[ChartAnalyst] %s 분석 완료.", ticker)
        return ChartAnalystOutput(chart_data=chart_data, plain_explanation=explanation)

    # ── 내부: 지표 수집 ──────────────────────────────────────────────────────

    def _fetch_indicators(self, ticker: str) -> Optional[dict]:
        """
        yfinance에서 기술적 지표를 가져온다.
        데이터 부족(20거래일 미만) 또는 예외 발생 시 None을 반환한다.
        """
        try:
            indicators = get_technical_indicators(ticker)
        except Exception as e:
            logger.error("[ChartAnalyst] 지표 조회 예외 (%s): %s", ticker, e)
            return None

        if indicators is None:
            logger.warning("[ChartAnalyst] %s: 기술지표 데이터 부족.", ticker)
        return indicators

    # ── 내부: ChartData 구성 ─────────────────────────────────────────────────

    @staticmethod
    def _build_chart_data(ticker: str, ind: dict) -> ChartData:
        """기술지표 딕셔너리를 ChartData Pydantic 모델로 변환한다."""
        ma5           = ind.get("ma5")
        ma20          = ind.get("ma20")
        current_price = ind.get("current_price", 0)

        # 이동평균 배열로 단기 추세 판별
        if ma5 and ma20 and current_price > ma5 > ma20:
            trend_summary = "단기·중기 이동평균 위 — 상승 추세"
        elif ma5 and ma20 and current_price < ma5 < ma20:
            trend_summary = "단기·중기 이동평균 아래 — 하락 추세"
        elif ma5 and ma20 and ma5 > ma20:
            trend_summary = "단기 이동평균이 중기 위 — 상승 전환 신호"
        elif ma5 and ma20 and ma5 < ma20:
            trend_summary = "단기 이동평균이 중기 아래 — 하락 전환 신호"
        else:
            trend_summary = "이동평균 혼조 — 방향성 불확실"

        return ChartData(
            ticker=ticker,
            current_price=current_price,
            change_pct=ind.get("change_pct", 0.0),
            rsi=ind.get("rsi"),
            ma5=ma5,
            ma20=ma20,
            ma60=ind.get("ma60"),
            volume_signal=ind.get("volume_signal", "NORMAL"),
            trend_summary=trend_summary,
        )

    # ── 내부: 초보자 해설 생성 ───────────────────────────────────────────────

    def _explain_in_plain_korean(self, chart_data: ChartData, ind: dict) -> str:
        """
        Claude Haiku로 기술적 지표를 초보자 언어로 해설한다.
        API 호출 실패 시 지표 수치 기반 기본 해설을 반환한다.
        """
        ma5_str   = f"{chart_data.ma5:,.0f}원" if chart_data.ma5 else "데이터 없음"
        ma20_str  = f"{chart_data.ma20:,.0f}원" if chart_data.ma20 else "데이터 없음"
        ma60_str  = f"{chart_data.ma60:,.0f}원" if chart_data.ma60 else "데이터 없음"
        rsi_str   = f"{chart_data.rsi:.1f}" if chart_data.rsi is not None else "데이터 없음"
        upper_str = f"{ind.get('upper_band', 0):,.0f}원" if ind.get("upper_band") else "데이터 없음"
        lower_str = f"{ind.get('lower_band', 0):,.0f}원" if ind.get("lower_band") else "데이터 없음"

        vol_desc = {
            "VERY_HIGH": "최근 5일 거래량이 평소보다 2배 이상 — 투자자 관심 급증",
            "HIGH":      "최근 5일 거래량이 평소보다 30% 이상 — 주목할 수준",
            "NORMAL":    "거래량이 평소와 비슷한 안정적 수준",
        }.get(chart_data.volume_signal, "데이터 없음")

        prompt = (
            "당신은 주식 투자 완전 초보자에게 차트 지표를 쉽게 설명하는 전문가입니다.\n\n"
            f"[분석 종목: {chart_data.ticker}]\n"
            f"현재가: {chart_data.current_price:,.0f}원 (전일 대비 {chart_data.change_pct:+.2f}%)\n"
            f"현재 추세 요약: {chart_data.trend_summary}\n\n"
            "[기술적 지표 수치]\n"
            f"이동평균(MA):\n"
            f"  5일 평균가(MA5, 최근 1주): {ma5_str}\n"
            f"  20일 평균가(MA20, 최근 1달): {ma20_str}\n"
            f"  60일 평균가(MA60, 최근 3달): {ma60_str}\n"
            f"RSI(상대강도지수): {rsi_str}  ← 70 이상=과매수, 30 이하=과매도, 50 근처=중립\n"
            f"볼린저밴드: 상단 {upper_str} / 하단 {lower_str}\n"
            f"거래량 신호: {vol_desc}\n\n"
            "[해설 작성 규칙]\n"
            "1. 이동평균: 현재 주가가 5일·20일 평균 대비 어느 위치인지 직관적으로 설명\n"
            "   (예: '지난 한 달 평균보다 높은 곳에 있다'는 식으로)\n"
            "2. RSI: 수치의 의미를 비유로 설명 (예: '주가의 힘이 세다/약하다')\n"
            "3. 볼린저밴드: 현재 가격이 '정상 범위' 내에 있는지, 벗어났는지 설명\n"
            "4. 거래량: 투자자들의 관심도와 연결해 설명\n"
            "5. 종합: 위 내용을 1~2문장으로 요약\n\n"
            "[절대 금지]\n"
            "- '매수하세요', '매도하세요', '지금 사세요' 등 직접적인 매매 권유\n"
            "- '반드시 오릅니다', '손실 없음' 등 수익 보장 표현\n"
            "- 특정 목표 주가 제시\n\n"
            "전문 용어는 반드시 괄호로 설명 병기하세요. "
            "응답은 250단어 이내, 마크다운 없이 자연스러운 한국어로 작성하세요."
        )

        try:
            response = self._client.messages.create(
                model=_MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            explanation = response.content[0].text.strip()
            return explanation if explanation else self._default_explanation(chart_data)
        except (RateLimitError, APITimeoutError, APIConnectionError, APIError) as e:
            logger.warning("[ChartAnalyst] 해설 생성 실패: %s", e)
            return self._default_explanation(chart_data)

    @staticmethod
    def _default_explanation(chart_data: ChartData) -> str:
        """Claude 호출 실패 시 수치 기반 기본 해설을 반환한다."""
        direction = "상승" if chart_data.change_pct >= 0 else "하락"

        rsi_comment = ""
        if chart_data.rsi is not None:
            if chart_data.rsi >= 70:
                rsi_comment = f" RSI({chart_data.rsi:.1f})가 70 이상으로 단기 과매수 구간입니다."
            elif chart_data.rsi <= 30:
                rsi_comment = f" RSI({chart_data.rsi:.1f})가 30 이하로 단기 과매도 구간입니다."
            else:
                rsi_comment = f" RSI({chart_data.rsi:.1f})는 중립 구간에 위치합니다."

        return (
            f"현재가 {chart_data.current_price:,.0f}원으로 전일 대비 "
            f"{abs(chart_data.change_pct):.2f}% {direction}했습니다."
            f"{rsi_comment} {chart_data.trend_summary}."
        )

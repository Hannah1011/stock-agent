"""
agents/term_explainer.py

용어 설명 에이전트.

실행 흐름:
  [TERM_QUERY] 사용자가 직접 용어를 묻는 경우:
    1. Claude Haiku로 질문에서 경제·금융 용어 추출 (최대 3개)
    2. 각 용어에 대해 get_term_explanation_with_fallback() 호출
       → RAG Hybrid Search 우선 → Wikipedia+Claude Haiku 폴백

  [STOCK_QUERY / MARKET_TREND] 뉴스에서 용어를 추출하는 경우:
    1. 뉴스 기사 제목+요약 텍스트를 extract_and_explain_terms()로 처리
       → BM25 + Dense 하이브리드 검색으로 금융 용어 자동 감지

  결과: TermExplainerOutput (용어별 TermAnnotation 목록)
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

from rag.retriever import extract_and_explain_terms, get_term_explanation_with_fallback
from schemas.models import (
    ExecutionPlan,
    IntentType,
    ModelTier,
    NewsCollectorOutput,
    TermAnnotation,
    TermExplainerOutput,
)

load_dotenv()
logger = logging.getLogger(__name__)

_MODEL     = ModelTier.LIGHT.value   # claude-haiku: 용어 추출 전용
_MAX_TERMS = 3   # TERM_QUERY에서 설명할 최대 용어 수
_MAX_FROM_NEWS = 5  # 뉴스 텍스트에서 추출할 최대 용어 수

# ─── 용어 추출 도구 스키마 ────────────────────────────────────────────────────
_EXTRACT_TERMS_TOOL: dict = {
    "name": "extract_financial_terms",
    "description": "사용자 질문에서 주식 투자 초보자에게 설명이 필요한 경제·금융 용어를 추출합니다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "terms": {
                "type": "array",
                "description": "추출된 경제·금융 용어 목록. 최대 3개.",
                "items": {"type": "string"},
                "maxItems": 3,
            }
        },
        "required": ["terms"],
    },
}


class TermExplainerError(Exception):
    """용어 설명 중 복구 불가능한 오류."""


class TermExplainerAgent:
    """
    경제·금융 용어를 추출하고 RAG(+폴백 크롤링)로 초보자용 설명을 생성한다.
    외부에서는 run() 메서드만 사용한다.
    """

    def __init__(self) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise TermExplainerError("ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")
        self._client = anthropic.Anthropic(api_key=api_key)

    # ── 공개 메서드 ──────────────────────────────────────────────────────────

    def run(
        self,
        plan: ExecutionPlan,
        news_output: Optional[NewsCollectorOutput] = None,
    ) -> TermExplainerOutput:
        """
        의도에 따라 용어를 추출·설명하고 TermExplainerOutput을 반환한다.

        Args:
            plan:        OrchestratorAgent가 반환한 실행 계획
            news_output: STOCK_QUERY/MARKET_TREND의 경우 뉴스 수집 결과
                         (용어 추출 소스로 활용)

        Returns:
            TermExplainerOutput — terms_explained=[]이면 설명할 용어가 없는 것
        """
        intent        = plan.orchestrator_output.intent
        query_summary = plan.orchestrator_output.query_summary

        if intent == IntentType.TERM_QUERY:
            return self._run_term_query(query_summary)

        # STOCK_QUERY / MARKET_TREND: 뉴스 텍스트에서 용어 추출
        if news_output and news_output.news_items:
            return self._run_from_news(query_summary, news_output)

        # 뉴스 없으면 질문 자체를 소스로 사용
        logger.info("[TermExplainer] 뉴스 없음. 질문 텍스트에서 용어 추출 시도.")
        return self._run_term_query(query_summary)

    # ── TERM_QUERY 처리 ──────────────────────────────────────────────────────

    def _run_term_query(self, query_summary: str) -> TermExplainerOutput:
        """사용자가 직접 묻는 용어를 추출하고 RAG로 설명한다."""
        terms = self._extract_terms_from_query(query_summary)
        logger.info("[TermExplainer] 추출된 용어: %s", terms)

        annotations = self._explain_terms(terms)
        return TermExplainerOutput(
            original_text=query_summary,
            annotated_text=query_summary,
            terms_explained=annotations,
        )

    def _extract_terms_from_query(self, query: str) -> list[str]:
        """
        Claude Haiku로 사용자 질문에서 경제·금융 용어를 추출한다.
        추출 실패 또는 API 오류 시 빈 리스트를 반환한다.
        """
        system_prompt = (
            "당신은 경제·금융 용어 추출 전문가입니다.\n\n"
            "[추출 규칙]\n"
            "- 주식 투자 초보자에게 설명이 필요한 경제·금융 전문 용어만 추출\n"
            "- 용어는 원형 그대로 추출 (예: 'RSI', 'PER', '코스피', '볼린저밴드', '양적완화')\n"
            "- 최대 3개. 가장 핵심적인 용어 우선\n"
            "- 일반 동사·명사는 제외 (예: '왜', '어때', '오늘', '분석', '설명')\n"
            "- 질문에 용어가 없으면 terms=[] 반환\n\n"
            "[예시]\n"
            "질문: 'RSI랑 볼린저밴드가 뭐야?' → terms: ['RSI', '볼린저밴드']\n"
            "질문: '코스피랑 코스닥 차이가 뭐야?' → terms: ['코스피', '코스닥']\n"
            "질문: '오늘 증시 어때?' → terms: []"
        )

        try:
            response = self._client.messages.create(
                model=_MODEL,
                max_tokens=256,
                system=system_prompt,
                tools=[_EXTRACT_TERMS_TOOL],
                tool_choice={"type": "tool", "name": "extract_financial_terms"},
                messages=[{"role": "user", "content": f"질문: {query}"}],
            )
        except (RateLimitError, APITimeoutError, APIConnectionError, APIError) as e:
            logger.warning("[TermExplainer] 용어 추출 Claude 호출 실패: %s", e)
            return []

        for block in response.content:
            if block.type == "tool_use" and block.name == "extract_financial_terms":
                terms = block.input.get("terms", [])
                return [t for t in terms if isinstance(t, str) and t.strip()][:_MAX_TERMS]

        logger.warning("[TermExplainer] tool_use 블록 없음. 빈 목록 반환.")
        return []

    # ── 뉴스 기반 용어 추출 ──────────────────────────────────────────────────

    def _run_from_news(
        self,
        query_summary: str,
        news_output: NewsCollectorOutput,
    ) -> TermExplainerOutput:
        """
        뉴스 기사 제목+요약 텍스트에서 금융 용어를 자동 감지하고 설명한다.
        RAG retriever의 extract_and_explain_terms (BM25+Dense 하이브리드)를 사용한다.
        """
        news_text = "\n".join(
            f"{item.title}. {item.summary}"
            for item in news_output.news_items
        )

        logger.info("[TermExplainer] 뉴스 텍스트(%d자)에서 용어 추출 시작.", len(news_text))

        try:
            term_dicts = extract_and_explain_terms(news_text, max_terms=_MAX_FROM_NEWS)
        except Exception as e:
            logger.error("[TermExplainer] RAG 용어 추출 실패: %s", e)
            term_dicts = []

        annotations = [
            TermAnnotation(term=d["term"], explanation=d["explanation"])
            for d in term_dicts
            if d.get("term") and d.get("explanation")
        ]

        return TermExplainerOutput(
            original_text=news_text,
            annotated_text=news_text,
            terms_explained=annotations,
        )

    # ── 공통: 용어 설명 ──────────────────────────────────────────────────────

    def _explain_terms(self, terms: list[str]) -> list[TermAnnotation]:
        """
        용어 목록 각각에 대해 RAG+폴백(Wikipedia+Claude)으로 설명을 생성한다.
        설명 생성에 실패한 용어는 건너뛴다.
        """
        annotations = []
        for term in terms:
            try:
                explanation = get_term_explanation_with_fallback(term)
            except Exception as e:
                logger.error("[TermExplainer] '%s' 설명 조회 예외: %s", term, e)
                explanation = None

            if explanation:
                annotations.append(TermAnnotation(term=term, explanation=explanation))
            else:
                logger.warning("[TermExplainer] '%s' 설명 생성 실패. 건너뜀.", term)

        return annotations

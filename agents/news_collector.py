"""
agents/news_collector.py

뉴스 수집 에이전트.

실행 흐름:
  1. ExecutionPlan에서 의도·ticker 파악
  2. STOCK_QUERY → TICKER_KEYWORD_MAP 키워드 사용
     MARKET_TREND → MARKET_KEYWORDS 사용
  3. NewsAPI → 네이버 RSS 폴백으로 기사 수집 (최대 10건)
  4. Claude Haiku로 관련도(relevance_score 0~1)·파급력(impact) 일괄 평가
  5. 관련도 내림차순 정렬 후 상위 5건 반환
"""

from __future__ import annotations

import json
import logging
import os
import sys

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import anthropic
from anthropic import APIConnectionError, APIError, APITimeoutError, RateLimitError
from dotenv import load_dotenv

from schemas.models import (
    ExecutionPlan,
    IntentType,
    ModelTier,
    NewsCollectorOutput,
    NewsItem,
)
from tools.keyword_map import MARKET_KEYWORDS, get_keywords_for_ticker
from tools.news_api import fetch_news

load_dotenv()
logger = logging.getLogger(__name__)

_MODEL      = ModelTier.LIGHT.value   # claude-haiku: 저비용 분류 작업
_MAX_FETCH  = 10   # 수집 기사 최대 수
_MAX_RETURN = 5    # 반환 기사 최대 수
_MAX_KW     = 4    # 검색 키워드 최대 수

# ─── Claude 스코어링 도구 스키마 ─────────────────────────────────────────────
_SCORE_TOOL: dict = {
    "name": "score_articles",
    "description": "뉴스 기사 목록의 관련도와 주가 파급력을 평가합니다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "description": "기사별 평가 결과. 입력 기사 순서와 동일한 인덱스 사용.",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "integer",
                            "description": "기사 번호 (0-based)",
                        },
                        "relevance_score": {
                            "type": "number",
                            "description": (
                                "조회 맥락과의 관련도. "
                                "1.0=핵심 기사, 0.7=간접 관련, 0.3=배경 정보, 0.0=무관"
                            ),
                        },
                        "impact": {
                            "type": "string",
                            "enum": ["HIGH", "MEDIUM", "LOW"],
                            "description": (
                                "HIGH=실적·M&A·규제 등 주가 5% 이상 영향 가능, "
                                "MEDIUM=업황·경쟁사 등 주목할 뉴스, "
                                "LOW=단순 정보성"
                            ),
                        },
                    },
                    "required": ["index", "relevance_score", "impact"],
                },
            }
        },
        "required": ["scores"],
    },
}


class NewsCollectorError(Exception):
    """뉴스 수집 중 복구 불가능한 오류."""


class NewsCollectorAgent:
    """
    키워드 기반으로 뉴스를 수집하고 Claude로 관련도·파급력을 평가해
    NewsCollectorOutput을 반환한다.
    """

    def __init__(self) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise NewsCollectorError("ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")
        self._client = anthropic.Anthropic(api_key=api_key)

    # ── 공개 메서드 ──────────────────────────────────────────────────────────

    def run(self, plan: ExecutionPlan) -> NewsCollectorOutput:
        """
        ExecutionPlan을 받아 뉴스를 수집·평가한 결과를 반환한다.

        뉴스가 없거나 수집에 실패해도 예외 없이 news_items=[]로 반환한다.
        """
        orch     = plan.orchestrator_output
        keywords = self._select_keywords(orch.intent, orch.ticker, orch.company_name)

        if not keywords:
            logger.warning("[NewsCollector] 사용 가능한 키워드가 없습니다.")
            return NewsCollectorOutput(
                ticker=orch.ticker,
                company_name=orch.company_name,
                news_items=[],
                keywords_used=[],
            )

        logger.info("[NewsCollector] 키워드 %d개로 수집 시작: %s", len(keywords), keywords)

        raw_articles = fetch_news(keywords[:_MAX_KW], max_per_keyword=_MAX_FETCH // _MAX_KW)

        if not raw_articles:
            logger.info("[NewsCollector] 수집된 기사 없음.")
            return NewsCollectorOutput(
                ticker=orch.ticker,
                company_name=orch.company_name,
                news_items=[],
                keywords_used=keywords,
            )

        raw_articles = raw_articles[:_MAX_FETCH]
        scored = self._score_articles(raw_articles, orch.intent, orch.ticker, orch.company_name, keywords)

        # 관련도 내림차순 정렬 후 상위 N건 반환
        scored.sort(key=lambda x: x["relevance_score"], reverse=True)
        news_items = [self._to_news_item(a) for a in scored[:_MAX_RETURN]]

        logger.info("[NewsCollector] 최종 반환: %d건", len(news_items))
        return NewsCollectorOutput(
            ticker=orch.ticker,
            company_name=orch.company_name,
            news_items=news_items,
            keywords_used=keywords,
        )

    # ── 내부: 키워드 선택 ────────────────────────────────────────────────────

    def _select_keywords(
        self,
        intent: IntentType,
        ticker: str | None,
        company_name: str | None,
    ) -> list[str]:
        """의도·ticker에 따라 뉴스 검색 키워드를 선택한다."""
        if intent == IntentType.STOCK_QUERY:
            if ticker:
                return get_keywords_for_ticker(ticker)
            if company_name:
                return [company_name]
            return []

        if intent == IntentType.MARKET_TREND:
            return MARKET_KEYWORDS[:_MAX_KW]

        logger.warning("[NewsCollector] 지원하지 않는 의도: %s", intent)
        return []

    # ── 내부: Claude 스코어링 ─────────────────────────────────────────────────

    def _score_articles(
        self,
        articles: list[dict],
        intent: IntentType,
        ticker: str | None,
        company_name: str | None,
        keywords: list[str],
    ) -> list[dict]:
        """
        Claude Haiku로 각 기사의 relevance_score와 impact를 일괄 평가한다.
        Claude 호출 실패 시 키워드 기반 폴백 스코어링으로 전환한다.
        """
        intent_desc = {
            IntentType.STOCK_QUERY:  f"{company_name or ticker} 종목 분석",
            IntentType.MARKET_TREND: "한국 주식시장 전체 동향",
        }.get(intent, "주식 투자 관련 동향")

        subject = company_name or ticker or "한국 증시 전반"

        articles_for_prompt = [
            {
                "index": i,
                "title": a.get("title", ""),
                "summary": (a.get("summary") or "")[:150],
            }
            for i, a in enumerate(articles)
        ]

        system_prompt = (
            "당신은 한국 주식 투자 뉴스 분석 전문가입니다.\n\n"
            f"[조회 맥락]\n"
            f"목적: {intent_desc}\n"
            f"분석 대상: {subject}\n"
            f"검색 키워드: {', '.join(keywords)}\n\n"
            "[관련도 기준]\n"
            "1.0 — 분석 대상을 제목·본문에 직접 다루는 핵심 기사\n"
            "0.7 — 관련 업종·경쟁사·공급망 이슈 등 간접 관련 기사\n"
            "0.3 — 환율·금리 등 배경 시장 정보\n"
            "0.0 — 분석 대상과 무관한 기사\n\n"
            "[파급력 기준]\n"
            "HIGH   — 실적 발표, M&A, 규제 이슈, 경영진 변동 등 주가 5% 이상 영향 예상\n"
            "MEDIUM — 업황 변화, 신제품, 경쟁사 동향 등 주목할 뉴스\n"
            "LOW    — 단순 정보성·시장 배경 기사"
        )

        articles_json = json.dumps(articles_for_prompt, ensure_ascii=False)

        try:
            response = self._client.messages.create(
                model=_MODEL,
                max_tokens=1024,
                system=system_prompt,
                tools=[_SCORE_TOOL],
                tool_choice={"type": "tool", "name": "score_articles"},
                messages=[{
                    "role": "user",
                    "content": f"다음 기사들을 평가해주세요:\n\n{articles_json}",
                }],
            )
        except (RateLimitError, APITimeoutError, APIConnectionError, APIError) as e:
            logger.warning("[NewsCollector] Claude 스코어링 실패, 폴백 적용: %s", e)
            return self._fallback_score(articles, keywords)

        for block in response.content:
            if block.type == "tool_use" and block.name == "score_articles":
                return self._merge_scores(articles, block.input.get("scores", []))

        logger.warning("[NewsCollector] tool_use 블록 없음. 폴백 스코어링 적용.")
        return self._fallback_score(articles, keywords)

    def _merge_scores(self, articles: list[dict], scores: list[dict]) -> list[dict]:
        """Claude 평가 결과를 원본 기사 딕셔너리에 병합한다."""
        score_map = {
            s["index"]: (s.get("relevance_score", 0.5), s.get("impact", "MEDIUM"))
            for s in scores
            if isinstance(s.get("index"), int)
        }
        result = []
        for i, article in enumerate(articles):
            rel, impact = score_map.get(i, (0.5, "MEDIUM"))
            result.append({
                **article,
                "relevance_score": float(max(0.0, min(1.0, rel))),
                "impact": impact if impact in ("HIGH", "MEDIUM", "LOW") else "MEDIUM",
            })
        return result

    def _fallback_score(self, articles: list[dict], keywords: list[str]) -> list[dict]:
        """
        Claude 호출 불가 시 키워드 매칭 기반으로 점수를 부여한다.
        title + summary 내 키워드 포함 비율이 relevance_score가 된다.
        """
        HIGH_IMPACT_KW = {"급등", "급락", "실적", "계약", "인수", "합병", "규제", "처벌", "파산", "분기"}
        result = []
        for article in articles:
            text        = (article.get("title", "") + " " + (article.get("summary") or "")).lower()
            match_count = sum(1 for kw in keywords if kw.lower() in text)
            rel         = min(1.0, match_count / max(len(keywords), 1))
            impact      = "HIGH" if any(kw in text for kw in HIGH_IMPACT_KW) else "MEDIUM"
            result.append({**article, "relevance_score": rel, "impact": impact})
        return result

    # ── 내부: 모델 변환 ──────────────────────────────────────────────────────

    @staticmethod
    def _to_news_item(article: dict) -> NewsItem:
        """딕셔너리를 NewsItem Pydantic 모델로 변환한다."""
        return NewsItem(
            title=article.get("title", "(제목 없음)"),
            summary=article.get("summary", "") or "",
            source=article.get("source", "알 수 없음"),
            published_at=article.get("published_at", ""),
            relevance_score=article.get("relevance_score", 0.5),
            impact=article.get("impact", "MEDIUM"),
        )

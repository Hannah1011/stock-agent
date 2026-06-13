"""
agents/report_generator.py

최종 리포트 생성 에이전트.

실행 흐름:
  1. 이전 에이전트 출력(뉴스·차트·포트폴리오·용어)을 마크다운 컨텍스트로 조합
  2. Claude Sonnet으로 초보 투자자용 마크다운 리포트 합성
  3. 투자 권유 문구 필터링 (guardrails.filter_investment_advice)
  4. 면책 고지 추가 (guardrails.ensure_disclaimer)
  5. FinalReport 반환

Claude Sonnet을 사용하는 이유:
  최종 리포트는 다양한 소스의 정보를 일관된 어조로 통합하고,
  초보 투자자 눈높이에 맞게 재구성해야 하므로 고품질 모델이 필요하다.
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
    ExecutionPlan,
    FinalReport,
    ModelTier,
    NewsCollectorOutput,
    PortfolioAnalyzerOutput,
    TermAnnotation,
    TermExplainerOutput,
)
from utils.guardrails import (
    ensure_disclaimer,
    ensure_investment_caution_section,
    filter_investment_advice,
)

load_dotenv()
logger = logging.getLogger(__name__)

_MODEL = ModelTier.STANDARD.value   # claude-sonnet-4-6: 최종 합성은 고품질 모델
_REPORT_MAX_TOKENS = 4096
_CONTINUATION_MAX_TOKENS = 2048


class ReportGeneratorError(Exception):
    """리포트 생성 중 복구 불가능한 오류."""


class ReportGeneratorAgent:
    """
    이전 에이전트들의 출력을 받아 초보 투자자용 마크다운 리포트를 합성한다.
    """

    def __init__(self) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ReportGeneratorError("ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")
        self._client = anthropic.Anthropic(api_key=api_key)

    # ── 공개 메서드 ──────────────────────────────────────────────────────────

    def run(
        self,
        plan: ExecutionPlan,
        news_output: Optional[NewsCollectorOutput]          = None,
        chart_output: Optional[ChartAnalystOutput]          = None,
        portfolio_output: Optional[PortfolioAnalyzerOutput] = None,
        term_output: Optional[TermExplainerOutput]          = None,
    ) -> FinalReport:
        """
        수집된 에이전트 결과를 하나의 마크다운 리포트로 합성한다.

        None인 출력은 건너뛴다. 모든 입력이 None이면 기본 응답 메시지를 반환한다.
        Claude 호출 실패 시 컨텍스트 원본을 리포트로 반환한다.
        """
        orch       = plan.orchestrator_output
        context_md = self._build_context_markdown(news_output, chart_output, portfolio_output, term_output)

        logger.info("[ReportGenerator] 리포트 합성 시작 (intent=%s)", orch.intent.value)

        full_md = self._generate_report(orch.query_summary, context_md)
        full_md = filter_investment_advice(full_md)
        full_md = ensure_investment_caution_section(full_md)
        full_md = ensure_disclaimer(full_md)

        one_line = self._extract_one_line_summary(full_md)

        logger.info("[ReportGenerator] 리포트 생성 완료 (%d자)", len(full_md))

        return FinalReport(
            one_line_summary=one_line,
            news_section=self._extract_section(full_md, "뉴스"),
            chart_section=self._extract_section(full_md, "차트"),
            portfolio_section=self._extract_section(full_md, "포트폴리오"),
            risk_alerts=self._build_risk_alerts(portfolio_output, chart_output),
            terms_glossary=self._build_terms_glossary(term_output),
            full_report_md=full_md,
        )

    # ── 내부: 컨텍스트 조합 ──────────────────────────────────────────────────

    def _build_context_markdown(
        self,
        news: Optional[NewsCollectorOutput],
        chart: Optional[ChartAnalystOutput],
        portfolio: Optional[PortfolioAnalyzerOutput],
        terms: Optional[TermExplainerOutput],
    ) -> str:
        """
        각 에이전트 출력을 마크다운 블록으로 변환해 하나의 컨텍스트 문자열로 조합한다.
        Claude가 이 컨텍스트를 기반으로 최종 리포트를 작성한다.
        """
        sections: list[str] = []

        if news and news.news_items:
            item_blocks = []
            for i, item in enumerate(news.news_items):
                # URL은 Google News 리다이렉트 URL 신뢰 문제로 제목만 표시
                block = (
                    f"[{i + 1}] [{item.impact}] {item.title}\n"
                    f"    출처: {item.source} | 날짜: {item.published_at}"
                )
                if item.summary:
                    block += f"\n    요약: {item.summary}"
                item_blocks.append(block)
            items_md = "\n".join(item_blocks)
            sections.append(f"## 수집된 뉴스\n{items_md}")

        if chart:
            cd = chart.chart_data
            sections.append(
                f"## 차트 분석\n"
                f"종목: {cd.ticker} | 현재가: {cd.current_price:,.0f}원 ({cd.change_pct:+.2f}%)\n"
                f"추세: {cd.trend_summary}\n"
                f"RSI: {cd.rsi} | MA5: {cd.ma5} | MA20: {cd.ma20} | MA60: {cd.ma60}\n"
                f"거래량 신호: {cd.volume_signal}\n\n"
                f"[초보자 해설]\n{chart.plain_explanation}"
            )

        if portfolio:
            holdings_md = "\n".join(
                f"- {h.name}({h.ticker}): 현재가 {h.current_price:,.0f}원, "
                f"수익률 {h.profit_loss_pct:+.1f}%, 평가금액 {h.eval_amount:,.0f}원"
                for h in portfolio.holdings
            )
            sector_md = ", ".join(
                f"{s}: {w * 100:.1f}%"
                for s, w in portfolio.sector_distribution.items()
            )
            sections.append(
                f"## 포트폴리오 현황\n{holdings_md}\n\n"
                f"총 평가금액: {portfolio.total_eval:,.0f}원 | "
                f"총 수익률: {portfolio.total_profit_loss_pct:+.2f}%\n"
                f"섹터 분포: {sector_md}\n"
                f"위험 등급: {portfolio.risk.volatility_level} | "
                f"집중도: {portfolio.risk.sector_concentration:.1%}\n"
                f"분산 조언: {portfolio.risk.diversification_advice}"
            )

        if terms and terms.terms_explained:
            terms_md = "\n".join(
                f"- {t.term}: {t.explanation}"
                for t in terms.terms_explained
            )
            sections.append(f"## 용어 해설\n{terms_md}")

        return "\n\n".join(sections) if sections else "수집된 분석 데이터가 없습니다."

    # ── 내부: Claude 리포트 합성 ─────────────────────────────────────────────

    def _generate_report(self, query_summary: str, context_md: str) -> str:
        """
        Claude Sonnet으로 최종 마크다운 리포트를 생성한다.
        API 오류 시 수집된 컨텍스트를 그대로 리포트로 반환한다.
        """
        prompt = (
            "당신은 주식 투자 완전 초보자를 위한 재테크 AI 리포트 작성 전문가입니다.\n\n"
            f"[사용자 질문]\n{query_summary}\n\n"
            f"[수집된 분석 데이터]\n{context_md}\n\n"
            "[리포트 작성 지침]\n"
            "해당 데이터가 없는 섹션은 생략합니다. 아래 순서대로 작성하세요.\n\n"
            "1. ### 한 줄 요약\n"
            "   - 이번 분석의 핵심을 15단어 이내 한 문장으로\n\n"
            "2. ### 주요 뉴스 (뉴스 데이터가 있는 경우)\n"
            "   - 중요도 높은 순서대로 최대 3개 정리\n"
            "   - 기사 제목은 일반 텍스트로만 작성 (마크다운 링크 금지)\n"
            "   - 각 뉴스가 주가에 어떤 영향을 줄 수 있는지 1문장 추가\n\n"
            "3. ### 차트 분석 (차트 데이터가 있는 경우)\n"
            "   - 초보자 해설을 기반으로 주가 현황을 정리\n"
            "   - 수치보다 '지금 주가가 어느 위치에 있는지'를 중심으로\n\n"
            "4. ### 내 포트폴리오 (포트폴리오 데이터가 있는 경우)\n"
            "   - 종목별 수익률을 한눈에 볼 수 있게 표로 정리\n"
            "   - 전체 평가금액·수익률·위험도를 요약\n"
            "   - 분산 조언 포함\n\n"
            "5. ### 오늘의 금융 용어 (용어 데이터가 있는 경우)\n"
            "   - 분석에 등장한 용어를 초보자 언어로 최대 3개 설명\n\n"
            "6. ### 투자 유의사항\n"
            "   - 반드시 포함. 본 정보가 투자 권유가 아님을 명시\n\n"
            "[문체]\n"
            "- 친근하고 이해하기 쉬운 한국어\n"
            "- 전문 용어는 괄호로 쉬운 설명 병기 (예: RSI(주가 힘의 세기 지표))\n"
            "- 숫자는 단위 포함 (예: 68,000원, +5.3%, 시가총액 2조 원)\n"
            "- 일반 본문과 설명 문장은 굵게 표시하지 않음\n"
            "- 굵게 표시는 핵심 용어명·종목명·짧은 핵심 라벨에만 제한\n"
            "- 문장 전체, 목록 전체, 설명 전체를 **굵게** 표시하지 않음\n"
            "- 오늘의 금융 용어는 '**용어명**: 굵지 않은 설명' 형식으로 작성\n\n"
            "[절대 금지]\n"
            "- 이모지·이모티콘·특수문자 아이콘 일체 금지 (예: 📊 ✅ ⚠️ 📈 등)\n"
            "- 마크다운 링크([텍스트](URL)) 형식 금지 — 제목은 일반 텍스트로만\n"
            "- '매수하세요', '지금 사야 합니다', '팔아야 합니다' 등 직접적 매매 권유\n"
            "- '반드시 오릅니다', '손실 없음' 등 수익 보장 표현\n"
            "- 특정 종목의 목표 주가 제시"
        )

        try:
            response = self._client.messages.create(
                model=_MODEL,
                max_tokens=_REPORT_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            report = self._extract_text(response)
            if response.stop_reason == "max_tokens" and report:
                logger.warning("[ReportGenerator] 출력 토큰 한도 도달, 리포트를 이어서 생성합니다.")
                continuation = self._continue_report(prompt, report)
                if continuation:
                    report = f"{report.rstrip()}\n{continuation.lstrip()}"
            return report if report else context_md
        except (RateLimitError, APITimeoutError, APIConnectionError, APIError) as e:
            logger.error("[ReportGenerator] Claude 호출 실패: %s", e)
            return (
                f"## 분석 결과\n\n{context_md}\n\n"
                "> AI 리포트 자동 생성에 실패해 수집된 원본 데이터를 표시합니다."
            )

    def _continue_report(self, original_prompt: str, partial_report: str) -> str:
        """토큰 한도로 잘린 리포트의 남은 부분만 이어서 생성한다."""
        try:
            response = self._client.messages.create(
                model=_MODEL,
                max_tokens=_CONTINUATION_MAX_TOKENS,
                messages=[
                    {"role": "user", "content": original_prompt},
                    {"role": "assistant", "content": partial_report},
                    {
                        "role": "user",
                        "content": (
                            "응답이 토큰 한도로 중간에 끊겼습니다. "
                            "이미 작성한 내용을 반복하지 말고 끊긴 지점부터 이어서 작성하세요. "
                            "마지막에는 반드시 '### 투자 유의사항' 섹션을 완결하세요."
                        ),
                    },
                ],
            )
            return self._extract_text(response)
        except (RateLimitError, APITimeoutError, APIConnectionError, APIError) as e:
            logger.warning("[ReportGenerator] 리포트 이어쓰기 실패, 후처리로 보완합니다: %s", e)
            return ""

    @staticmethod
    def _extract_text(response: anthropic.types.Message) -> str:
        """Claude 응답의 모든 text 블록을 하나의 문자열로 합친다."""
        return "\n".join(
            block.text
            for block in response.content
            if block.type == "text" and block.text
        ).strip()

    # ── 내부: 섹션 추출 헬퍼 ────────────────────────────────────────────────

    @staticmethod
    def _extract_one_line_summary(report_md: str) -> str:
        """리포트에서 '한 줄 요약' 섹션의 첫 번째 문장을 추출한다."""
        lines = report_md.splitlines()
        in_summary = False
        for line in lines:
            if "한 줄 요약" in line:
                in_summary = True
                continue
            if in_summary:
                # 다음 섹션 헤더가 나오면 종료
                if line.startswith("##"):
                    break
                stripped = line.strip().lstrip("-").strip()
                if stripped:
                    return stripped[:100]
        # 한 줄 요약 섹션이 없으면 첫 번째 비어있지 않은 비헤더 줄 사용
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return stripped[:100]
        return "분석 완료"

    @staticmethod
    def _extract_section(report_md: str, keyword: str) -> Optional[str]:
        """
        리포트 마크다운에서 keyword를 포함한 섹션 텍스트를 추출한다.
        다음 ## 헤더 전까지의 내용을 반환한다.
        """
        lines         = report_md.splitlines()
        in_section    = False
        section_lines: list[str] = []

        for line in lines:
            if line.startswith("##") and keyword in line:
                in_section = True
                continue
            if in_section:
                if line.startswith("##"):
                    break
                section_lines.append(line)

        content = "\n".join(section_lines).strip()
        return content if content else None

    @staticmethod
    def _build_risk_alerts(
        portfolio: Optional[PortfolioAnalyzerOutput],
        chart: Optional[ChartAnalystOutput],
    ) -> list[str]:
        """포트폴리오·차트 데이터에서 주의 알림 목록을 생성한다."""
        alerts: list[str] = []

        if portfolio:
            risk = portfolio.risk
            if risk.volatility_level == "HIGH":
                alerts.append(
                    f"포트폴리오 집중도가 높습니다 "
                    f"(집중도 {risk.sector_concentration:.1%}). "
                    "다양한 섹터로 분산을 고려해 보세요."
                )
            for h in portfolio.holdings:
                if h.profit_loss_pct <= -15.0:
                    alerts.append(
                        f"{h.name}: 평균 매수가 대비 {h.profit_loss_pct:.1f}% 하락 중입니다."
                    )

        if chart and chart.chart_data.rsi is not None:
            rsi = chart.chart_data.rsi
            if rsi >= 70:
                alerts.append(
                    f"RSI {rsi:.1f}: 단기 과매수 구간입니다. "
                    "추가 상승 여력을 신중히 판단하세요."
                )
            elif rsi <= 30:
                alerts.append(
                    f"RSI {rsi:.1f}: 단기 과매도 구간입니다. "
                    "추가 하락 가능성에 유의하세요."
                )

        return alerts

    @staticmethod
    def _build_terms_glossary(
        term_output: Optional[TermExplainerOutput],
    ) -> list[TermAnnotation]:
        """TermExplainerOutput에서 용어 해설 목록을 반환한다 (최대 3개)."""
        if term_output and term_output.terms_explained:
            return term_output.terms_explained[:3]
        return []

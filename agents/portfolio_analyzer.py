"""
agents/portfolio_analyzer.py

포트폴리오 분석 에이전트.

실행 흐름:
  1. portfolio.json 로드
     - 기본 경로: PROJECT_ROOT/portfolio.json
     - 환경변수 PORTFOLIO_PATH로 경로 덮어쓰기 가능
  2. 보유 종목별 yfinance 현재가 조회
  3. 종목별 평가금액·수익률 계산
  4. stock_resolver로 섹터 정보 수집 → 평가금액 기준 섹터 분포 계산
  5. 섹터·종목 집중도로 위험 등급(LOW/MEDIUM/HIGH) 산정
  6. Claude Haiku로 분산투자 조언 생성
  7. PortfolioAnalyzerOutput 반환
"""

from __future__ import annotations

import json
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
    ExecutionPlan,
    HoldingStatus,
    ModelTier,
    PortfolioAnalyzerOutput,
    RiskScore,
)
from tools.stock_api import get_current_price
from tools.stock_resolver import get_sector_by_ticker

load_dotenv()
logger = logging.getLogger(__name__)

_MODEL         = ModelTier.LIGHT.value   # claude-haiku: 조언 텍스트 생성
PORTFOLIO_PATH = os.path.join(PROJECT_ROOT, "portfolio.json")


class PortfolioAnalyzerError(Exception):
    """포트폴리오 분석 중 복구 불가능한 오류."""


class PortfolioAnalyzerAgent:
    """
    portfolio.json을 읽어 보유 종목 수익률·리스크를 분석하고
    PortfolioAnalyzerOutput을 반환한다.
    """

    def __init__(self) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise PortfolioAnalyzerError("ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")
        self._client = anthropic.Anthropic(api_key=api_key)

    # ── 공개 메서드 ──────────────────────────────────────────────────────────

    def run(self, plan: ExecutionPlan) -> PortfolioAnalyzerOutput:
        """
        포트폴리오를 분석하고 결과를 반환한다.

        Raises:
            PortfolioAnalyzerError:
              - portfolio.json이 없거나 JSON 형식이 잘못된 경우
              - 현재가 조회에 성공한 종목이 한 개도 없는 경우
        """
        raw_holdings     = self._load_portfolio()
        holding_statuses = self._build_holding_statuses(raw_holdings)

        if not holding_statuses:
            raise PortfolioAnalyzerError(
                "현재가 조회에 성공한 보유 종목이 없습니다. "
                "ticker 코드와 인터넷 연결을 확인해 주세요."
            )

        total_cost   = sum(h.shares * h.avg_price for h in holding_statuses)
        total_eval   = sum(h.eval_amount for h in holding_statuses)
        total_pl_pct = ((total_eval - total_cost) / total_cost * 100) if total_cost > 0 else 0.0

        sector_dist = self._build_sector_distribution(holding_statuses)
        risk        = self._build_risk_score(holding_statuses, sector_dist, total_eval)

        logger.info(
            "[PortfolioAnalyzer] 분석 완료: 종목 %d개, 총평가 %.0f원, 수익률 %.2f%%",
            len(holding_statuses), total_eval, total_pl_pct,
        )

        return PortfolioAnalyzerOutput(
            holdings=holding_statuses,
            total_eval=round(total_eval, 0),
            total_profit_loss_pct=round(total_pl_pct, 2),
            sector_distribution=sector_dist,
            risk=risk,
        )

    # ── 내부: 포트폴리오 로드 ────────────────────────────────────────────────

    def _load_portfolio(self) -> list[dict]:
        """
        portfolio.json을 로드한다.

        예상 형식:
          {"holdings": [
            {"ticker": "005930.KS", "name": "삼성전자", "shares": 10, "avg_price": 68000},
            ...
          ]}

        Raises:
            PortfolioAnalyzerError: 파일 없음·파싱 실패·필수 필드 누락
        """
        portfolio_path = os.getenv("PORTFOLIO_PATH", PORTFOLIO_PATH)

        if not os.path.exists(portfolio_path):
            raise PortfolioAnalyzerError(
                f"portfolio.json 파일이 없습니다: {portfolio_path}\n"
                "프로젝트 루트에 portfolio.json을 생성하거나 "
                "PORTFOLIO_PATH 환경변수로 경로를 지정해 주세요."
            )

        try:
            with open(portfolio_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise PortfolioAnalyzerError(f"portfolio.json 파싱 실패: {e}") from e

        holdings = data.get("holdings", [])
        if not holdings:
            raise PortfolioAnalyzerError("portfolio.json에 보유 종목(holdings)이 없습니다.")

        required_fields = {"ticker", "name", "shares", "avg_price"}
        for i, h in enumerate(holdings):
            missing = required_fields - set(h.keys())
            if missing:
                raise PortfolioAnalyzerError(
                    f"portfolio.json [{i}]번 항목에 필수 필드가 없습니다: {missing}"
                )

        return holdings

    # ── 내부: 보유 상태 계산 ─────────────────────────────────────────────────

    def _build_holding_statuses(self, raw_holdings: list[dict]) -> list[HoldingStatus]:
        """
        각 보유 종목의 현재가를 조회해 HoldingStatus 목록을 반환한다.
        현재가 조회에 실패한 종목은 경고 로그를 남기고 건너뛴다.
        """
        statuses = []
        for h in raw_holdings:
            ticker    = h["ticker"]
            avg_price = float(h["avg_price"])
            shares    = int(h["shares"])

            price_data = get_current_price(ticker)
            if price_data is None:
                logger.warning("[PortfolioAnalyzer] 현재가 조회 실패, 건너뜀: %s", ticker)
                continue

            current_price = price_data["current_price"]
            eval_amount   = current_price * shares
            pl_pct        = ((current_price - avg_price) / avg_price * 100) if avg_price > 0 else 0.0

            statuses.append(HoldingStatus(
                ticker=ticker,
                name=h["name"],
                shares=shares,
                avg_price=avg_price,
                current_price=round(current_price, 0),
                profit_loss_pct=round(pl_pct, 2),
                eval_amount=round(eval_amount, 0),
            ))

        return statuses

    # ── 내부: 섹터 분포 ──────────────────────────────────────────────────────

    def _build_sector_distribution(self, holdings: list[HoldingStatus]) -> dict:
        """
        보유 종목의 평가금액 기준 섹터 분포(비율)를 계산한다.
        섹터 정보를 가져오지 못한 종목은 '기타'로 분류한다.
        """
        total_eval = sum(h.eval_amount for h in holdings)
        if total_eval == 0:
            return {}

        sector_totals: dict[str, float] = {}
        for h in holdings:
            try:
                sector = get_sector_by_ticker(h.ticker) or "기타"
            except Exception as e:
                logger.debug("[PortfolioAnalyzer] 섹터 조회 실패 (%s): %s", h.ticker, e)
                sector = "기타"
            sector_totals[sector] = sector_totals.get(sector, 0.0) + h.eval_amount

        # 비율 변환 후 비중 내림차순 정렬
        return {
            sector: round(amount / total_eval, 4)
            for sector, amount in sorted(sector_totals.items(), key=lambda x: -x[1])
        }

    # ── 내부: 리스크 점수 ────────────────────────────────────────────────────

    def _build_risk_score(
        self,
        holdings: list[HoldingStatus],
        sector_dist: dict,
        total_eval: float,
    ) -> RiskScore:
        """
        섹터 집중도와 단일 종목 비중 중 더 높은 값을 기준으로 RiskScore를 산정한다.

        sector_concentration:
          max(섹터 비중, 최대 단일 종목 비중) — 0~1
        volatility_level:
          >= 0.60 → HIGH
          >= 0.35 → MEDIUM
          <  0.35 → LOW
        """
        max_sector_weight = max(sector_dist.values(), default=0.0)
        max_stock_weight  = (
            max(h.eval_amount for h in holdings) / total_eval
            if total_eval > 0 and holdings else 0.0
        )

        # 섹터 또는 단일 종목 집중도 중 더 높은 값 사용
        concentration = max(max_sector_weight, max_stock_weight)

        if concentration >= 0.60:
            volatility_level = "HIGH"
        elif concentration >= 0.35:
            volatility_level = "MEDIUM"
        else:
            volatility_level = "LOW"

        advice = self._generate_diversification_advice(holdings, sector_dist, volatility_level)

        return RiskScore(
            sector_concentration=round(concentration, 4),
            volatility_level=volatility_level,
            diversification_advice=advice,
        )

    def _generate_diversification_advice(
        self,
        holdings: list[HoldingStatus],
        sector_dist: dict,
        volatility_level: str,
    ) -> str:
        """
        Claude Haiku로 현재 포트폴리오에 맞는 분산투자 조언을 생성한다.
        API 오류 시 기본 조언 텍스트를 반환한다.
        """
        holdings_summary = "\n".join(
            f"- {h.name}({h.ticker}): 평가금액 {h.eval_amount:,.0f}원, "
            f"수익률 {h.profit_loss_pct:+.1f}%"
            for h in holdings
        )
        sector_summary = ", ".join(
            f"{sector}: {weight * 100:.1f}%"
            for sector, weight in sector_dist.items()
        )
        risk_desc = {"HIGH": "고위험", "MEDIUM": "중위험", "LOW": "저위험"}[volatility_level]

        prompt = (
            "다음 포트폴리오를 분석하고 주식 투자 초보자를 위한 분산투자 조언을 "
            "2~3문장으로 작성해주세요.\n\n"
            f"[포트폴리오 현황]\n{holdings_summary}\n\n"
            f"[섹터 분포]\n{sector_summary}\n\n"
            f"[위험 등급]: {risk_desc}\n\n"
            "[작성 지침]\n"
            "- '~를 매수하세요', '~를 팔아야 합니다' 등 직접적인 매매 권유 절대 금지\n"
            "- 현재 포트폴리오의 구체적인 특징(집중 섹터, 수익률 현황 등)을 언급\n"
            "- 분산투자의 필요성 또는 현 구성의 강점을 언급\n"
            "- 친근하고 이해하기 쉬운 표현 사용\n"
            "- 조언 본문만 출력 (제목·번호·마크다운 없이)"
        )

        try:
            response = self._client.messages.create(
                model=_MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            advice = response.content[0].text.strip()
            return advice if advice else self._default_advice(volatility_level)
        except (RateLimitError, APITimeoutError, APIConnectionError, APIError) as e:
            logger.warning("[PortfolioAnalyzer] 조언 생성 실패: %s", e)
            return self._default_advice(volatility_level)

    @staticmethod
    def _default_advice(volatility_level: str) -> str:
        """API 호출 실패 시 위험 등급별 기본 조언을 반환한다."""
        defaults = {
            "HIGH":   "포트폴리오가 특정 섹터나 종목에 집중되어 있습니다. 다양한 섹터로 분산하면 위험을 낮출 수 있습니다.",
            "MEDIUM": "전반적으로 양호한 분산 수준이지만, 추가 종목 편입을 고려해 보세요.",
            "LOW":    "포트폴리오가 비교적 잘 분산되어 있습니다. 현재 구성을 유지하면서 정기적으로 리밸런싱을 점검하세요.",
        }
        return defaults.get(volatility_level, "포트폴리오 현황을 정기적으로 점검하세요.")

"""
agents/orchestrator.py

Orchestrator Agent — 전체 워크플로의 첫 번째 관문.

실행 순서:
  1. Guardrail: 재테크 외 질문 즉시 차단
  2. Claude 의도 분류: tool_use로 intent·company_name·query_summary 추출
  3. Ticker 해석: stock_resolver로 회사명 → ticker 변환
     3a. ticker 확정  → 바로 실행 계획 수립
     3b. ticker 불명확 → 사용자에게 후보 확인 요청 플랜 반환
  4. 실행 계획 반환: parallel_groups 포함 ExecutionPlan
"""

import logging
import os
from typing import Optional

import anthropic
from anthropic import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    RateLimitError,
)
from dotenv import load_dotenv

from schemas.models import (
    ClarificationCandidate,
    ExecutionPlan,
    IntentType,
    ModelTier,
    OrchestratorOutput,
)
from tools.stock_resolver import ResolveResult, resolve_company
from utils.guardrails import validate_scope

load_dotenv()
logger = logging.getLogger(__name__)

# ─── 상수 ───────────────────────────────────────────────────────────────────
_MODEL = ModelTier.STANDARD.value  # claude-sonnet-4-6

_SYSTEM_PROMPT = """\
당신은 재테크 AI 어시스턴트의 오케스트레이터입니다.
사용자 질문을 분석해 의도를 분류하고, 언급된 회사명을 추출해야 합니다.

[의도 분류 기준]
A (STOCK_QUERY)   : 특정 종목의 뉴스·차트·분석 질문
  예) "삼성전자 오늘 왜 떨어졌어?", "카카오 차트 분석해줘", "하이닉스 전망"
B (PORTFOLIO)     : 사용자 보유 포트폴리오 분석 요청
  예) "내 포트폴리오 분석해줘", "보유 주식 리스크 알려줘"
C (TERM_QUERY)    : 경제·금융 용어 설명 요청
  예) "RSI가 뭐야?", "코스피랑 코스닥 차이가 뭐야?", "PER 설명해줘"
D (MARKET_TREND)  : 시장 전체 동향·분위기 질문
  예) "오늘 증시 어때?", "코스피 동향", "시장 전망 알려줘"
E (OUT_OF_SCOPE)  : 재테크·주식과 전혀 무관한 질문 (Guardrail에서 이미 걸러지나 안전망)

[company_name 추출 규칙]
- 사용자가 언급한 회사명·별칭을 그대로 추출한다 (ticker 코드 변환 불필요)
- 예) "삼전" → "삼전", "하이닉스" → "하이닉스", "삼성" → "삼성"
- 종목 언급이 없으면 null
- 여러 종목이 언급되면 가장 핵심 종목 하나만 반환
"""

# Claude에게 요청할 tool 스키마
_CLASSIFY_TOOL: dict = {
    "name": "classify_intent",
    "description": "사용자 질문의 의도를 분류하고 관련 종목명을 추출합니다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": ["A", "B", "C", "D", "E"],
                "description": "의도 유형 코드",
            },
            "company_name": {
                "type": "string",
                "description": "사용자가 언급한 회사명·별칭 (없으면 null)",
            },
            "query_summary": {
                "type": "string",
                "description": "사용자 질문을 한 문장으로 요약",
            },
        },
        "required": ["intent", "query_summary"],
    },
}

# 의도별 실행 그룹 정의
# 내부 리스트 = 동시 실행 그룹, 리스트 순서 = 실행 순서
_EXECUTION_MAP: dict[IntentType, list[list[str]]] = {
    IntentType.STOCK_QUERY:  [["NewsCollector", "ChartAnalyst"], ["TermExplainer"], ["ReportGenerator"]],
    IntentType.PORTFOLIO:    [["PortfolioAnalyzer"], ["ReportGenerator"]],
    IntentType.TERM_QUERY:   [["TermExplainer"], ["ReportGenerator"]],
    IntentType.MARKET_TREND: [["NewsCollector"], ["TermExplainer"], ["ReportGenerator"]],
    IntentType.OUT_OF_SCOPE: [],
}

# 의도별 LLM 모델 티어
_MODEL_TIER_MAP: dict[IntentType, ModelTier] = {
    IntentType.STOCK_QUERY:  ModelTier.STANDARD,
    IntentType.PORTFOLIO:    ModelTier.STANDARD,
    IntentType.TERM_QUERY:   ModelTier.LIGHT,
    IntentType.MARKET_TREND: ModelTier.STANDARD,
    IntentType.OUT_OF_SCOPE: ModelTier.LIGHT,
}


# ─── 예외 ────────────────────────────────────────────────────────────────────
class OrchestratorError(Exception):
    """복구 불가능한 Orchestrator 오류. 사용자에게 노출 가능한 메시지를 담는다."""


# ─── Agent 클래스 ────────────────────────────────────────────────────────────
class OrchestratorAgent:
    """
    사용자 입력 → Guardrail → 의도 분류 → 실행 계획 수립.
    외부에서는 run() 또는 run_with_ticker() 두 메서드만 사용한다.
    """

    def __init__(self) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise OrchestratorError("ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")
        self._client = anthropic.Anthropic(api_key=api_key)

    # ── 공개 메서드 ──────────────────────────────────────────────────────────

    def run(self, user_input: str) -> ExecutionPlan:
        """
        일반 진입점. 사용자 입력을 받아 ExecutionPlan을 반환한다.

        반환된 plan의 상태:
          - needs_clarification=True  → UI가 후보를 보여주고 사용자 선택 대기
          - rejection_message 설정    → 범위 외 질문
          - 그 외                     → 정상 실행
        """
        user_input = user_input.strip()
        if not user_input:
            return self._make_rejection_plan("질문을 입력해 주세요.")

        # Step 1: Guardrail — 항상 첫 번째로 실행
        allowed, rejection_msg = validate_scope(user_input)
        if not allowed:
            logger.info("[Orchestrator] Guardrail 차단: %.50s", user_input)
            return self._make_rejection_plan(rejection_msg)

        # Step 2: Claude로 의도 분류
        orch_output = self._classify(user_input)

        # Step 3: 종목 질문이면 ticker 해석
        if orch_output.intent == IntentType.STOCK_QUERY and orch_output.company_name:
            return self._resolve_ticker_and_plan(orch_output)

        # 포트폴리오·용어·시장 동향은 ticker 불필요 → 바로 계획 수립
        return self._make_execution_plan(orch_output)

    def run_with_ticker(self, user_input: str, confirmed_ticker: str, confirmed_name: str) -> ExecutionPlan:
        """
        사용자가 후보 중 종목을 선택한 뒤 호출하는 진입점.
        ticker 해석 단계를 건너뛰고 바로 실행 계획을 수립한다.
        """
        user_input = user_input.strip()
        orch_output = self._classify(user_input)
        orch_output = orch_output.model_copy(update={
            "ticker": confirmed_ticker,
            "company_name": confirmed_name,
        })
        return self._make_execution_plan(orch_output)

    # ── 내부 메서드: 의도 분류 ────────────────────────────────────────────────

    def _classify(self, user_input: str) -> OrchestratorOutput:
        """Claude tool_use로 의도를 분류하고 OrchestratorOutput을 반환한다."""
        raw_input = self._call_claude(user_input)
        return self._parse_tool_output(raw_input, user_input)

    def _call_claude(self, user_input: str) -> dict:
        """
        Claude API를 호출해 tool_use 입력값을 반환한다.
        API 오류 유형별로 구체적인 메시지를 담아 OrchestratorError로 래핑한다.
        """
        try:
            response = self._client.messages.create(
                model=_MODEL,
                max_tokens=512,
                system=_SYSTEM_PROMPT,
                tools=[_CLASSIFY_TOOL],
                tool_choice={"type": "tool", "name": "classify_intent"},
                messages=[{"role": "user", "content": user_input}],
            )
        except RateLimitError as e:
            raise OrchestratorError(
                "API 요청 한도를 초과했습니다. 잠시 후 다시 시도해 주세요."
            ) from e
        except APITimeoutError as e:
            raise OrchestratorError(
                "API 응답이 시간 초과됐습니다. 네트워크 상태를 확인해 주세요."
            ) from e
        except APIConnectionError as e:
            raise OrchestratorError(
                "Anthropic 서버에 연결할 수 없습니다. 인터넷 연결을 확인해 주세요."
            ) from e
        except APIError as e:
            raise OrchestratorError(
                f"Anthropic API 오류 (HTTP {e.status_code}): {e.message}"
            ) from e

        return self._extract_tool_input(response)

    def _extract_tool_input(self, response: anthropic.types.Message) -> dict:
        """응답 content에서 tool_use 블록의 input을 꺼낸다."""
        for block in response.content:
            if block.type == "tool_use" and block.name == "classify_intent":
                return block.input

        # tool_use 블록이 없는 비정상 응답 → 안전 폴백
        logger.warning("[Orchestrator] tool_use 블록 누락. D(시장동향)로 폴백합니다.")
        return {"intent": "D", "query_summary": "시장 동향 질문으로 처리됩니다."}

    def _parse_tool_output(self, tool_input: dict, original_query: str) -> OrchestratorOutput:
        """tool_use 입력값을 검증해 OrchestratorOutput으로 변환한다."""
        # intent 검증
        raw_intent = tool_input.get("intent", "D")
        try:
            intent = IntentType(raw_intent)
        except ValueError:
            logger.warning("[Orchestrator] 알 수 없는 intent '%s'. D로 폴백합니다.", raw_intent)
            intent = IntentType.MARKET_TREND

        # company_name: None 또는 빈 문자열 통일
        company_name: Optional[str] = tool_input.get("company_name") or None

        # query_summary: 없으면 원본 질문 앞부분 사용
        query_summary: str = tool_input.get("query_summary") or original_query[:100]

        return OrchestratorOutput(
            intent=intent,
            ticker=None,              # ticker는 stock_resolver에서 채움
            company_name=company_name,
            query_summary=query_summary,
            assigned_model=_MODEL_TIER_MAP.get(intent, ModelTier.STANDARD),
        )

    # ── 내부 메서드: ticker 해석 ──────────────────────────────────────────────

    def _resolve_ticker_and_plan(self, orch_output: OrchestratorOutput) -> ExecutionPlan:
        """
        company_name으로 ticker를 해석하고 결과에 따라 분기한다.
        - 확정 → 실행 계획 반환
        - 후보 여러 개 → 사용자 확인 요청 플랜 반환
        - 매칭 없음 → ticker 없이 뉴스 중심으로 실행
        """
        company_name = orch_output.company_name
        try:
            result: ResolveResult = resolve_company(company_name)
        except Exception as e:
            # stock_resolver 오류는 치명적이지 않으므로 ticker 없이 진행
            logger.warning("[Orchestrator] stock_resolver 오류 (%s). ticker 없이 진행합니다.", e)
            return self._make_execution_plan(orch_output)

        if result.is_confirmed:
            # ticker 확정 → OrchestratorOutput에 채워서 실행 계획 수립
            updated = orch_output.model_copy(update={
                "ticker": result.ticker,
                "company_name": result.name,
            })
            logger.info(
                "[Orchestrator] ticker 확정: %s → %s (신뢰도 %.1f%%)",
                company_name, result.ticker, result.confidence,
            )
            return self._make_execution_plan(updated)

        if result.candidates:
            # 후보가 있지만 신뢰도 부족 → 사용자 확인 요청
            logger.info(
                "[Orchestrator] ticker 불명확: '%s' → 후보 %d개 반환",
                company_name, len(result.candidates),
            )
            return self._make_clarification_plan(orch_output, result)

        # 관련 종목을 전혀 찾지 못함 → ticker 없이 뉴스 위주 실행
        logger.info("[Orchestrator] '%s' 종목을 찾을 수 없습니다. ticker 없이 진행합니다.", company_name)
        return self._make_execution_plan(orch_output)

    # ── 내부 메서드: 플랜 생성 ───────────────────────────────────────────────

    def _make_execution_plan(self, orch_output: OrchestratorOutput) -> ExecutionPlan:
        """OrchestratorOutput으로 정상 실행 계획을 생성한다."""
        parallel_groups = _EXECUTION_MAP.get(orch_output.intent, [])
        agents_to_run = [agent for group in parallel_groups for agent in group]
        return ExecutionPlan(
            orchestrator_output=orch_output,
            agents_to_run=agents_to_run,
            parallel_groups=parallel_groups,
        )

    def _make_clarification_plan(
        self,
        orch_output: OrchestratorOutput,
        resolve_result: ResolveResult,
    ) -> ExecutionPlan:
        """ticker 후보를 사용자에게 보여줄 확인 요청 플랜을 생성한다."""
        candidates = [
            ClarificationCandidate(
                ticker=c.ticker,
                name=c.name,
                market=c.market,
                sector=c.sector,
                confidence=c.confidence,
            )
            for c in resolve_result.candidates
        ]
        candidate_names = ", ".join(f"{c.name}({c.ticker})" for c in candidates)
        message = (
            f"'{resolve_result.query}'에 해당하는 종목이 여러 개 있습니다. "
            f"어떤 종목을 말씀하신 건가요?\n후보: {candidate_names}"
        )
        return ExecutionPlan(
            orchestrator_output=orch_output,
            agents_to_run=[],
            parallel_groups=[],
            needs_clarification=True,
            clarification_message=message,
            clarification_candidates=candidates,
        )

    @staticmethod
    def _make_rejection_plan(message: str) -> ExecutionPlan:
        """범위 외 질문에 대한 거절 플랜을 생성한다."""
        orch_output = OrchestratorOutput(
            intent=IntentType.OUT_OF_SCOPE,
            query_summary="범위 외 질문",
            assigned_model=ModelTier.LIGHT,
        )
        return ExecutionPlan(
            orchestrator_output=orch_output,
            agents_to_run=[],
            parallel_groups=[],
            rejection_message=message,
        )

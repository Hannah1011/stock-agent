"""
graph/workflow.py

LangGraph 기반 멀티에이전트 워크플로우.

그래프 구조 (의도별):
  STOCK_QUERY:  orchestrate → [NewsCollector + ChartAnalyst 병렬] → TermExplainer → ReportGenerator
  MARKET_TREND: orchestrate → NewsCollector → TermExplainer → ReportGenerator
  PORTFOLIO:    orchestrate → PortfolioAnalyzer → ReportGenerator
  TERM_QUERY:   orchestrate → TermExplainer → ReportGenerator
  OUT_OF_SCOPE: orchestrate → END (rejection_message 또는 clarification 설정)

병렬 실행:
  STOCK_QUERY의 첫 번째 그룹([NewsCollector, ChartAnalyst])은
  ThreadPoolExecutor로 동시 실행한다. asyncio 대신 ThreadPoolExecutor를 쓰는 이유:
  Streamlit은 자체 이벤트 루프를 사용하므로, asyncio.run()과 충돌을 피하기 위함.
"""

from __future__ import annotations

import logging
import operator
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Annotated, Callable, Optional, TypedDict

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from agents.chart_analyst import ChartAnalystAgent, ChartAnalystError
from agents.news_collector import NewsCollectorAgent, NewsCollectorError
from agents.orchestrator import OrchestratorAgent, OrchestratorError
from agents.portfolio_analyzer import PortfolioAnalyzerAgent, PortfolioAnalyzerError
from agents.report_generator import ReportGeneratorAgent, ReportGeneratorError
from agents.term_explainer import TermExplainerAgent, TermExplainerError
from schemas.models import (
    AgentLog,
    ChartAnalystOutput,
    ExecutionPlan,
    FinalReport,
    IntentType,
    ModelTier,
    NewsCollectorOutput,
    OrchestratorOutput,
    PortfolioAnalyzerOutput,
    TermExplainerOutput,
)

load_dotenv()
logger = logging.getLogger(__name__)

# on_log 콜백을 스레드 안전하게 전달하기 위한 thread-local 저장소.
# Streamlit은 멀티스레드 환경이므로 인스턴스 변수 대신 thread-local을 사용한다.
_thread_local = threading.local()


# ─── 워크플로우 결과 타입 ─────────────────────────────────────────────────────
@dataclass
class WorkflowResult:
    """
    run() / run_with_confirmed_ticker() 의 반환값.

    상태 플래그:
      is_rejection:      범위 외 질문 또는 시스템 오류
      needs_clarification: ticker 후보 선택 필요
      is_success:        최종 리포트 생성 완료
    """
    plan:         ExecutionPlan
    final_report: Optional[FinalReport]      = None
    logs:         list[AgentLog]             = field(default_factory=list)

    @property
    def is_rejection(self) -> bool:
        return bool(self.plan.rejection_message)

    @property
    def needs_clarification(self) -> bool:
        return self.plan.needs_clarification

    @property
    def is_success(self) -> bool:
        return (
            not self.is_rejection
            and not self.needs_clarification
            and self.final_report is not None
        )


# ─── LangGraph 공유 상태 ─────────────────────────────────────────────────────
class AgentState(TypedDict, total=False):
    """
    LangGraph 노드 간에 공유되는 상태.

    logs 필드만 Annotated[list, operator.add]로 병렬 노드에서 자동 병합된다.
    나머지 Optional 필드는 각 노드가 담당 필드만 업데이트한다.
    """
    user_input:       str
    plan:             Optional[ExecutionPlan]
    news_output:      Optional[NewsCollectorOutput]
    chart_output:     Optional[ChartAnalystOutput]
    portfolio_output: Optional[PortfolioAnalyzerOutput]
    term_output:      Optional[TermExplainerOutput]
    final_report:     Optional[FinalReport]
    logs:             Annotated[list[AgentLog], operator.add]
    error:            Optional[str]


# ─── 워크플로우 클래스 ────────────────────────────────────────────────────────
class StockAgentWorkflow:
    """
    멀티에이전트 워크플로우 실행 엔진.

    Streamlit에서는 @st.cache_resource로 싱글턴으로 사용한다.
    """

    def __init__(self) -> None:
        self._orchestrator = OrchestratorAgent()
        self._news         = NewsCollectorAgent()
        self._chart        = ChartAnalystAgent()
        self._portfolio    = PortfolioAnalyzerAgent()
        self._terms        = TermExplainerAgent()
        self._report       = ReportGeneratorAgent()
        self._graph        = self._build_graph()

    # ── 공개 메서드 ──────────────────────────────────────────────────────────

    def run(
        self,
        user_input: str,
        on_log: Optional[Callable[[AgentLog], None]] = None,
    ) -> WorkflowResult:
        """
        사용자 입력으로 전체 워크플로우를 실행한다.

        Args:
            user_input: 사용자 질문 원문
            on_log:     에이전트 로그 생성 시마다 즉시 호출되는 콜백
                        (Streamlit 실시간 UI 업데이트용)

        Returns:
            WorkflowResult — plan, final_report, logs 포함
        """
        _thread_local.on_log = on_log
        try:
            final_state = self._graph.invoke(self._initial_state(user_input))
            return self._to_result(final_state)
        except Exception as e:
            logger.error("[Workflow] 예상치 못한 오류: %s", e)
            return self._make_error_result(str(e), [])
        finally:
            _thread_local.on_log = None

    def run_with_confirmed_ticker(
        self,
        user_input: str,
        confirmed_ticker: str,
        confirmed_name: str,
        on_log: Optional[Callable[[AgentLog], None]] = None,
    ) -> WorkflowResult:
        """
        사용자가 ticker 후보를 선택한 후 호출되는 진입점.
        Orchestrator의 ticker 해석 단계를 건너뛰고 바로 실행 계획을 수립한다.
        """
        _thread_local.on_log = on_log
        try:
            plan = self._orchestrator.run_with_ticker(
                user_input, confirmed_ticker, confirmed_name
            )
            orch_log = self._make_log(
                "Orchestrator", "done",
                f"ticker 확정 → {confirmed_ticker} ({confirmed_name})",
                ModelTier.STANDARD.value,
            )
            initial = self._initial_state(user_input)
            initial["plan"] = plan
            initial["logs"] = [orch_log]

            final_state = self._graph.invoke(initial)
            return self._to_result(final_state)
        except OrchestratorError as e:
            err_log = self._make_log("Orchestrator", "error", error=str(e))
            return self._make_error_result(str(e), [err_log])
        except Exception as e:
            logger.error("[Workflow] 예상치 못한 오류 (confirmed ticker): %s", e)
            return self._make_error_result(str(e), [])
        finally:
            _thread_local.on_log = None

    # ── 내부: 그래프 빌드 ────────────────────────────────────────────────────

    def _build_graph(self):
        """
        LangGraph StateGraph를 구성하고 컴파일한다.

        그래프 엣지 요약:
          START → orchestrate
          orchestrate --조건부--> {
            "stock":     news_chart_parallel
            "market":    news_only
            "portfolio": portfolio
            "term":      term_explainer
            "end":       END
          }
          news_chart_parallel → term_explainer
          news_only           → term_explainer
          term_explainer      → report_generator
          portfolio           → report_generator
          report_generator    → END
        """
        g = StateGraph(AgentState)

        g.add_node("orchestrate",         self._orchestrate_node)
        g.add_node("news_chart_parallel", self._news_chart_parallel_node)
        g.add_node("news_only",           self._news_only_node)
        g.add_node("portfolio",           self._portfolio_node)
        g.add_node("term_explainer",      self._term_node)
        g.add_node("report_generator",    self._report_node)

        g.add_edge(START, "orchestrate")
        g.add_conditional_edges(
            "orchestrate",
            self._route_after_orchestrate,
            {
                "stock":     "news_chart_parallel",
                "market":    "news_only",
                "portfolio": "portfolio",
                "term":      "term_explainer",
                "end":       END,
            },
        )
        g.add_edge("news_chart_parallel", "term_explainer")
        g.add_edge("news_only",           "term_explainer")
        g.add_edge("term_explainer",      "report_generator")
        g.add_edge("portfolio",           "report_generator")
        g.add_edge("report_generator",    END)

        return g.compile()

    # ── 내부: 라우팅 ─────────────────────────────────────────────────────────

    @staticmethod
    def _route_after_orchestrate(state: AgentState) -> str:
        """
        orchestrate 노드 완료 후 다음 노드를 결정한다.

        rejection / needs_clarification / error → "end" (LangGraph END로 이동)
        intent별 → "stock" | "market" | "portfolio" | "term"
        """
        plan = state.get("plan")
        if not plan:
            return "end"
        if plan.rejection_message or plan.needs_clarification or state.get("error"):
            return "end"

        routing = {
            IntentType.STOCK_QUERY:  "stock",
            IntentType.MARKET_TREND: "market",
            IntentType.PORTFOLIO:    "portfolio",
            IntentType.TERM_QUERY:   "term",
            IntentType.OUT_OF_SCOPE: "end",
        }
        return routing.get(plan.orchestrator_output.intent, "end")

    # ── 내부: 노드 함수 ──────────────────────────────────────────────────────

    def _orchestrate_node(self, state: AgentState) -> dict:
        """
        Orchestrator 노드: 의도 분류·ticker 해석·실행 계획 수립.

        run_with_confirmed_ticker()로 진입한 경우 plan이 이미 설정되어 있으므로
        Orchestrator 재실행 없이 즉시 반환한다.
        """
        # confirmed ticker 흐름: plan이 이미 설정됨 → 건너뜀
        if state.get("plan") is not None:
            return {}

        run_log = self._make_log(
            "Orchestrator", "running", "의도 분류 중…", ModelTier.STANDARD.value
        )
        try:
            plan     = self._orchestrator.run(state["user_input"])
            done_log = self._make_log(
                "Orchestrator", "done",
                f"intent={plan.orchestrator_output.intent.value}"
                + (f" | {plan.orchestrator_output.company_name}" if plan.orchestrator_output.company_name else ""),
                ModelTier.STANDARD.value,
            )
            return {"plan": plan, "logs": [run_log, done_log]}
        except OrchestratorError as e:
            err_log = self._make_log("Orchestrator", "error", error=str(e))
            logger.error("[Workflow] Orchestrator 오류: %s", e)
            return {"error": str(e), "logs": [run_log, err_log]}
        except Exception as e:
            err_log = self._make_log("Orchestrator", "error", error=f"예상치 못한 오류: {e}")
            logger.error("[Workflow] Orchestrator 예외: %s", e)
            return {"error": str(e), "logs": [run_log, err_log]}

    def _news_chart_parallel_node(self, state: AgentState) -> dict:
        """
        NewsCollector + ChartAnalyst를 ThreadPoolExecutor로 동시 실행한다.
        STOCK_QUERY 전용.
        두 에이전트가 독립적이므로 진정한 병렬 실행이 가능하다.
        """
        plan       = state["plan"]
        news_log   = self._make_log("NewsCollector", "running", "관련 뉴스 수집 중…", ModelTier.LIGHT.value)
        chart_log  = self._make_log("ChartAnalyst",  "running", "차트 지표 분석 중…", ModelTier.LIGHT.value)
        result_logs = [news_log, chart_log]

        news_output:  Optional[NewsCollectorOutput] = None
        chart_output: Optional[ChartAnalystOutput]  = None

        with ThreadPoolExecutor(max_workers=2) as executor:
            news_future  = executor.submit(self._news.run,  plan)
            chart_future = executor.submit(self._chart.run, plan)

            try:
                news_output = news_future.result()
                result_logs.append(self._make_log(
                    "NewsCollector", "done",
                    f"{len(news_output.news_items)}건 수집 완료",
                    ModelTier.LIGHT.value,
                ))
            except Exception as e:
                logger.error("[Workflow] NewsCollector 오류: %s", e)
                result_logs.append(self._make_log("NewsCollector", "error", error=str(e)))

            try:
                chart_output = chart_future.result()
                preview = (
                    f"{chart_output.chart_data.ticker} 분석 완료 "
                    f"(RSI: {chart_output.chart_data.rsi})"
                    if chart_output else "데이터 부족 — 건너뜀"
                )
                result_logs.append(self._make_log(
                    "ChartAnalyst", "done", preview, ModelTier.LIGHT.value
                ))
            except Exception as e:
                logger.error("[Workflow] ChartAnalyst 오류: %s", e)
                result_logs.append(self._make_log("ChartAnalyst", "error", error=str(e)))

        return {
            "news_output":  news_output,
            "chart_output": chart_output,
            "logs":         result_logs,
        }

    def _news_only_node(self, state: AgentState) -> dict:
        """NewsCollector만 실행한다. MARKET_TREND 전용."""
        plan    = state["plan"]
        run_log = self._make_log("NewsCollector", "running", "시장 뉴스 수집 중…", ModelTier.LIGHT.value)
        try:
            news_output = self._news.run(plan)
            done_log = self._make_log(
                "NewsCollector", "done",
                f"{len(news_output.news_items)}건 수집 완료",
                ModelTier.LIGHT.value,
            )
            return {"news_output": news_output, "logs": [run_log, done_log]}
        except (NewsCollectorError, Exception) as e:
            err_log = self._make_log("NewsCollector", "error", error=str(e))
            logger.error("[Workflow] NewsCollector 오류: %s", e)
            return {"logs": [run_log, err_log]}

    def _portfolio_node(self, state: AgentState) -> dict:
        """PortfolioAnalyzer를 실행한다. PORTFOLIO 전용."""
        plan    = state["plan"]
        run_log = self._make_log(
            "PortfolioAnalyzer", "running", "포트폴리오 분석 중…", ModelTier.STANDARD.value
        )
        try:
            portfolio_output = self._portfolio.run(plan)
            done_log = self._make_log(
                "PortfolioAnalyzer", "done",
                f"종목 {len(portfolio_output.holdings)}개 | "
                f"총 수익률 {portfolio_output.total_profit_loss_pct:+.2f}%",
                ModelTier.STANDARD.value,
            )
            return {"portfolio_output": portfolio_output, "logs": [run_log, done_log]}
        except (PortfolioAnalyzerError, Exception) as e:
            err_log = self._make_log("PortfolioAnalyzer", "error", error=str(e))
            logger.error("[Workflow] PortfolioAnalyzer 오류: %s", e)
            return {"logs": [run_log, err_log]}

    def _term_node(self, state: AgentState) -> dict:
        """TermExplainer를 실행한다."""
        plan    = state["plan"]
        run_log = self._make_log("TermExplainer", "running", "경제 용어 분석 중…", ModelTier.LIGHT.value)
        try:
            term_output = self._terms.run(plan, news_output=state.get("news_output"))
            done_log = self._make_log(
                "TermExplainer", "done",
                f"용어 {len(term_output.terms_explained)}개 설명 완료",
                ModelTier.LIGHT.value,
            )
            return {"term_output": term_output, "logs": [run_log, done_log]}
        except (TermExplainerError, Exception) as e:
            err_log = self._make_log("TermExplainer", "error", error=str(e))
            logger.error("[Workflow] TermExplainer 오류: %s", e)
            return {"logs": [run_log, err_log]}

    def _report_node(self, state: AgentState) -> dict:
        """ReportGenerator를 실행한다. 모든 에이전트 출력을 최종 리포트로 합성한다."""
        plan    = state["plan"]
        run_log = self._make_log("ReportGenerator", "running", "리포트 작성 중…", ModelTier.STANDARD.value)
        try:
            final_report = self._report.run(
                plan,
                news_output=state.get("news_output"),
                chart_output=state.get("chart_output"),
                portfolio_output=state.get("portfolio_output"),
                term_output=state.get("term_output"),
            )
            summary = final_report.one_line_summary or "완료"
            done_log = self._make_log(
                "ReportGenerator", "done",
                summary[:70] + ("…" if len(summary) > 70 else ""),
                ModelTier.STANDARD.value,
            )
            return {"final_report": final_report, "logs": [run_log, done_log]}
        except (ReportGeneratorError, Exception) as e:
            err_log = self._make_log("ReportGenerator", "error", error=str(e))
            logger.error("[Workflow] ReportGenerator 오류: %s", e)
            return {"logs": [run_log, err_log]}

    # ── 내부: 헬퍼 ───────────────────────────────────────────────────────────

    def _make_log(
        self,
        agent: str,
        status: str,
        action: str = "",
        model_used: Optional[str] = None,
        error: Optional[str] = None,
    ) -> AgentLog:
        """
        AgentLog를 생성하고 on_log 콜백을 즉시 호출한다.
        thread-local에서 콜백을 가져오므로 스레드 안전하다.
        """
        log = AgentLog(
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent=agent,
            status=status,
            action=action,
            model_used=model_used,
            error=error,
        )
        callback = getattr(_thread_local, "on_log", None)
        if callback:
            try:
                callback(log)
            except Exception as cb_err:
                logger.debug("[Workflow] on_log 콜백 오류 (무시): %s", cb_err)
        return log

    @staticmethod
    def _initial_state(user_input: str) -> AgentState:
        """그래프 invoke()에 전달할 초기 상태를 반환한다."""
        return {
            "user_input":       user_input,
            "plan":             None,
            "news_output":      None,
            "chart_output":     None,
            "portfolio_output": None,
            "term_output":      None,
            "final_report":     None,
            "logs":             [],
            "error":            None,
        }

    @staticmethod
    def _to_result(state: AgentState) -> WorkflowResult:
        """LangGraph 최종 상태를 WorkflowResult로 변환한다."""
        return WorkflowResult(
            plan=state["plan"],
            final_report=state.get("final_report"),
            logs=state.get("logs", []),
        )

    @staticmethod
    def _make_error_result(error_msg: str, logs: list[AgentLog]) -> WorkflowResult:
        """복구 불가능한 오류 발생 시 거절 플랜을 담은 WorkflowResult를 반환한다."""
        error_plan = ExecutionPlan(
            orchestrator_output=OrchestratorOutput(
                intent=IntentType.OUT_OF_SCOPE,
                query_summary="시스템 오류",
                assigned_model=ModelTier.LIGHT,
            ),
            agents_to_run=[],
            parallel_groups=[],
            rejection_message=f"시스템 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.\n상세: {error_msg}",
        )
        return WorkflowResult(plan=error_plan, logs=logs)

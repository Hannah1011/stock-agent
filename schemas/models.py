from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class IntentType(str, Enum):
    STOCK_QUERY = "A"      # 종목 질문
    PORTFOLIO = "B"        # 포트폴리오 분석
    TERM_QUERY = "C"       # 용어 질문
    MARKET_TREND = "D"     # 시장 동향
    OUT_OF_SCOPE = "E"     # 범위 외


class ModelTier(str, Enum):
    LIGHT = "claude-haiku-4-5-20251001"
    STANDARD = "claude-sonnet-4-6"


class OrchestratorOutput(BaseModel):
    intent: IntentType
    ticker: Optional[str] = None          # 예: "005930.KS"
    company_name: Optional[str] = None    # 예: "삼성전자"
    query_summary: str
    assigned_model: ModelTier


class NewsItem(BaseModel):
    title: str
    summary: str
    source: str
    published_at: str
    url: Optional[str] = None
    relevance_score: float = Field(ge=0, le=1)
    impact: str  # "HIGH" | "MEDIUM" | "LOW"


class NewsCollectorOutput(BaseModel):
    ticker: Optional[str]
    company_name: Optional[str]
    news_items: List[NewsItem]
    keywords_used: List[str]


class ChartData(BaseModel):
    ticker: str
    current_price: float
    change_pct: float
    rsi: Optional[float] = None
    ma5: Optional[float] = None
    ma20: Optional[float] = None
    ma60: Optional[float] = None
    volume_signal: str  # "NORMAL" | "HIGH" | "VERY_HIGH"
    trend_summary: str


class ChartAnalystOutput(BaseModel):
    chart_data: ChartData
    plain_explanation: str


class TermAnnotation(BaseModel):
    term: str
    explanation: str


class TermExplainerOutput(BaseModel):
    original_text: str
    annotated_text: str
    terms_explained: List[TermAnnotation]


class HoldingStatus(BaseModel):
    ticker: str
    name: str
    shares: int
    avg_price: float
    current_price: float
    profit_loss_pct: float
    eval_amount: float


class RiskScore(BaseModel):
    sector_concentration: float  # 0–1
    volatility_level: str        # "LOW" | "MEDIUM" | "HIGH"
    diversification_advice: str


class PortfolioAnalyzerOutput(BaseModel):
    holdings: List[HoldingStatus]
    total_eval: float
    total_profit_loss_pct: float
    sector_distribution: dict
    risk: RiskScore


class FinalReport(BaseModel):
    one_line_summary: str
    news_section: Optional[str] = None
    chart_section: Optional[str] = None
    portfolio_section: Optional[str] = None
    risk_alerts: List[str] = Field(default_factory=list)
    terms_glossary: List[TermAnnotation] = Field(default_factory=list)
    disclaimer: str = "본 정보는 투자 판단을 위한 참고 자료이며, 투자 권유가 아닙니다."
    full_report_md: str


class ClarificationCandidate(BaseModel):
    """사용자에게 종목 확인을 요청할 때 보여줄 후보 항목."""
    ticker: str
    name: str
    market: str   # "KOSPI" | "KOSDAQ"
    sector: str
    confidence: float


class ExecutionPlan(BaseModel):
    """
    Orchestrator가 수립한 실행 계획.

    parallel_groups: 각 내부 리스트가 하나의 병렬 실행 그룹.
      그룹 간 순서 = 실행 순서.
      예) [["NewsCollector","ChartAnalyst"], ["TermExplainer"], ["ReportGenerator"]]

    상태 플래그 (셋 중 하나만 True):
      - 정상 실행: rejection_message=None, needs_clarification=False
      - 범위 외:   rejection_message 설정
      - ticker 불명확: needs_clarification=True, clarification_candidates 설정
    """
    orchestrator_output: OrchestratorOutput
    agents_to_run: List[str]           # 평탄화된 실행 목록 (모니터링용)
    parallel_groups: List[List[str]]   # 실제 실행 그룹

    # 범위 외 질문일 때 설정
    rejection_message: Optional[str] = None

    # ticker가 불명확할 때 설정
    needs_clarification: bool = False
    clarification_message: Optional[str] = None
    clarification_candidates: List[ClarificationCandidate] = Field(default_factory=list)


class AgentLog(BaseModel):
    timestamp: str
    agent: str          # "Orchestrator" | "News Collector" | ...
    status: str         # "running" | "done" | "error" | "skipped"
    action: str
    result_preview: Optional[str] = None
    model_used: Optional[str] = None
    error: Optional[str] = None

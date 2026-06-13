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


class AgentLog(BaseModel):
    timestamp: str
    agent: str          # "Orchestrator" | "News Collector" | ...
    status: str         # "running" | "done" | "error" | "skipped"
    action: str
    result_preview: Optional[str] = None
    model_used: Optional[str] = None
    error: Optional[str] = None

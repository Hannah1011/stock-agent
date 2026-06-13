"""
tools/stock_resolver.py

KRX 전체 종목 목록을 pyKrx로 조회해 자연어 회사명을 ticker로 변환한다.

해석 흐름:
  입력 → 정식 종목명 정확 일치 → ticker 확정 (is_confirmed=True)
       → 부분 일치·별칭        → 후보 반환   (is_confirmed=False)
       → 관련 종목 없음        → 결과 없음    (candidates=[])

종목 목록은 로컬 큐레이션 목록과 portfolio.json을 기본으로 구성하고,
Yahoo Finance에서 섹터를 보완해 일별 JSON 캐시에 저장한다.
pyKrx 조회는 STOCK_CATALOG_PROVIDER=pykrx로 명시한 경우에만 시도한다.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

# ─── 경로 ───────────────────────────────────────────────────────────────────
_CACHE_DIR  = os.path.join(os.path.dirname(__file__), ".cache")
_CACHE_FILE = os.path.join(_CACHE_DIR, "krx_{date}.json")
_PORTFOLIO_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "portfolio.json")

# ─── 후보 검색 설정 ─────────────────────────────────────────────────────────
CONFIDENCE_SUGGEST = 50   # 이상: 후보로 표시 / 미만: 무시
MAX_CANDIDATES     = 3    # 사용자에게 보여줄 최대 후보 수


# ─── 결과 데이터 타입 ────────────────────────────────────────────────────────
@dataclass
class Candidate:
    ticker: str       # "005930.KS"
    name: str         # "삼성전자"
    market: str       # "KOSPI" | "KOSDAQ"
    sector: str       # "전기·전자"
    confidence: float # 0~100


@dataclass
class ResolveResult:
    """
    종목 해석 결과.

    is_confirmed=True  → ticker/name이 확정됨, Orchestrator가 바로 사용 가능
    is_confirmed=False → candidates를 사용자에게 보여주고 선택 요청
    candidates=[]      → 관련 종목을 찾지 못함
    """
    query: str
    ticker: Optional[str]
    name: Optional[str]
    confidence: float
    is_confirmed: bool
    candidates: list[Candidate] = field(default_factory=list)


# ─── 캐시 관리 ──────────────────────────────────────────────────────────────
def _last_business_day() -> str:
    """캐시 기준일을 반환한다. 주말이면 가장 최근 금요일을 사용한다."""
    d = date.today()
    if d.weekday() == 5:    # 토요일
        d -= timedelta(days=1)
    elif d.weekday() == 6:  # 일요일
        d -= timedelta(days=2)
    return d.strftime("%Y%m%d")


def _cleanup_old_caches(keep_date: str) -> None:
    """보관 날짜 이외의 오래된 캐시 파일을 삭제한다."""
    if not os.path.isdir(_CACHE_DIR):
        return
    for fname in os.listdir(_CACHE_DIR):
        if fname.startswith("krx_") and fname.endswith(".json"):
            date_part = fname[4:-5]  # "krx_20260613.json" → "20260613"
            if date_part != keep_date:
                try:
                    os.remove(os.path.join(_CACHE_DIR, fname))
                except OSError as e:
                    logger.warning("[stock_resolver] 구 캐시 삭제 실패 (%s): %s", fname, e)


def _is_valid_cache(data: dict) -> bool:
    """종목과 이름 매핑이 모두 있는 캐시만 유효한 것으로 본다."""
    return bool(data.get("stocks") and data.get("name_to_ticker"))


def _build_fallback_catalog(biz_date: str) -> dict:
    """
    KRX 조회 실패 시 앱 내 큐레이션 종목과 portfolio.json으로 최소 목록을 만든다.
    """
    from tools.keyword_map import TICKER_KEYWORD_MAP
    from tools.stock_api import get_company_sectors

    ticker_to_name = {
        ticker: keywords[0]
        for ticker, keywords in TICKER_KEYWORD_MAP.items()
        if keywords
    }

    portfolio_path = os.getenv("PORTFOLIO_PATH", _PORTFOLIO_FILE)
    try:
        with open(portfolio_path, encoding="utf-8") as f:
            for holding in json.load(f).get("holdings", []):
                ticker = holding.get("ticker")
                name = holding.get("name")
                if ticker and name:
                    ticker_to_name[ticker] = name
    except (OSError, json.JSONDecodeError):
        pass

    sectors = get_company_sectors(list(ticker_to_name))
    stocks = [
        {
            "ticker": ticker,
            "name": name,
            "market": "KOSPI" if ticker.endswith(".KS") else "KOSDAQ",
            "sector": sectors.get(ticker, ""),
        }
        for ticker, name in ticker_to_name.items()
    ]
    return {
        "date": biz_date,
        "generated_at": date.today().isoformat(),
        "stocks": stocks,
        "name_to_ticker": {stock["name"]: stock["ticker"] for stock in stocks},
        "source": "local+yahoo",
    }


def _fetch_from_krx(biz_date: str) -> dict:
    """
    KOSPI + KOSDAQ 전체 종목을 pyKrx로 조회해 구조화된 딕셔너리로 반환한다.

    반환 구조:
    {
      "date": "20260613",
      "stocks": [
        {"ticker": "005930.KS", "name": "삼성전자", "market": "KOSPI", "sector": "전기·전자"},
        ...
      ],
      "name_to_ticker": {"삼성전자": "005930.KS", ...}
    }
    """
    from pykrx import stock as krx

    stocks: list[dict] = []
    name_to_ticker: dict[str, str] = {}

    for market, suffix in [("KOSPI", ".KS"), ("KOSDAQ", ".KQ")]:
        # 종목 코드 목록 조회
        try:
            raw_tickers = krx.get_market_ticker_list(biz_date, market=market)
        except Exception as e:
            logger.debug("[stock_resolver] %s 종목 목록 조회 실패: %s", market, e)
            continue
        if not raw_tickers:
            logger.debug("[stock_resolver] %s 종목 목록이 비어 있습니다.", market)
            continue

        # 종목 목록이 있을 때만 섹터 정보를 조회한다.
        sector_df = None
        try:
            sector_df = krx.get_market_sector_classifications(biz_date, market)
        except Exception as e:
            logger.debug("[stock_resolver] %s 섹터 조회 실패: %s", market, e)

        for raw_ticker in raw_tickers:
            full_ticker = raw_ticker + suffix

            try:
                name = krx.get_market_ticker_name(raw_ticker)
            except Exception:
                name = raw_ticker  # 이름 조회 실패 시 코드 자체를 이름으로 사용

            sector = _extract_sector(sector_df, raw_ticker)

            stocks.append({
                "ticker": full_ticker,
                "name": name,
                "market": market,
                "sector": sector,
            })
            # 이름 중복 시 먼저 들어온 항목(KOSPI) 우선
            if name not in name_to_ticker:
                name_to_ticker[name] = full_ticker

    return {"date": biz_date, "stocks": stocks, "name_to_ticker": name_to_ticker}


def _extract_sector(sector_df, raw_ticker: str) -> str:
    """DataFrame에서 종목의 업종명을 안전하게 추출한다."""
    if sector_df is None:
        return ""
    try:
        if raw_ticker not in sector_df.index:
            return ""
        row = sector_df.loc[raw_ticker]
        for col in ["업종명", "Sector", "sector"]:
            if col in sector_df.columns:
                return str(row[col])
    except Exception:
        pass
    return ""


def _load_cache() -> dict:
    """일별 캐시를 로드한다. 없거나 손상됐으면 pyKrx에서 새로 빌드한다."""
    biz_date = _last_business_day()
    path = _CACHE_FILE.format(date=biz_date)

    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                cached = json.load(f)
            if _is_valid_cache(cached):
                return cached
            logger.warning("[stock_resolver] 빈 KRX 캐시를 무시하고 다시 조회합니다.")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("[stock_resolver] 캐시 손상, 재빌드합니다: %s", e)

    provider = os.getenv("STOCK_CATALOG_PROVIDER", "local").lower()
    if provider == "pykrx":
        logger.info("[stock_resolver] pyKrx 종목 목록 갱신 중 (기준 거래일: %s)…", biz_date)
        data = _fetch_from_krx(biz_date)
        if not _is_valid_cache(data):
            logger.info("[stock_resolver] pyKrx 조회 불가, 로컬+Yahoo 목록을 사용합니다.")
            data = _build_fallback_catalog(biz_date)
    else:
        logger.info("[stock_resolver] 로컬+Yahoo 종목 목록 생성 중 (기준 거래일: %s)…", biz_date)
        data = _build_fallback_catalog(biz_date)

    os.makedirs(_CACHE_DIR, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        _cleanup_old_caches(biz_date)
        logger.info("[stock_resolver] 캐시 저장 완료 (%d 종목)", len(data["stocks"]))
    except OSError as e:
        logger.warning("[stock_resolver] 캐시 저장 실패 (메모리 캐시로 진행): %s", e)

    return data


# 모듈 수준 lazy 캐시 — 첫 resolve_company() 호출 시 초기화
_cache: Optional[dict] = None


def _get_cache() -> dict:
    global _cache
    if _cache is None:
        _cache = _load_cache()
    return _cache


# ─── 공개 API ────────────────────────────────────────────────────────────────
def _is_exact_name_mentioned(name: str, text: str) -> bool:
    """정식 종목명이 원문에서 독립된 이름으로 언급됐는지 확인한다."""
    particle_or_boundary = (
        r"(?=$|[^0-9A-Za-z가-힣]|"
        r"은|는|이|가|을|를|의|도|만|와|과|로|으로|에서|에게)"
    )
    pattern = rf"(?<![0-9A-Za-z가-힣]){re.escape(name)}{particle_or_boundary}"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def resolve_company(query: str, original_query: Optional[str] = None) -> ResolveResult:
    """
    자연어 회사명·별칭으로 KRX ticker를 찾는다.

    Args:
        query: Orchestrator가 추출한 회사명 또는 축약어 (예: "삼전", "에코")
        original_query: 사용자의 원문 질문. 정식 종목명이 원문에 직접 등장한
            경우에만 자동 확정하기 위해 사용한다.

    Returns:
        ResolveResult:
          is_confirmed=True  → ticker 확정 사용 가능
          is_confirmed=False → candidates를 사용자에게 보여 선택 요청

    Example:
        result = resolve_company("삼전")
        if result.is_confirmed:
            use(result.ticker)       # "005930.KS"
        else:
            show_candidates(result.candidates)
    """
    cache = _get_cache()
    name_to_ticker: dict[str, str] = cache.get("name_to_ticker", {})
    name_list = list(name_to_ticker.keys())

    if not name_list:
        logger.warning("[stock_resolver] 사용할 수 있는 종목 목록이 없습니다.")
        return ResolveResult(
            query=query, ticker=None, name=None,
            confidence=0.0, is_confirmed=False,
        )

    # LLM이 "에코"를 "에코프로"로 임의 확장할 수 있으므로, 자동 확정은
    # 원문에 등록된 정식 종목명이 직접 등장한 경우에만 허용한다.
    source = original_query or query
    exact_names = [
        name for name in name_list
        if _is_exact_name_mentioned(name, source)
    ]
    if exact_names:
        # 이름이 겹치는 종목이 함께 감지된 경우 가장 구체적인 이름을 우선한다.
        exact_name = max(exact_names, key=lambda name: len(name.replace(" ", "")))
        ticker = name_to_ticker[exact_name]
        return ResolveResult(
            query=query,
            ticker=ticker,
            name=exact_name,
            confidence=100.0,
            is_confirmed=True,
        )

    # rapidfuzz WRatio: 부분 일치·순서 모두 고려하는 복합 스코어
    matches = process.extract(
        query, name_list,
        scorer=fuzz.WRatio,
        limit=MAX_CANDIDATES,
    )

    # 하한 신뢰도 필터
    valid = [(name, score) for name, score, _ in matches if score >= CONFIDENCE_SUGGEST]

    if not valid:
        return ResolveResult(
            query=query, ticker=None, name=None,
            confidence=0.0, is_confirmed=False,
        )

    # 상세 정보 조회를 위한 ticker→stock 맵 (조회 효율을 위해 1회 빌드)
    ticker_detail: dict[str, dict] = {s["ticker"]: s for s in cache.get("stocks", [])}

    candidates: list[Candidate] = []
    for name, score in valid:
        t = name_to_ticker[name]
        detail = ticker_detail.get(t, {})
        candidates.append(Candidate(
            ticker=t,
            name=name,
            market=detail.get("market", ""),
            sector=detail.get("sector", ""),
            confidence=round(score, 1),
        ))

    return ResolveResult(
        query=query,
        ticker=None,
        name=None,
        confidence=candidates[0].confidence,
        is_confirmed=False,
        candidates=candidates,
    )


def get_sector_by_ticker(ticker: str) -> Optional[str]:
    """ticker로 섹터명을 반환한다. 캐시에 없으면 None."""
    for stock in _get_cache().get("stocks", []):
        if stock["ticker"] == ticker:
            return stock["sector"] or None
    return None


def get_all_stocks() -> list[dict]:
    """전체 종목 목록을 반환한다 (모니터링·디버깅용)."""
    return _get_cache().get("stocks", [])


def refresh_cache() -> None:
    """강제로 캐시를 갱신한다 (수동 호출 또는 테스트용)."""
    global _cache
    _cache = None
    biz_date = _last_business_day()
    path = _CACHE_FILE.format(date=biz_date)
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
    _cache = _load_cache()

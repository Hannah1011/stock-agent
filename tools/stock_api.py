import yfinance as yf
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

_SECTOR_FALLBACKS = {
    "086520.KQ": "Basic Materials",  # 에코프로
}


def get_company_sector(ticker: str) -> Optional[str]:
    """Yahoo Finance에서 종목 섹터를 반환한다."""
    try:
        info = yf.Ticker(ticker).info
        return info.get("sector") or info.get("industry") or _SECTOR_FALLBACKS.get(ticker)
    except Exception:
        return _SECTOR_FALLBACKS.get(ticker)


def get_company_sectors(tickers: list[str], max_workers: int = 6) -> dict[str, str]:
    """여러 종목의 Yahoo Finance 섹터를 병렬 조회한다."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        sectors = executor.map(get_company_sector, tickers)
    return {
        ticker: sector
        for ticker, sector in zip(tickers, sectors)
        if sector
    }


def get_current_price(ticker: str) -> Optional[dict]:
    """현재가, 전일 대비 등락률, 52주 고저 반환."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        current = info.get("currentPrice") or info.get("regularMarketPrice")
        prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
        week52_high = info.get("fiftyTwoWeekHigh")
        week52_low = info.get("fiftyTwoWeekLow")

        if current is None or prev_close is None:
            return None

        change_pct = ((current - prev_close) / prev_close) * 100
        return {
            "ticker": ticker,
            "current_price": current,
            "prev_close": prev_close,
            "change_pct": round(change_pct, 2),
            "week52_high": week52_high,
            "week52_low": week52_low,
        }
    except Exception:
        return None


def get_ohlcv(ticker: str, period: str = "3mo") -> Optional[pd.DataFrame]:
    """OHLCV 데이터를 DataFrame으로 반환."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, float("inf"))
    return 100 - (100 / (1 + rs))


def get_technical_indicators(ticker: str) -> Optional[dict]:
    """
    이동평균(5/20/60일), RSI(14일), 볼린저 밴드, 거래량 신호를 계산해 반환.
    데이터 부족 시 None 반환.
    """
    df = get_ohlcv(ticker, period="6mo")
    if df is None or len(df) < 20:
        return None

    close = df["Close"]
    volume = df["Volume"]

    ma5 = close.rolling(5).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1] if len(df) >= 60 else None

    rsi = calc_rsi(close).iloc[-1]

    # 볼린저 밴드 (20일 기준)
    std20 = close.rolling(20).std().iloc[-1]
    upper_band = ma20 + 2 * std20
    lower_band = ma20 - 2 * std20

    # 거래량 신호: 최근 5일 평균 vs 20일 평균
    avg_vol_5 = volume.rolling(5).mean().iloc[-1]
    avg_vol_20 = volume.rolling(20).mean().iloc[-1]
    vol_ratio = avg_vol_5 / avg_vol_20 if avg_vol_20 > 0 else 1.0
    if vol_ratio >= 2.0:
        volume_signal = "VERY_HIGH"
    elif vol_ratio >= 1.3:
        volume_signal = "HIGH"
    else:
        volume_signal = "NORMAL"

    current = close.iloc[-1]
    prev = close.iloc[-2]
    change_pct = ((current - prev) / prev) * 100

    return {
        "ticker": ticker,
        "current_price": round(current, 2),
        "change_pct": round(change_pct, 2),
        "ma5": round(ma5, 2) if ma5 else None,
        "ma20": round(ma20, 2) if ma20 else None,
        "ma60": round(ma60, 2) if ma60 else None,
        "rsi": round(rsi, 2) if not pd.isna(rsi) else None,
        "upper_band": round(upper_band, 2),
        "lower_band": round(lower_band, 2),
        "volume_signal": volume_signal,
    }

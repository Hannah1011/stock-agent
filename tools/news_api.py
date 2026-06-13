import os
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Optional
import requests


def fetch_newsapi(query: str, page_size: int = 5) -> list[dict]:
    """NewsAPI에서 한국어 경제 뉴스를 가져온다."""
    api_key = os.getenv("NEWS_API_KEY", "")
    if not api_key:
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "ko",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        return [
            {
                "title": a.get("title", ""),
                "summary": a.get("description", "") or "",
                "source": a.get("source", {}).get("name", "NewsAPI"),
                "published_at": a.get("publishedAt", ""),
                "url": a.get("url", ""),
            }
            for a in articles
        ]
    except Exception:
        return []


def fetch_naver_rss(query: str, display: int = 5) -> list[dict]:
    """네이버 뉴스 검색 RSS로 기사를 가져온다 (NewsAPI 폴백)."""
    encoded_query = urllib.parse.quote(query)
    url = f"https://news.naver.com/search/results.nhn?query={encoded_query}&field=0&where=news"

    # 네이버 공식 뉴스 검색 RSS 엔드포인트
    rss_url = (
        f"https://rss.naver.com/main/search.nhn"
        f"?type=news&query={encoded_query}&display={display}"
    )
    try:
        req = urllib.request.Request(
            rss_url,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read()
        root = ET.fromstring(raw)
        items = root.findall(".//item")
        results = []
        for item in items[:display]:
            title = _strip_cdata(_get_text(item, "title"))
            desc = _strip_cdata(_get_text(item, "description"))
            pub_date = _get_text(item, "pubDate")
            link = _get_text(item, "link")
            results.append(
                {
                    "title": title,
                    "summary": desc,
                    "source": "네이버뉴스",
                    "published_at": pub_date,
                    "url": link,
                }
            )
        return results
    except Exception:
        return []


def fetch_news(keywords: list[str], max_per_keyword: int = 3) -> list[dict]:
    """
    키워드 목록으로 뉴스를 수집한다.
    NewsAPI 우선 → 실패 시 네이버 RSS 폴백.
    중복 제목 제거 후 반환.
    """
    seen_titles: set[str] = set()
    results: list[dict] = []

    for kw in keywords:
        articles = fetch_newsapi(kw, page_size=max_per_keyword)
        if not articles:
            articles = fetch_naver_rss(kw, display=max_per_keyword)

        for art in articles:
            title = art.get("title", "").strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                results.append(art)

    return results


def _get_text(element, tag: str) -> str:
    node = element.find(tag)
    return node.text.strip() if node is not None and node.text else ""


def _strip_cdata(text: str) -> str:
    text = text.replace("<![CDATA[", "").replace("]]>", "")
    # 간단한 HTML 태그 제거
    import re
    return re.sub(r"<[^>]+>", "", text).strip()

"""
rag/term_crawler.py

RAG 용어 사전에 없는 경제 용어를 동적으로 조회·설명하는 폴백 모듈.

흐름:
  1. dynamic_terms.json 캐시 확인 → 있으면 즉시 반환
  2. 위키피디아 한국어판 REST API로 원문 정의 조회 (키 불필요, 무료)
  3. 위키피디아에 없으면 Claude의 학습 지식으로 바로 생성
  4. Claude Haiku가 초보자 친화적인 설명으로 변환
  5. dynamic_terms.json에 캐시 저장 → 이후 RAG(BM25·ChromaDB)에서 바로 검색 가능

위키피디아를 사용하는 이유:
  - Claude만 쓰면 용어 정의가 일관성이 없을 수 있음
  - 위키피디아는 검증된 출처이며, Claude가 이를 "쉬운 말"로만 변환
  - API 키 불필요, 한국어 금융 용어 커버리지가 높음
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
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DYNAMIC_TERMS_PATH = os.path.join(SCRIPT_DIR, "terms_corpus", "dynamic_terms.json")

# 위키피디아 한국어판 REST API (키 불필요)
_WIKI_SUMMARY_URL = "https://ko.wikipedia.org/api/rest_v1/page/summary/{term}"
_WIKI_SEARCH_URL  = "https://ko.wikipedia.org/w/api.php"
_REQUEST_HEADERS  = {"User-Agent": "StockAgent-RAG/1.0 (educational; contact: github.com)"}
_REQUEST_TIMEOUT  = 6   # seconds


# ─── 캐시 관리 ────────────────────────────────────────────────────────────────
def load_dynamic_terms() -> dict[str, dict]:
    """dynamic_terms.json을 {term: entry} 딕셔너리로 로드한다."""
    if not os.path.exists(DYNAMIC_TERMS_PATH):
        return {}
    try:
        with open(DYNAMIC_TERMS_PATH, encoding="utf-8") as f:
            entries = json.load(f)
        return {e["term"]: e for e in entries if "term" in e}
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("[Crawler] dynamic_terms.json 로드 실패: %s", e)
        return {}


def save_dynamic_terms(terms_dict: dict[str, dict]) -> None:
    """dynamic_terms.json에 전체 캐시를 저장한다."""
    try:
        with open(DYNAMIC_TERMS_PATH, "w", encoding="utf-8") as f:
            json.dump(list(terms_dict.values()), f, ensure_ascii=False, indent=2)
    except OSError as e:
        logger.warning("[Crawler] dynamic_terms.json 저장 실패: %s", e)


# ─── 위키피디아 조회 ──────────────────────────────────────────────────────────
def _fetch_wiki_summary(term: str) -> str | None:
    """
    위키피디아 한국어판에서 용어 요약을 가져온다.

    1차: 제목 직접 조회 (대부분의 금융 용어는 이걸로 충분)
    2차: 검색 API로 가장 관련성 높은 문서 제목 찾아 재조회
    """
    # 1차: 직접 제목 조회
    raw = _request_wiki_summary(term)
    if raw:
        return raw

    # 2차: 검색 API로 가장 유사한 문서 탐색
    title = _search_wiki_title(f"{term} 경제 금융")
    if title and title != term:
        raw = _request_wiki_summary(title)
        if raw:
            return raw

    return None


def _request_wiki_summary(title: str) -> str | None:
    """위키피디아 REST API에서 지정된 제목의 요약을 가져온다."""
    url = _WIKI_SUMMARY_URL.format(term=requests.utils.quote(title))
    try:
        resp = requests.get(url, headers=_REQUEST_HEADERS, timeout=_REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return None
        extract = resp.json().get("extract", "").strip()
        # 너무 짧은 결과(수식·연표 등)는 제외
        return extract[:600] if len(extract) > 40 else None
    except requests.RequestException as e:
        logger.debug("[Crawler] Wikipedia 직접 조회 실패 (%s): %s", title, e)
        return None


def _search_wiki_title(query: str) -> str | None:
    """위키피디아 검색 API에서 가장 관련성 높은 문서 제목을 반환한다."""
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srlimit": 1,
        "srprop": "title",
    }
    try:
        resp = requests.get(
            _WIKI_SEARCH_URL, params=params,
            headers=_REQUEST_HEADERS, timeout=_REQUEST_TIMEOUT,
        )
        results = resp.json().get("query", {}).get("search", [])
        return results[0]["title"] if results else None
    except (requests.RequestException, KeyError, IndexError) as e:
        logger.debug("[Crawler] Wikipedia 검색 실패 (%s): %s", query, e)
        return None


# ─── Claude 설명 생성 ─────────────────────────────────────────────────────────
def _generate_explanation(term: str, raw_content: str | None) -> str | None:
    """
    Claude Haiku로 초보자 친화적인 용어 설명을 생성한다.

    raw_content가 있으면: "위키 원문을 쉽게 풀어달라"
    없으면:              "Claude가 알고 있는 내용으로 직접 설명해달라"
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning("[Crawler] ANTHROPIC_API_KEY 없음 — 설명 생성 불가")
        return None

    if raw_content:
        user_msg = (
            f"다음은 '{term}'에 대한 설명입니다:\n\n{raw_content}\n\n"
            "위 내용을 주식·재테크 완전 초보자도 이해할 수 있도록 2~3문장으로 설명해주세요.\n"
            "- 전문 용어는 일상 언어로 풀어쓰세요\n"
            "- 주식 투자와의 연관성도 한 문장 포함하세요\n"
            "- 설명 본문만 출력하세요 (제목·번호·마크다운 없이)"
        )
    else:
        user_msg = (
            f"'{term}'이라는 경제·금융 용어를 주식 투자 완전 초보자가 이해할 수 있게 "
            "2~3문장으로 설명해주세요.\n"
            "- 전문 용어를 일상 언어로 쉽게 설명하세요\n"
            "- 주식 투자와의 연관성도 포함하세요\n"
            "- 설명 본문만 출력하세요 (제목·번호·마크다운 없이)"
        )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",  # 비용 절감
            max_tokens=250,
            messages=[{"role": "user", "content": user_msg}],
        )
        explanation = response.content[0].text.strip()
        return explanation if explanation else None
    except Exception as e:
        logger.error("[Crawler] Claude 설명 생성 실패 (%s): %s", term, e)
        return None


# ─── 공개 API ─────────────────────────────────────────────────────────────────
def crawl_and_explain(term: str) -> str | None:
    """
    경제 용어를 동적으로 조회해 초보자 친화적인 설명을 반환한다.

    Args:
        term: 설명이 필요한 경제·금융 용어 (예: "내재가치", "FOMC")

    Returns:
        초보자용 설명 문자열, 또는 생성 실패 시 None

    Side effect:
        성공 시 dynamic_terms.json에 캐시 저장.
        retriever.py가 이 파일을 BM25 인덱스와 ChromaDB에 반영한다.
    """
    # 1. 캐시 확인
    cached = load_dynamic_terms()
    if term in cached:
        logger.debug("[Crawler] 캐시 히트: '%s'", term)
        return cached[term]["explanation"]

    logger.info("[Crawler] 동적 조회 시작: '%s'", term)

    # 2. 위키피디아 원문 조회
    raw_content = _fetch_wiki_summary(term)
    source = "wikipedia+claude" if raw_content else "claude"
    if raw_content:
        logger.info("[Crawler] Wikipedia 원문 확보: '%s'", term)
    else:
        logger.info("[Crawler] Wikipedia 없음 — Claude 단독 생성: '%s'", term)

    # 3. Claude로 쉬운 설명 생성
    explanation = _generate_explanation(term, raw_content)
    if not explanation:
        logger.warning("[Crawler] 설명 생성 실패: '%s'", term)
        return None

    # 4. 캐시 저장
    cached[term] = {
        "term": term,
        "explanation": explanation,
        "category": "동적생성",
        "related_terms": [],
        "source": source,
    }
    save_dynamic_terms(cached)
    logger.info("[Crawler] 캐시 저장 완료: '%s' (출처: %s)", term, source)

    return explanation

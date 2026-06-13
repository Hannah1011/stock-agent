"""
rag/retriever.py

경제 용어 RAG 검색 인터페이스.

[검색 전략: Hybrid Search]
단순 Dense(임베딩) 검색만 쓰면 정확한 키워드를 놓치고,
단순 BM25(키워드) 검색만 쓰면 표현이 다른 동의어를 놓친다.
두 방식을 RRF(Reciprocal Rank Fusion)로 병합해 두 장점을 취한다.

  Dense  (ChromaDB): "금리를 올린다" → 기준금리, 양적완화 (의미 유사)
  BM25   (rank_bm25): "RSI"          → RSI 용어 정확 매칭
  RRF    병합        : 두 결과를 순위 기반으로 합산해 최종 순위 결정

[3단계 용어 추출 흐름 (extract_and_explain_terms)]
  1단계 정확 매칭: 텍스트에 용어명이 그대로 포함되어 있으면 즉시 추가
  2단계 BM25: 용어집 키워드와 유사한 단어가 텍스트에 있으면 추가
  3단계 Dense: 위 두 방법으로 찾지 못한 의미적 연관 용어 보충

[임베딩 제공자]
  .env의 EMBEDDING_PROVIDER로 "local"(기본) 또는 "openai" 선택 가능.
  build_vectordb.py와 동일한 제공자를 사용해야 한다.
"""

import json
import logging
import os
import re
import sys
from typing import Optional

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()
logger = logging.getLogger(__name__)

CHROMA_DIR      = os.path.join(SCRIPT_DIR, "chroma_db")
CORPUS_PATH     = os.path.join(SCRIPT_DIR, "terms_corpus", "economy_terms.json")
COLLECTION_NAME = "economy_terms"

# Hybrid Search 파라미터
_RRF_K          = 60    # RRF 상수 (클수록 상위 랭크 과점 완화)
_SIM_THRESHOLD  = 0.25  # Dense 유사도 하한 (코사인 유사도 기준)
_MAX_RESULTS    = 5     # extract_and_explain_terms 최대 반환 수


# ─── 모듈 레벨 lazy 싱글턴 ─────────────────────────────────────────────────
_collection    = None
_chroma_client = None
_bm25_index:  Optional[BM25Okapi]  = None
_bm25_corpus: Optional[list[dict]] = None   # BM25 인덱스와 연결된 원본 용어 목록


# ─── 초기화 ──────────────────────────────────────────────────────────────────
def _get_embedding_function():
    """build_vectordb.py와 동일한 임베딩 함수를 반환한다."""
    provider = os.getenv("EMBEDDING_PROVIDER", "local").lower()
    if provider == "openai":
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model_name="text-embedding-3-small",
        )
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS"
    )


def _ensure_db_built() -> None:
    """ChromaDB가 없거나 비어 있으면 자동 빌드한다."""
    marker = os.path.join(CHROMA_DIR, "chroma.sqlite3")
    needs_build = not os.path.exists(marker)

    if not needs_build:
        # sqlite3는 있지만 컬렉션이 없는 경우 (부분 실패 복구)
        try:
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            existing = [c.name for c in client.list_collections()]
            needs_build = COLLECTION_NAME not in existing
        except Exception:
            needs_build = True

    if needs_build:
        logger.info("[RAG] chroma_db 없음 — 자동 빌드 시작")
        from rag.build_vectordb import build
        build()
        logger.info("[RAG] 벡터 DB 구축 완료")


def _get_collection():
    """ChromaDB 컬렉션 싱글턴을 반환한다."""
    global _collection, _chroma_client
    if _collection is None:
        _ensure_db_built()
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = _chroma_client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=_get_embedding_function(),
        )
    return _collection


def _get_bm25() -> tuple[BM25Okapi, list[dict]]:
    """
    BM25 인덱스 싱글턴을 반환한다.

    ChromaDB와 달리 BM25는 파일로 저장하지 않고 매번 corpus에서 즉시 빌드한다.
    용어 수(80개)가 적으므로 빌드 시간이 무시할 수준이다.

    토큰화: 공백 분리 + 용어명 단독 추가
      한국어는 조사가 붙어있어 "기준금리를"이 "기준금리"와 다르게 인식될 수 있다.
      용어명을 별도 토큰으로 추가해 정확 매칭 확률을 높인다.
    """
    global _bm25_index, _bm25_corpus
    if _bm25_index is None:
        with open(CORPUS_PATH, encoding="utf-8") as f:
            terms = json.load(f)

        tokenized_corpus: list[list[str]] = []
        for term in terms:
            text = f"{term['term']} {term['explanation']}"
            tokens = text.split()
            # 용어명 자체를 첫 번째 토큰으로 한 번 더 추가 → 정확 매칭 가중치 부여
            tokens = [term["term"]] + tokens
            tokenized_corpus.append(tokens)

        _bm25_corpus = terms
        _bm25_index  = BM25Okapi(tokenized_corpus)

    return _bm25_index, _bm25_corpus


# ─── Dense 검색 (ChromaDB) ───────────────────────────────────────────────────
def _dense_search(query: str, n: int) -> list[dict]:
    """
    ChromaDB 임베딩 유사도 검색.

    반환값의 'score'는 코사인 유사도 (0~1, 높을수록 유사).
    ChromaDB가 반환하는 distance는 1 - cosine_similarity 이므로
    similarity = 1 - distance 로 변환한다.
    """
    collection = _get_collection()
    n = min(n, collection.count())
    if n == 0:
        return []

    try:
        results = collection.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.error("[RAG] Dense 검색 실패: %s", e)
        return []

    output: list[dict] = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = 1.0 - dist   # cosine distance → cosine similarity
        if similarity < _SIM_THRESHOLD:
            continue
        explanation = doc.split(": ", 1)[-1] if ": " in doc else doc
        output.append({
            "term": meta["term"],
            "explanation": explanation,
            "category": meta.get("category", ""),
            "related_terms": meta.get("related_terms", ""),
            "score": round(similarity, 4),
        })

    return output


# ─── Sparse 검색 (BM25) ──────────────────────────────────────────────────────
def _bm25_search(query: str, n: int) -> list[dict]:
    """
    BM25 키워드 기반 검색.

    정확한 용어명 매칭에 강하다. 예) "RSI" → RSI 용어 상위 순위
    반환값의 'score'는 BM25 원시 점수 (양수, 클수록 관련성 높음).
    """
    bm25, corpus = _get_bm25()
    query_tokens = query.split()
    scores = bm25.get_scores(query_tokens)

    # 점수와 인덱스를 함께 정렬해 상위 N개 추출
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    output: list[dict] = []
    for idx, score in ranked[:n]:
        if score <= 0:
            break
        term = corpus[idx]
        output.append({
            "term": term["term"],
            "explanation": term["explanation"],
            "category": term.get("category", ""),
            "related_terms": ", ".join(term.get("related_terms", [])),
            "score": round(float(score), 4),
        })

    return output


# ─── RRF 병합 ────────────────────────────────────────────────────────────────
def _rrf_merge(
    dense_results: list[dict],
    bm25_results: list[dict],
    n: int,
) -> list[dict]:
    """
    Reciprocal Rank Fusion으로 Dense와 BM25 결과를 병합한다.

    RRF 공식: score(d) = Σ 1 / (k + rank(d))
    같은 용어가 두 리스트 모두에서 높은 순위이면 최종 점수가 높아진다.
    """
    rrf_scores: dict[str, float] = {}

    for rank, item in enumerate(dense_results):
        key = item["term"]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)

    for rank, item in enumerate(bm25_results):
        key = item["term"]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)

    # 점수 높은 순으로 정렬, 원본 상세 정보는 dense 결과 우선으로 가져옴
    dense_map = {item["term"]: item for item in dense_results}
    bm25_map  = {item["term"]: item for item in bm25_results}

    sorted_terms = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    merged: list[dict] = []
    for term_name, rrf_score in sorted_terms[:n]:
        detail = dense_map.get(term_name) or bm25_map.get(term_name, {})
        merged.append({
            "term": term_name,
            "explanation": detail.get("explanation", ""),
            "category": detail.get("category", ""),
            "related_terms": detail.get("related_terms", ""),
            "rrf_score": round(rrf_score, 6),
        })

    return merged


# ─── 공개 검색 API ───────────────────────────────────────────────────────────
def search_term(query: str, n_results: int = 3) -> list[dict]:
    """
    Hybrid Search (Dense + BM25 + RRF)로 관련 용어를 반환한다.

    Args:
        query: 검색 쿼리 (자연어 문장 또는 용어명)
        n_results: 반환할 최대 결과 수

    Returns:
        [{"term": str, "explanation": str, "category": str,
          "related_terms": str, "rrf_score": float}]
    """
    candidate_n = n_results * 3   # 병합 전 여유 있게 가져옴
    dense  = _dense_search(query, candidate_n)
    sparse = _bm25_search(query, candidate_n)
    return _rrf_merge(dense, sparse, n_results)


def get_term_explanation(term: str) -> Optional[str]:
    """
    단일 용어명의 설명을 반환한다.

    정확 매칭 우선: 용어명이 결과의 'term'과 일치하거나 포함 관계이면 반환.
    없으면 Hybrid Search 결과 중 가장 유사한 것을 반환.
    """
    results = search_term(term, n_results=3)
    if not results:
        return None

    # 정확 매칭 또는 포함 관계 우선
    for r in results:
        if r["term"] == term or term in r["term"] or r["term"] in term:
            return r["explanation"]

    # 없으면 최상위 결과 반환
    return results[0]["explanation"]


def extract_and_explain_terms(
    text: str,
    max_terms: int = _MAX_RESULTS,
) -> list[dict]:
    """
    텍스트에서 경제 용어를 추출하고 각 용어의 설명을 반환한다.

    3단계 순서로 진행하며 앞 단계에서 찾은 용어는 이후 단계에서 중복 추가하지 않는다.

    1단계 — 정확 매칭:
      BM25 corpus의 용어명이 텍스트에 그대로 포함되어 있으면 즉시 수집.
      예) "RSI가 42로 하락" → "RSI" 용어 직접 검출

    2단계 — BM25:
      텍스트 전체를 BM25 쿼리로 사용해 키워드 유사 용어를 추가.
      예) "과매수 구간 진입" → "RSI" (과매수 단어 포함)

    3단계 — Dense:
      위 두 단계로 찾지 못한 의미적 연관 용어를 임베딩 검색으로 보충.
      예) "연준이 금리 인상" → "기준금리", "양적완화", "테이퍼링"

    Args:
        text: 뉴스 기사 등 분석할 텍스트
        max_terms: 최대 반환 용어 수

    Returns:
        [{"term": str, "explanation": str}]
    """
    found: dict[str, str] = {}   # term → explanation (순서 보존을 위해 dict 사용)
    _, corpus = _get_bm25()

    # ── 1단계: 정확 매칭 ──────────────────────────────────────────────────────
    for item in corpus:
        term_name = item["term"]
        if term_name in text and term_name not in found:
            found[term_name] = item["explanation"]
            if len(found) >= max_terms:
                break

    if len(found) >= max_terms:
        return [{"term": t, "explanation": e} for t, e in found.items()]

    # ── 2단계: BM25 키워드 검색 ──────────────────────────────────────────────
    remaining = max_terms - len(found)
    bm25_hits = _bm25_search(text, n=remaining * 2)
    for hit in bm25_hits:
        if hit["term"] not in found:
            found[hit["term"]] = hit["explanation"]
        if len(found) >= max_terms:
            break

    if len(found) >= max_terms:
        return [{"term": t, "explanation": e} for t, e in found.items()]

    # ── 3단계: Dense 의미 검색 ───────────────────────────────────────────────
    remaining = max_terms - len(found)
    dense_hits = _dense_search(text, n=remaining * 2)
    for hit in dense_hits:
        if hit["term"] not in found:
            found[hit["term"]] = hit["explanation"]
        if len(found) >= max_terms:
            break

    return [{"term": t, "explanation": e} for t, e in found.items()]


def is_db_ready() -> bool:
    """ChromaDB가 구축되어 있고 데이터가 있으면 True를 반환한다."""
    try:
        return _get_collection().count() > 0
    except Exception:
        return False

import os
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(SCRIPT_DIR, "chroma_db")
COLLECTION_NAME = "economy_terms"

_client: Optional[chromadb.PersistentClient] = None
_collection = None


def _ensure_db_exists() -> None:
    """chroma_db가 없거나 비어 있으면 자동으로 구축한다."""
    marker = os.path.join(CHROMA_DIR, "chroma.sqlite3")
    if not os.path.exists(marker):
        print("[RAG] chroma_db가 없습니다. 자동으로 벡터 DB를 구축합니다...")
        from rag.build_vectordb import build
        build()
        print("[RAG] 벡터 DB 구축 완료")


def _get_collection():
    global _client, _collection
    if _collection is None:
        _ensure_db_exists()
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS"
        )
        existing = [c.name for c in _client.list_collections()]
        if COLLECTION_NAME not in existing:
            # sqlite3는 있지만 컬렉션이 없는 경우 (부분 실패 복구)
            from rag.build_vectordb import build
            build()
        _collection = _client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
        )
    return _collection


def search_term(query: str, n_results: int = 3) -> list[dict]:
    """
    쿼리 텍스트와 가장 유사한 용어를 ChromaDB에서 검색한다.
    반환값: [{"term": str, "explanation": str, "category": str, "related_terms": str}]
    """
    collection = _get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # cosine distance → similarity (낮을수록 유사)
        similarity = 1 - dist
        if similarity < 0.3:
            continue
        explanation = doc.split(": ", 1)[-1] if ": " in doc else doc
        output.append(
            {
                "term": meta["term"],
                "explanation": explanation,
                "category": meta.get("category", ""),
                "related_terms": meta.get("related_terms", ""),
                "similarity": round(similarity, 3),
            }
        )
    return output


def get_term_explanation(term: str) -> Optional[str]:
    """
    정확한 용어명으로 설명을 가져온다.
    먼저 exact match를 시도하고, 없으면 유사 검색으로 대체.
    """
    results = search_term(term, n_results=1)
    if not results:
        return None
    best = results[0]
    # 용어명이 쿼리와 충분히 일치하면 반환
    if best["similarity"] >= 0.5 or term in best["term"] or best["term"] in term:
        return best["explanation"]
    return None


def extract_and_explain_terms(text: str, known_terms: Optional[list[str]] = None) -> list[dict]:
    """
    텍스트에서 경제 용어를 찾아 설명 목록을 반환한다.
    known_terms 목록이 있으면 해당 용어만 검색하고,
    없으면 텍스트 전체를 쿼리로 사용해 관련 용어를 찾는다.
    """
    if known_terms:
        results = []
        for term in known_terms:
            explanation = get_term_explanation(term)
            if explanation:
                results.append({"term": term, "explanation": explanation})
        return results

    # 텍스트 전체 기반 유사 검색
    raw = search_term(text, n_results=5)
    return [{"term": r["term"], "explanation": r["explanation"]} for r in raw]


def is_db_ready() -> bool:
    """ChromaDB가 구축되어 있는지 확인한다."""
    try:
        collection = _get_collection()
        return collection.count() > 0
    except Exception:
        return False

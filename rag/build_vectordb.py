"""
rag/build_vectordb.py

경제 용어 사전을 ChromaDB 벡터 DB에 적재한다.
retriever.py가 앱 최초 실행 시 자동으로 호출하므로 수동 실행은 필수가 아니다.

수동 실행 (캐시 초기화 등):
    python rag/build_vectordb.py

임베딩 제공자 선택 (.env):
    EMBEDDING_PROVIDER=local   # snunlp/KR-SBERT (기본, 무료, 400MB 다운로드)
    EMBEDDING_PROVIDER=openai  # text-embedding-3-small (유료, 즉시 사용)
"""

import json
import logging
import os
import sys

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

CORPUS_PATH      = os.path.join(SCRIPT_DIR, "terms_corpus", "economy_terms.json")
CHROMA_DIR       = os.path.join(SCRIPT_DIR, "chroma_db")
COLLECTION_NAME  = "economy_terms"

# 로컬 한국어 SBERT 모델
_LOCAL_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"

# OpenAI 임베딩 모델 (text-embedding-3-small: 저비용, 고품질)
_OPENAI_MODEL = "text-embedding-3-small"


# ─── 임베딩 함수 선택 ─────────────────────────────────────────────────────────
def _get_embedding_function():
    """
    EMBEDDING_PROVIDER 환경변수에 따라 임베딩 함수를 반환한다.

    local  → snunlp/KR-SBERT (한국어 특화, 로컬 실행, 무료)
    openai → text-embedding-3-small (다국어 고품질, API 호출)
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "local").lower()

    if provider == "openai":
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_key:
            raise ValueError(
                "EMBEDDING_PROVIDER=openai로 설정했지만 OPENAI_API_KEY가 없습니다. "
                ".env에 OPENAI_API_KEY를 추가하거나 EMBEDDING_PROVIDER=local로 변경하세요."
            )
        logger.info("[RAG] 임베딩: OpenAI %s", _OPENAI_MODEL)
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_key,
            model_name=_OPENAI_MODEL,
        )

    # 기본값: 로컬 한국어 SBERT
    logger.info("[RAG] 임베딩: 로컬 SBERT (%s)", _LOCAL_MODEL)
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=_LOCAL_MODEL
    )


# ─── 코어 빌드 함수 ──────────────────────────────────────────────────────────
def load_terms(path: str = CORPUS_PATH) -> list[dict]:
    """경제 용어 JSON 파일을 로드한다."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build(corpus_path: str = CORPUS_PATH, chroma_dir: str = CHROMA_DIR) -> None:
    """
    용어 사전을 ChromaDB에 적재한다.

    문서 텍스트 구성 전략:
      "{term}: {explanation}" 형태로 합쳐 저장한다.
      이유: 용어명 단독 임베딩보다 설명 문장이 포함되면
           "RSI가 뭐야?" 같은 자연어 질문에서 더 정확히 매칭된다.
    """
    terms = load_terms(corpus_path)
    print(f"[RAG] 용어 {len(terms)}개 로드 완료")

    embed_fn = _get_embedding_function()

    client = chromadb.PersistentClient(path=chroma_dir)

    # 이미 존재하면 삭제 후 재생성 (재실행 안전)
    existing_names = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_names:
        client.delete_collection(COLLECTION_NAME)
        print("[RAG] 기존 컬렉션 초기화")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for i, term in enumerate(terms):
        # 검색 텍스트: 용어명 + 설명을 합쳐 의미적 검색 품질을 높인다
        doc_text = f"{term['term']}: {term['explanation']}"
        documents.append(doc_text)
        metadatas.append({
            "term": term["term"],
            "category": term.get("category", ""),
            "related_terms": ", ".join(term.get("related_terms", [])),
        })
        ids.append(f"term_{i:04d}")

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"[RAG] ChromaDB 적재 완료: {len(documents)}개 → {chroma_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build()

"""
최초 1회 실행: 경제 용어 사전을 ChromaDB에 적재합니다.
  python rag/build_vectordb.py
"""

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import chromadb
from chromadb.utils import embedding_functions

CORPUS_PATH = os.path.join(SCRIPT_DIR, "terms_corpus", "economy_terms.json")
CHROMA_DIR = os.path.join(SCRIPT_DIR, "chroma_db")
COLLECTION_NAME = "economy_terms"


def load_terms(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build(corpus_path: str = CORPUS_PATH, chroma_dir: str = CHROMA_DIR) -> None:
    terms = load_terms(corpus_path)
    print(f"용어 수: {len(terms)}개 로드 완료")

    client = chromadb.PersistentClient(path=chroma_dir)

    # 이미 존재하면 삭제 후 재생성 (재실행 안전)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print("기존 컬렉션 삭제 완료")

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS"
    )

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    documents, metadatas, ids = [], [], []
    for i, term in enumerate(terms):
        # 검색 텍스트: 용어명 + 설명 조합 (한국어 임베딩 품질 향상)
        doc_text = f"{term['term']}: {term['explanation']}"
        documents.append(doc_text)
        metadatas.append(
            {
                "term": term["term"],
                "category": term.get("category", ""),
                "related_terms": ", ".join(term.get("related_terms", [])),
            }
        )
        ids.append(f"term_{i:04d}")

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"ChromaDB 적재 완료: {len(documents)}개 → {chroma_dir}")


if __name__ == "__main__":
    build()

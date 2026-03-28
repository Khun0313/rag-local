"""ChromaDB 벡터 스토어 관리."""

from __future__ import annotations

import logging

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

from rag_local.config import settings

logger = logging.getLogger(__name__)

_client: chromadb.ClientAPI | None = None


def get_chroma_client() -> chromadb.ClientAPI:
    """ChromaDB PersistentClient 싱글톤을 반환한다."""
    global _client
    if _client is None:
        settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
        logger.info("ChromaDB 클라이언트 초기화: %s", settings.chroma_persist_dir)
    return _client


def get_vector_store() -> ChromaVectorStore:
    """ChromaDB 벡터 스토어를 반환한다."""
    client = get_chroma_client()
    collection = client.get_or_create_collection("rag_local")
    return ChromaVectorStore(chroma_collection=collection)

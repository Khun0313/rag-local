"""하이브리드 검색: Vector + BM25 (Kiwi 한국어 형태소 분석) + RRF."""

from __future__ import annotations

import logging

from kiwipiepy import Kiwi
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.storage import StorageContext
from llama_index.retrievers.bm25 import BM25Retriever

from rag_local.config import settings
from rag_local.store.chroma import get_vector_store
from rag_local.store.docstore import get_docstore

logger = logging.getLogger(__name__)

# Kiwi 형태소 분석기 (싱글톤)
_kiwi: Kiwi | None = None


def _get_kiwi() -> Kiwi:
    global _kiwi
    if _kiwi is None:
        _kiwi = Kiwi()
        logger.info("Kiwi 형태소 분석기 초기화")
    return _kiwi


def korean_tokenizer(text: str) -> list[str]:
    """한국어 텍스트를 형태소로 분리한다.

    명사(N*), 동사(V*), 부사(MA*) 품사만 추출하여 BM25 검색 품질을 향상.
    """
    kiwi = _get_kiwi()
    tokens = kiwi.tokenize(text)
    return [t.form for t in tokens if t.tag.startswith(("N", "V", "MA"))]


def get_hybrid_retriever() -> QueryFusionRetriever:
    """하이브리드 검색기 (Vector + BM25 + RRF)를 반환한다."""
    vector_store = get_vector_store()
    docstore = get_docstore()

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=docstore,
    )

    # VectorStoreIndex 로드
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )

    # Vector retriever
    vector_retriever = index.as_retriever(similarity_top_k=settings.top_k)

    # BM25 retriever (Kiwi 한국어 토크나이저)
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=docstore,
        similarity_top_k=settings.top_k,
        tokenizer=korean_tokenizer,
    )

    # Hybrid: Reciprocal Rank Fusion
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        mode="reciprocal_rerank",
        num_queries=1,  # 쿼리 확장 비활성화
        similarity_top_k=settings.top_k,
    )

    logger.info("하이브리드 검색기 초기화 (Vector + BM25, top_k=%d)", settings.top_k)
    return hybrid_retriever

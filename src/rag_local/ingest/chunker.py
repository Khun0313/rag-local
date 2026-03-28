"""문서 청킹: SentenceSplitter로 문서를 노드로 분할."""

from __future__ import annotations

import logging

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, Document

from rag_local.config import settings

logger = logging.getLogger(__name__)


def get_splitter() -> SentenceSplitter:
    """설정에 따른 SentenceSplitter를 반환한다."""
    return SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def chunk_documents(documents: list[Document]) -> list[BaseNode]:
    """문서 리스트를 청킹하여 노드 리스트로 반환한다."""
    if not documents:
        return []
    splitter = get_splitter()
    nodes = splitter.get_nodes_from_documents(documents)
    logger.info(
        "청킹 완료: %d 문서 → %d 노드 (chunk_size=%d, overlap=%d)",
        len(documents), len(nodes), settings.chunk_size, settings.chunk_overlap,
    )
    return nodes

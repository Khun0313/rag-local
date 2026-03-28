"""SimpleDocumentStore 관리 (BM25 검색용 원문 텍스트 보관)."""

from __future__ import annotations

import logging

from llama_index.core.storage.docstore import SimpleDocumentStore

from rag_local.config import settings

logger = logging.getLogger(__name__)

_docstore: SimpleDocumentStore | None = None


def get_docstore() -> SimpleDocumentStore:
    """SimpleDocumentStore 싱글톤을 반환한다.

    기존 파일이 있으면 로드하고, 없으면 새로 생성한다.
    """
    global _docstore
    if _docstore is None:
        if settings.docstore_path.exists():
            _docstore = SimpleDocumentStore.from_persist_path(
                str(settings.docstore_path)
            )
            logger.info("DocStore 로드: %s", settings.docstore_path)
        else:
            _docstore = SimpleDocumentStore()
            logger.info("새 DocStore 생성")
    return _docstore


def persist_docstore() -> None:
    """DocStore를 디스크에 저장한다."""
    if _docstore is not None:
        settings.docstore_path.parent.mkdir(parents=True, exist_ok=True)
        _docstore.persist(str(settings.docstore_path))
        logger.info("DocStore 저장: %s", settings.docstore_path)

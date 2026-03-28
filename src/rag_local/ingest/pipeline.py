"""문서 처리 파이프라인: 스캔 → 파싱 → 청킹 → 임베딩 → 저장."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex

from rag_local.config import settings
from rag_local.ingest.chunker import chunk_documents
from rag_local.ingest.parsers import SUPPORTED_EXTENSIONS, parse_file
from rag_local.store.chroma import get_vector_store
from rag_local.store.docstore import get_docstore, persist_docstore

logger = logging.getLogger(__name__)


def _file_hash(file_path: Path) -> str:
    """파일의 SHA-256 해시를 반환한다."""
    h = hashlib.sha256()
    h.update(file_path.read_bytes())
    return h.hexdigest()


def _load_ingestion_state() -> dict:
    """이전 처리 상태를 로드한다."""
    if settings.ingestion_state_path.exists():
        return json.loads(settings.ingestion_state_path.read_text(encoding="utf-8"))
    return {}


def _save_ingestion_state(state: dict) -> None:
    """처리 상태를 저장한다."""
    settings.ingestion_state_path.parent.mkdir(parents=True, exist_ok=True)
    settings.ingestion_state_path.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def scan_documents(directory: Path | None = None) -> list[Path]:
    """디렉토리에서 지원하는 문서 파일을 검색한다."""
    doc_dir = directory or settings.documents_dir
    if not doc_dir.exists():
        logger.warning("문서 디렉토리가 없습니다: %s", doc_dir)
        return []
    files = [
        f for f in sorted(doc_dir.rglob("*"))
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    logger.info("스캔 완료: %d개 문서 발견 (%s)", len(files), doc_dir)
    return files


def ingest(directory: Path | None = None) -> dict:
    """문서를 처리하여 벡터 스토어와 DocStore에 저장한다.

    증분 처리: SHA-256 해시로 변경된 파일만 처리한다.

    Returns:
        {"ingested": int, "skipped": int, "errors": list[str]}
    """
    files = scan_documents(directory)
    state = _load_ingestion_state()

    new_files = []
    skipped = 0
    for f in files:
        file_hash = _file_hash(f)
        if state.get(str(f)) == file_hash:
            skipped += 1
            continue
        new_files.append((f, file_hash))

    if not new_files:
        logger.info("새로운/변경된 문서 없음 (스킵: %d)", skipped)
        return {"ingested": 0, "skipped": skipped, "errors": []}

    logger.info("처리 대상: %d개 (스킵: %d)", len(new_files), skipped)

    # 파싱
    all_documents = []
    errors = []
    for file_path, file_hash in new_files:
        try:
            docs = parse_file(file_path)
            all_documents.extend(docs)
            state[str(file_path)] = file_hash
        except Exception as e:
            logger.error("파싱 실패 (%s): %s", file_path.name, e)
            errors.append(f"{file_path.name}: {e}")

    if not all_documents:
        logger.warning("파싱된 문서가 없습니다.")
        return {"ingested": 0, "skipped": skipped, "errors": errors}

    # 청킹
    nodes = chunk_documents(all_documents)

    # 스토어에 저장
    vector_store = get_vector_store()
    docstore = get_docstore()

    # DocStore에 명시적으로 노드 추가 (BM25 검색용)
    docstore.add_documents(nodes)

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=docstore,
    )

    VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        show_progress=True,
    )

    # 영구 저장
    persist_docstore()
    _save_ingestion_state(state)

    ingested_count = len(new_files) - len(errors)
    logger.info("처리 완료: %d개 문서, %d개 노드", ingested_count, len(nodes))
    return {"ingested": ingested_count, "skipped": skipped, "errors": errors}

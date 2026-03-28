"""FastAPI REST API 서버."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from rag_local.config import settings

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Local API",
    description="로컬 RAG 시스템 REST API",
    version="0.1.0",
)

# ── 요청/응답 모델 ──


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class SourceInfo(BaseModel):
    file_name: str
    page_number: Optional[int] = None
    score: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]


class IngestRequest(BaseModel):
    directory: Optional[str] = None


class IngestResponse(BaseModel):
    ingested: int
    skipped: int
    errors: list[str]


class DocumentInfo(BaseModel):
    file_name: str
    file_path: str
    file_type: str


class HealthResponse(BaseModel):
    status: str
    logged_in: bool
    model: str
    documents_count: int


# ── 엔드포인트 ──


@app.get("/health", response_model=HealthResponse)
def health():
    """상태 확인."""
    from rag_local.auth.token import is_logged_in

    doc_count = 0
    if settings.documents_dir.exists():
        doc_count = sum(1 for f in settings.documents_dir.rglob("*") if f.is_file())

    return HealthResponse(
        status="ok",
        logged_in=is_logged_in(),
        model=settings.llm_model,
        documents_count=doc_count,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest = IngestRequest()):
    """문서 처리."""
    from rag_local.llm.provider import init_llama_settings
    from rag_local.ingest.pipeline import ingest as run_ingest

    init_llama_settings()
    directory = Path(req.directory) if req.directory else None
    result = run_ingest(directory)
    return IngestResponse(**result)


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    """질문에 답변."""
    from rag_local.rag.engine import query as run_query

    try:
        result = run_query(req.question)
        return QueryResponse(
            answer=result.answer,
            sources=[SourceInfo(**s) for s in result.sources],
        )
    except Exception as e:
        logger.error("쿼리 실패: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", response_model=list[DocumentInfo])
def list_documents():
    """처리된 문서 목록."""
    from rag_local.ingest.parsers import SUPPORTED_EXTENSIONS

    if not settings.documents_dir.exists():
        return []

    docs = []
    for f in sorted(settings.documents_dir.rglob("*")):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.append(DocumentInfo(
                file_name=f.name,
                file_path=str(f),
                file_type=f.suffix.lower().lstrip("."),
            ))
    return docs

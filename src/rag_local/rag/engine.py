"""RAG 쿼리 엔진: 검색 → 프롬프트 구성 → 생성."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from llama_index.core import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer

from rag_local.llm.provider import get_llm, init_llama_settings
from rag_local.retrieval.hybrid import get_hybrid_retriever

logger = logging.getLogger(__name__)

RAG_PROMPT_TEMPLATE = PromptTemplate(
    "다음 컨텍스트를 참고하여 질문에 답해주세요.\n"
    "답변은 컨텍스트에 있는 정보만 사용하세요.\n"
    "컨텍스트에 답이 없으면 '제공된 문서에서 관련 정보를 찾을 수 없습니다'라고 답하세요.\n"
    "답변 마지막에 출처(파일명, 페이지)를 표시해주세요.\n\n"
    "컨텍스트:\n"
    "-----\n"
    "{context_str}\n"
    "-----\n\n"
    "질문: {query_str}\n\n"
    "답변:"
)


@dataclass
class QueryResult:
    """RAG 쿼리 결과."""
    answer: str
    sources: list[dict] = field(default_factory=list)


def get_query_engine() -> RetrieverQueryEngine:
    """RAG 쿼리 엔진을 생성한다."""
    init_llama_settings()
    retriever = get_hybrid_retriever()
    llm = get_llm()

    response_synthesizer = get_response_synthesizer(
        llm=llm,
        text_qa_template=RAG_PROMPT_TEMPLATE,
        response_mode="compact",
    )

    engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )

    logger.info("RAG 쿼리 엔진 초기화 완료")
    return engine


def query(question: str) -> QueryResult:
    """질문에 대한 RAG 답변을 반환한다."""
    engine = get_query_engine()
    response = engine.query(question)

    # 출처 추출
    sources = []
    for node in response.source_nodes:
        metadata = node.node.metadata
        sources.append({
            "file_name": metadata.get("file_name", "unknown"),
            "page_number": metadata.get("page_number"),
            "score": round(node.score, 4) if node.score else None,
        })

    return QueryResult(
        answer=str(response),
        sources=sources,
    )

"""LLM 프로바이더: ChatGPT Codex Responses API + LlamaIndex 연동.

ChatGPT 구독 OAuth 토큰은 표준 OpenAI API(api.openai.com)가 아닌
Codex 백엔드(chatgpt.com/backend-api/codex/responses)를 사용한다.
LlamaIndex의 CustomLLM으로 래핑하여 RAG 파이프라인에서 사용한다.

참고: auto-trade/auto_trader/llm/codex_client.py
"""

from __future__ import annotations

import json
import logging
from typing import Any, Generator

import requests
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms import CustomLLM
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.settings import Settings as LlamaSettings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from rag_local.auth.token import (
    CODEX_RESPONSES_URL,
    get_auth_headers,
    refresh_access_token,
)
from rag_local.config import settings

logger = logging.getLogger(__name__)


class CodexLLM(CustomLLM):
    """ChatGPT Codex Responses API를 사용하는 LlamaIndex LLM.

    Codex 백엔드는 OpenAI Responses API 형식을 사용하며,
    SSE 스트리밍(stream=True)이 필수이다.
    """

    model: str = settings.llm_model
    max_tokens: int = 4096
    temperature: float = 0.1

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model,
            context_window=128000,
            num_output=self.max_tokens,
            is_chat_model=True,
        )

    def _build_payload(self, prompt: str) -> dict:
        """Responses API 형식의 페이로드를 구성한다."""
        return {
            "model": self.model,
            "instructions": "You are a helpful AI assistant. Answer based on the provided context. Respond in Korean.",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
            "store": False,
            "stream": True,
        }

    def _call_codex(self, prompt: str) -> str:
        """Codex 백엔드 API를 호출하고 SSE 응답을 파싱한다."""
        headers = get_auth_headers()
        payload = self._build_payload(prompt)

        try:
            resp = requests.post(
                CODEX_RESPONSES_URL,
                headers=headers,
                json=payload,
                timeout=120,
                stream=True,
            )
            resp.raise_for_status()
            return self._parse_sse_response(resp)

        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0

            if status == 401:
                logger.warning("401 Unauthorized — 토큰 갱신 시도...")
                new_token = refresh_access_token()
                if new_token:
                    headers = get_auth_headers()
                    resp2 = requests.post(
                        CODEX_RESPONSES_URL,
                        headers=headers,
                        json=payload,
                        timeout=120,
                        stream=True,
                    )
                    resp2.raise_for_status()
                    return self._parse_sse_response(resp2)
                else:
                    raise RuntimeError(
                        "토큰 갱신 실패. 외부 프로그램으로 토큰을 갱신해주세요."
                    ) from e
            raise

    def _parse_sse_response(self, resp: requests.Response) -> str:
        """SSE 스트리밍 응답에서 텍스트를 조립한다."""
        text_parts = []
        for line in resp.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8") if isinstance(line, bytes) else line
            if not line_str.startswith("data: "):
                continue
            data_str = line_str[6:]
            if data_str == "[DONE]":
                break
            try:
                event = json.loads(data_str)
                event_type = event.get("type", "")
                if event_type == "response.output_text.delta":
                    text_parts.append(event.get("delta", ""))
            except json.JSONDecodeError:
                continue
        result = "".join(text_parts)
        if not result:
            logger.warning("SSE 응답에서 텍스트를 추출하지 못했습니다.")
        return result

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        text = self._call_codex(prompt)
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> Generator:
        """스트리밍 완료 (현재는 non-streaming으로 구현)."""
        text = self._call_codex(prompt)
        yield CompletionResponse(text=text, delta=text)

    def chat(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
        # 시스템 메시지를 instructions로, 나머지를 user prompt로 분리
        system_parts = []
        user_parts = []
        for m in messages:
            content = m.content or ""
            if not content.strip():
                continue
            if m.role == MessageRole.SYSTEM:
                system_parts.append(content)
            else:
                user_parts.append(content)

        instructions = "\n".join(system_parts) if system_parts else self._build_payload("")["instructions"]
        prompt = "\n".join(user_parts)

        headers = get_auth_headers()
        payload = {
            "model": self.model,
            "instructions": instructions,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
            "store": False,
            "stream": True,
        }

        try:
            resp = requests.post(
                CODEX_RESPONSES_URL, headers=headers, json=payload,
                timeout=120, stream=True,
            )
            resp.raise_for_status()
            text = self._parse_sse_response(resp)
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            body = e.response.text[:300] if e.response is not None else ""
            logger.error("Chat API 에러 HTTP %s: %s", status, body)
            raise

        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=text),
        )

    def stream_chat(self, messages: list[ChatMessage], **kwargs: Any) -> Generator:
        response = self.chat(messages, **kwargs)
        yield response


def get_llm() -> CodexLLM:
    """Codex LLM 인스턴스를 반환한다."""
    llm = CodexLLM()
    logger.info(
        "CodexLLM 초기화: model=%s, endpoint=%s",
        settings.llm_model, CODEX_RESPONSES_URL
    )
    return llm


def get_embed_model() -> HuggingFaceEmbedding:
    """한국어 임베딩 모델을 반환한다."""
    embed = HuggingFaceEmbedding(model_name=settings.embed_model)
    logger.info("임베딩 모델 초기화: %s", settings.embed_model)
    return embed


def init_llama_settings() -> None:
    """LlamaIndex 전역 설정을 초기화한다."""
    LlamaSettings.llm = get_llm()
    LlamaSettings.embed_model = get_embed_model()
    LlamaSettings.chunk_size = settings.chunk_size
    LlamaSettings.chunk_overlap = settings.chunk_overlap
    logger.info("LlamaIndex 전역 설정 완료")

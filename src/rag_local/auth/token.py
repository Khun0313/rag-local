"""ChatGPT Codex OAuth 토큰 관리.

~/.codex/auth.json에서 토큰을 읽고, API 호출용 헤더를 구성한다.
토큰 갱신은 외부 프로그램이 자동 처리하므로, 여기서는 읽기 + 헤더 구성만 수행한다.
만료 시 1회 자동 갱신을 시도하고, 실패하면 안내 메시지를 출력한다.

참고: auto-trade/auto_trader/llm/codex_auth.py
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from rag_local.config import settings

logger = logging.getLogger(__name__)

# ChatGPT Codex 백엔드 엔드포인트
CODEX_CHATGPT_BASE = "https://chatgpt.com/backend-api"
CODEX_RESPONSES_URL = f"{CODEX_CHATGPT_BASE}/codex/responses"

# OAuth 토큰 갱신 엔드포인트
OAUTH_TOKEN_ENDPOINT = "https://auth.openai.com/oauth/token"

# 만료 5분 전부터 선제 갱신
TOKEN_EXPIRY_BUFFER_SEC = 300


class TokenError(Exception):
    """토큰 관련 에러."""


def _read_auth_file() -> dict:
    """~/.codex/auth.json을 읽어 dict로 반환한다."""
    auth_path = settings.auth_token_path
    if not auth_path.exists():
        raise TokenError(
            f"토큰 파일을 찾을 수 없습니다: {auth_path}\n"
            "'codex login' 으로 먼저 인증을 완료해주세요."
        )
    try:
        return json.loads(auth_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise TokenError(f"auth.json 읽기 실패: {e}") from e


def _get_tokens_dict(auth: dict) -> dict:
    """auth.json에서 토큰 딕셔너리를 반환한다."""
    return auth.get("tokens") or auth


def get_access_token() -> str:
    """access_token을 반환한다."""
    auth = _read_auth_file()
    tokens = _get_tokens_dict(auth)
    for key in ("access_token", "accessToken", "token"):
        val = tokens.get(key)
        if val:
            logger.debug("access_token 로드 완료 (길이: %d)", len(val))
            return val
    raise TokenError("access_token을 찾을 수 없습니다.")


def get_refresh_token() -> Optional[str]:
    """refresh_token을 반환한다. 없으면 None."""
    auth = _read_auth_file()
    tokens = _get_tokens_dict(auth)
    for key in ("refresh_token", "refreshToken"):
        val = tokens.get(key)
        if val:
            return val
    return None


def get_account_id() -> Optional[str]:
    """account_id를 반환한다. 없으면 None."""
    auth = _read_auth_file()
    tokens = _get_tokens_dict(auth)
    return tokens.get("account_id") or None


def get_auth_mode() -> str:
    """auth.json의 auth_mode를 반환한다."""
    auth = _read_auth_file()
    return auth.get("auth_mode", "unknown")


def get_auth_headers() -> dict:
    """Codex 백엔드 API 호출용 헤더를 반환한다.

    auto-trade/codex_auth.py의 get_auth_headers()와 동일한 헤더 구성.
    """
    token = get_access_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "responses=experimental",
        "originator": "codex_cli_rs",
        "accept": "text/event-stream",
    }
    account_id = get_account_id()
    if account_id:
        headers["chatgpt-account-id"] = account_id
    return headers


def refresh_access_token() -> Optional[str]:
    """refresh_token으로 새 access_token을 발급받는다.

    외부 프로그램이 주로 갱신하지만, 만료 시 1회 자동 갱신 시도.
    """
    import requests

    refresh_token = get_refresh_token()
    if not refresh_token:
        logger.error("refresh_token이 없습니다. 재로그인이 필요합니다.")
        return None

    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }

    logger.info("OAuth 토큰 갱신 시도...")
    try:
        resp = requests.post(
            OAUTH_TOKEN_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        new_access_token = data.get("access_token")
        if not new_access_token:
            logger.error("갱신 응답에 access_token이 없습니다.")
            return None

        # auth.json 업데이트
        auth = _read_auth_file()
        tokens = _get_tokens_dict(auth)
        tokens["access_token"] = new_access_token
        new_refresh = data.get("refresh_token")
        if new_refresh:
            tokens["refresh_token"] = new_refresh

        auth_path = settings.auth_token_path
        auth_path.write_text(json.dumps(auth, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info("토큰 갱신 성공")
        return new_access_token

    except Exception as e:
        logger.error("토큰 갱신 실패: %s", e)
        return None


def is_logged_in() -> bool:
    """로그인 여부를 확인한다."""
    try:
        get_access_token()
        return True
    except TokenError:
        return False

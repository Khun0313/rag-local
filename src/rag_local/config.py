from pathlib import Path

from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # 경로
    documents_dir: Path = PROJECT_ROOT / "data" / "documents"
    chroma_persist_dir: Path = PROJECT_ROOT / "data" / "chroma_db"
    docstore_path: Path = PROJECT_ROOT / "data" / "docstore.json"
    ingestion_state_path: Path = PROJECT_ROOT / "data" / ".ingestion_state.json"
    auth_token_path: Path = Path.home() / ".codex" / "auth.json"

    # LLM (Codex 백엔드 사용 — chatgpt.com/backend-api/codex/responses)
    llm_model: str = "gpt-5.3-codex"

    # 임베딩
    embed_model: str = "upskyy/bge-m3-korean"

    # 청킹
    chunk_size: int = 512
    chunk_overlap: int = 64

    # 검색
    top_k: int = 5

    # API 서버
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Discord
    discord_bot_token: str = ""
    discord_channel_id: int = 0

    model_config = {"env_file": ".env", "env_prefix": "RAG_"}


settings = Settings()

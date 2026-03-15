"""
config.py — Centralised settings loaded from environment / .env file.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

_ENV_FILE = str(Path(__file__).parent.parent / ".env")

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Gemini ──────────────────────────────────────────────────────────────
    gemini_api_key: str = Field(..., alias="GEMINI_API_KEY")
    gemini_model: str = Field("gemini-1.5-flash", alias="GEMINI_MODEL")
    gemini_embedding_model: str = Field(
        "gemini-embedding-001", alias="GEMINI_EMBEDDING_MODEL"
    )

    # ── MongoDB ──────────────────────────────────────────────────────────────
    mongodb_uri: str = Field(..., alias="MONGODB_URI")
    mongodb_db: str = Field("aa_meetings", alias="MONGODB_DB")
    mongodb_collection: str = Field("meetings", alias="MONGODB_COLLECTION")

    # ── Vector search ─────────────────────────────────────────────────────
    vector_index_name: str = Field(
        "aa_meetings_vector_index", alias="VECTOR_INDEX_NAME"
    )
    embedding_dimensions: int = Field(3072, alias="EMBEDDING_DIMENSIONS")

    # ── Meeting Guide API ─────────────────────────────────────────────────
    meeting_guide_api_url: str = Field(
        "https://www.meetingguide.org/api/meetings",
        alias="MEETING_GUIDE_API_URL",
    )
    meeting_guide_region: str = Field("", alias="MEETING_GUIDE_REGION")

    # ── RAG ───────────────────────────────────────────────────────────────
    rag_top_k: int = Field(8, alias="RAG_TOP_K")
    rag_max_tokens: int = Field(1024, alias="RAG_MAX_TOKENS")

    # ── API server ────────────────────────────────────────────────────────
    api_host: str = Field("0.0.0.0", alias="API_HOST")
    api_port: int = Field(8000, alias="API_PORT")


# Singleton — import this everywhere
settings = Settings()
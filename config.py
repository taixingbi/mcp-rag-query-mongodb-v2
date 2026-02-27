# config.py
from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env", override=True)


class Settings(BaseSettings):
    """All config from .env; add new vars here with Field(..., alias="ENV_NAME")."""
    model_config = SettingsConfigDict(
        env_file=str(_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_version: str = Field(default="", alias="APP_VERSION")
    mcp_name: str = Field(default="mcp-tool-rag-query-mongodb-v1", alias="MCP_NAME")
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langchain_endpoint: str | None = Field(default=None, alias="LANGCHAIN_ENDPOINT")
    langchain_project: str | None = Field(default=None, alias="LANGCHAIN_PROJECT")
    langchain_api_key: str | None = Field(default=None, alias="LANGCHAIN_API_KEY")
    # Ollama (local or remote; for ECS set OLLAMA_BASE_URL to your Ollama service)
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_llm_model: str = Field(default="llama3.2", alias="OLLAMA_LLM_MODEL")
    ollama_embedding_model: str = Field(default="nomic-embed-text", alias="OLLAMA_EMBEDDING_MODEL")
    retrieval_k: int = Field(default=4, alias="RETRIEVAL_K")
    # MongoDB Atlas (dense + BM25)
    mongodb_uri: str | None = Field(default=None, alias="MONGODB_URI")
    mongodb_db: str = Field(default="db_hunt", alias="MONGODB_DB")
    mongodb_collection: str = Field(default="collection_taixingbi_dev", alias="MONGODB_COLLECTION")
    atlas_search_index: str = Field(default="default", alias="ATLAS_SEARCH_INDEX")
    atlas_search_path: str = Field(default="text", alias="ATLAS_SEARCH_PATH")
    atlas_vector_index: str = Field(default="vector_index", alias="ATLAS_VECTOR_INDEX")
    atlas_vector_path: str = Field(default="embedding", alias="ATLAS_VECTOR_PATH")
    # Dual recall + RRF
    top_k_dense: int = Field(default=50, alias="TOP_K_DENSE")
    top_k_bm25: int = Field(default=50, alias="TOP_K_BM25")
    top_k_final: int = Field(default=20, alias="TOP_K_FINAL")
    rrf_k: int = Field(default=60, alias="RRF_K")


@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()

_mongo_client = None


def get_mongodb_client():
    """Cached MongoDB client (thread-safe, long-lived singleton)."""
    global _mongo_client
    if _mongo_client is None:
        if not settings.mongodb_uri:
            raise RuntimeError("Missing MONGODB_URI. Check .env.")
        from pymongo import MongoClient
        _mongo_client = MongoClient(settings.mongodb_uri)
    return _mongo_client

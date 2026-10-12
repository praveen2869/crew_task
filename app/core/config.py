import os
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables or .env file.
    """
    # LLM Settings (OpenAI)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    # Jina AI Embeddings Settings
    jina_api_key: str = ""
    jina_embedding_model: str = "jina-embeddings-v5-omni-small"
    jina_embedding_dim: int = 1024

    # Qdrant Vector Database
    # If qdrant_url is empty/None, local embedded storage is used at qdrant_storage_path
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    qdrant_storage_path: str = "./qdrant_storage"
    qdrant_collection_name: str = "knowledge_base"

    # RAG Pipeline Configuration
    chunk_size: int = 800
    chunk_overlap: int = 150
    retrieval_top_k: int = 5
    score_threshold: float = 0.35

    # Server Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    streamlit_port: int = 8501
    log_level: str = "INFO"
    environment: str = "development"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.strip() and not self.openai_api_key.startswith("your_"))

    @property
    def has_jina_key(self) -> bool:
        return bool(self.jina_api_key and self.jina_api_key.strip() and not self.jina_api_key.startswith("your_"))


@lru_cache
def get_settings() -> Settings:
    """Returns cached application settings instance."""
    return Settings()

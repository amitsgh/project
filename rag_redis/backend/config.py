import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Redis Configuration
    redis_url: str = "redis://localhost:6379"
    redis_cache_ttl: int = 3600  # 1 hour

    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    # ollama_model: str = "gpt-oss:20b"
    ollama_model: str = "llama2"

    # Vector Store Configuration
    index_name: str = "documents"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    indexing: str = "HNSW"

    # Memory Configuration
    memory_key: str = "chat_history"
    memory_ttl: int = 1800  # 30 minutes

    # Performance Configuration
    max_tokens: int = 1000
    temperature: float = 0.7

    # Chunking Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200

    class Config:
        env_file = ".env"


settings = Settings()

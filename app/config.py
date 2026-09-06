"""
Centralized configuration management using Pydantic Settings.
All environment variables are validated at startup — fail fast, not at runtime.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Pydantic validates all values at startup — if any required
    variable is missing, the app crashes immediately with a clear error
    instead of failing hours later at runtime.
    """

    # --- Application ---
    APP_NAME: str = "BusinessAnalystAgent"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # --- LLM (OpenRouter — GPT models) ---
    OPENROUTER_API_KEY: str = Field(default="", description="OpenRouter API key")
    OPENROUTER_MODEL: str = Field(default="openai/gpt-4o-mini", description="Default model via OpenRouter")
    OPENROUTER_BASE_URL: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL (OpenAI-compatible)"
    )
    LLM_TEMPERATURE: float = Field(default=0.3, ge=0.0, le=2.0)
    LLM_TIMEOUT: int = Field(default=60, ge=10, description="LLM call timeout in seconds")

    # --- Database (PostgreSQL) ---
    DATABASE_URL: str = Field(
        default="postgresql+psycopg2://postgres:postgres@localhost:5433/business_analyst",
        description="PostgreSQL connection string"
    )
    DB_POOL_SIZE: int = Field(default=10, ge=1, le=50, description="Connection pool size")
    DB_MAX_OVERFLOW: int = Field(default=20, ge=0, le=100, description="Max overflow connections beyond pool size")
    DB_POOL_TIMEOUT: int = Field(default=30, ge=5, description="Seconds to wait for a connection from pool")
    DB_POOL_RECYCLE: int = Field(default=1800, description="Recycle connections after N seconds")
    DB_ECHO: bool = Field(default=False, description="Echo SQL statements for debugging")

    # --- Agent ---
    AGENT_RECURSION_LIMIT: int = Field(default=15, ge=3, le=50, description="Max graph steps before forced stop")
    AGENT_MAX_MESSAGES: int = Field(default=20, ge=5, le=100, description="Max messages kept in state")
    AGENT_TOOL_TIMEOUT: int = Field(default=30, ge=5, description="Per-tool execution timeout in seconds")

    # --- RAG ---
    CHROMA_PERSIST_DIR: str = "chroma_db"
    CHROMA_COLLECTION_NAME: str = "business_analyst_docs"
    RAG_CHUNK_SIZE: int = Field(default=1500, ge=200, le=5000)
    RAG_CHUNK_OVERLAP: int = Field(default=200, ge=0, le=500)
    RAG_TOP_K: int = Field(default=5, ge=1, le=20)

    # --- Data Analysis ---
    MAX_DATASET_ROWS: int = Field(default=50000, ge=100, description="Hard limit on rows per dataset")
    MAX_UPLOAD_SIZE_MB: int = Field(default=50, ge=1, le=200)

    # --- Circuit Breaker ---
    CB_FAILURE_THRESHOLD: int = Field(default=5, ge=1, description="Failures before circuit opens")
    CB_RECOVERY_TIMEOUT: int = Field(default=60, ge=10, description="Seconds before circuit half-opens")
    CB_HALF_OPEN_MAX_CALLS: int = Field(default=3, ge=1, description="Max test calls in half-open state")

    # --- Rate Limiting ---
    RATE_LIMIT_REQUESTS: int = Field(default=100, description="Requests per window")
    RATE_LIMIT_WINDOW: int = Field(default=60, description="Window in seconds")

    # --- Allowed Models (via OpenRouter) ---
    ALLOWED_MODELS: set = {
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
        "openai/gpt-4-turbo",
        "anthropic/claude-3.5-sonnet",
        "google/gemini-2.5-flash",
        "google/gemini-2.5-pro",
    }

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not v:
            raise ValueError("DATABASE_URL must be set")
        return v

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings singleton.
    Parsed once at startup, reused for all subsequent calls.
    """
    return Settings()

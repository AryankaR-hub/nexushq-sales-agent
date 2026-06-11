from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    anthropic_api_key: str
    database_url: str = "sqlite+aiosqlite:///./sales_agent.db"
    environment: str = "development"
    log_level: str = "INFO"
    agent_model: str = "claude-sonnet-4-20250514"
    eval_model: str = "claude-sonnet-4-20250514"
    max_memory_messages: int = 20          # messages kept in full before compression
    memory_summary_threshold: int = 15    # compress when history exceeds this
    confidence_flag_threshold: float = 0.60  # flag_for_human below this

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()

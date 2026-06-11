from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    gemini_api_key: str
    database_url: str = "sqlite+aiosqlite:///./sales_agent.db"
    environment: str = "development"
    log_level: str = "INFO"
    agent_model: str = "gemini-1.5-flash"
    eval_model: str = "gemini-1.5-flash"    
    max_memory_messages: int = 20
    memory_summary_threshold: int = 15
    confidence_flag_threshold: float = 0.60

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
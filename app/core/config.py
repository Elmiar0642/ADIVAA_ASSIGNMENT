from dataclasses import dataclass
from functools import lru_cache
import os


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(slots=True, frozen=True)
class Settings:
    app_name: str
    app_version: str
    environment: str
    log_level: str
    host: str
    port: int
    openai_api_key: str | None


@lru_cache
def get_settings() -> Settings:
    return Settings(
        app_name=os.getenv("APP_NAME", "Voice AI Agent"),
        app_version=os.getenv("APP_VERSION", "0.1.0"),
        environment=os.getenv("APP_ENV", "development"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        host=os.getenv("HOST", "0.0.0.0"),
        port=_get_int("PORT", 8000),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


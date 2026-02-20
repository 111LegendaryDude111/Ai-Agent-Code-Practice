from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _load_dotenv(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


def _read_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _read_env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw_value = _read_env(name)
    if raw_value is None:
        return max(minimum, default)

    try:
        parsed_value = int(raw_value)
    except ValueError:
        return max(minimum, default)

    return max(minimum, parsed_value)


@dataclass(frozen=True)
class Settings:
    app_env: str
    log_level: str
    database_url: str
    bot_token: str | None
    llm_api_key: str | None
    submission_rate_limit_count: int
    submission_rate_limit_window_seconds: int
    llm_rate_limit_count: int
    llm_rate_limit_window_seconds: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_dotenv()
    return Settings(
        app_env=_read_env("APP_ENV", "dev") or "dev",
        log_level=_read_env("LOG_LEVEL", "INFO") or "INFO",
        database_url=_read_env(
            "DATABASE_URL",
            "postgresql+psycopg://postgres:postgres@localhost:5432/interview_assistant",
        )
        or "postgresql+psycopg://postgres:postgres@localhost:5432/interview_assistant",
        bot_token=_read_env("BOT_TOKEN"),
        llm_api_key=_read_env("LLM_API_KEY"),
        submission_rate_limit_count=_read_env_int(
            "SUBMISSION_RATE_LIMIT_COUNT",
            default=15,
            minimum=1,
        ),
        submission_rate_limit_window_seconds=_read_env_int(
            "SUBMISSION_RATE_LIMIT_WINDOW_SECONDS",
            default=60,
            minimum=1,
        ),
        llm_rate_limit_count=_read_env_int(
            "LLM_RATE_LIMIT_COUNT",
            default=20,
            minimum=1,
        ),
        llm_rate_limit_window_seconds=_read_env_int(
            "LLM_RATE_LIMIT_WINDOW_SECONDS",
            default=60,
            minimum=1,
        ),
    )

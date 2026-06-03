import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


BACKEND_ROOT = Path(__file__).resolve().parents[2]


def load_environment(dotenv_path: str | Path | None = None, *, override: bool = False) -> None:
    load_dotenv(dotenv_path or BACKEND_ROOT / ".env", override=override)


load_environment()


def _split_csv(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str = "PostAI Backend"
    app_version: str = "0.1.0"
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    asset_dir: str = "generated"
    asset_url_path: str = "/assets"
    allow_model_fallback: bool = True
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str = "mock-text"
    llm_response_format: str = "json_schema"
    llm_temperature: float = 0.35
    llm_raw_temperature: float = 0.78
    vision_api_key: str | None = None
    vision_base_url: str | None = None
    vision_model: str = "mock-vision"
    vision_enable_thinking: bool = True
    vision_thinking_budget: int = 8192
    log_level: str = "INFO"
    log_file: str = ""


def get_settings() -> Settings:
    return Settings(
        app_name=os.getenv("APP_NAME", "PostAI Backend"),
        app_version=os.getenv("APP_VERSION", "0.1.0"),
        cors_origins=_split_csv(os.getenv("CORS_ORIGINS"), ["*"]),
        asset_dir=os.getenv("ASSET_DIR", "generated"),
        asset_url_path=os.getenv("ASSET_URL_PATH", "/assets"),
        allow_model_fallback=_env_bool("ALLOW_MODEL_FALLBACK", True),
        llm_api_key=os.getenv("LLM_API_KEY"),
        llm_base_url=os.getenv("LLM_BASE_URL"),
        llm_model=os.getenv("LLM_MODEL", "mock-text"),
        llm_response_format=os.getenv("LLM_RESPONSE_FORMAT", "json_schema"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.35")),
        llm_raw_temperature=float(os.getenv("LLM_RAW_TEMPERATURE", "0.78")),
        vision_api_key=os.getenv("VISION_API_KEY"),
        vision_base_url=os.getenv("VISION_BASE_URL"),
        vision_model=os.getenv("VISION_MODEL", "mock-vision"),
        vision_enable_thinking=_env_bool("VISION_ENABLE_THINKING", True),
        vision_thinking_budget=int(os.getenv("VISION_THINKING_BUDGET", "8192")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE", "logs/postai.log"),
    )

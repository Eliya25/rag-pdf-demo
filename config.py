"""Small, validated runtime configuration for both deployment modes."""
from dataclasses import dataclass
import os
from pathlib import Path


PUBLIC_DEMO_MODEL = "gemini-3.5-flash-lite"


def _integer(name: str, default: int, minimum: int, maximum: int) -> int:
    value = int(os.getenv(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True)
class Settings:
    provider: str
    gemini_model: str
    public_demo_mode: bool
    gemini_demo_enabled: bool
    top_k: int
    chunk_size: int
    chunk_overlap: int
    history_messages: int
    max_file_mb: int
    max_pages: int
    max_chunks: int
    max_generation_tokens: int
    max_text_chars: int
    max_question_chars: int


def load_settings() -> Settings:
    public_demo_mode = Path(__file__).with_name("PUBLIC_DEMO_MODE").is_file()
    # Invalid and missing values both fail closed for the public demo.
    gemini_demo_enabled = os.getenv("GEMINI_DEMO_ENABLED", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    provider = os.getenv("LLM_PROVIDER", "gemini" if public_demo_mode else "local").lower().strip()
    if provider not in {"local", "gemini"}:
        raise ValueError("LLM_PROVIDER must be local or gemini")
    gemini_model = os.getenv("GEMINI_MODEL", PUBLIC_DEMO_MODEL)
    if public_demo_mode and provider != "gemini":
        raise ValueError("Public demo mode requires the Gemini provider")
    if public_demo_mode and gemini_model != PUBLIC_DEMO_MODEL:
        raise ValueError("Public demo mode requires the approved Gemini model")

    chunk_size = _integer("CHUNK_SIZE", 800, 100, 2000)
    overlap = _integer("CHUNK_OVERLAP", 120, 0, chunk_size - 1)
    return Settings(
        provider=provider,
        gemini_model=gemini_model,
        public_demo_mode=public_demo_mode,
        gemini_demo_enabled=gemini_demo_enabled,
        top_k=4 if public_demo_mode else _integer("TOP_K", 5, 1, 20),
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        history_messages=4 if public_demo_mode else _integer("HISTORY_MESSAGES", 6, 0, 20),
        max_file_mb=5 if public_demo_mode else _integer("MAX_FILE_MB", 10, 1, 10),
        max_pages=25 if public_demo_mode else _integer("MAX_PAGES", 100, 1, 500),
        max_chunks=100 if public_demo_mode else _integer("MAX_CHUNKS", 500, 1, 2000),
        max_generation_tokens=256 if public_demo_mode else _integer("MAX_GENERATION_TOKENS", 512, 32, 2048),
        max_text_chars=100_000 if public_demo_mode else 500_000,
        max_question_chars=1_000 if public_demo_mode else 10_000,
    )


def validate_public_gemini_request(settings: Settings) -> None:
    """Fail before key access or Gemini client construction when disallowed."""
    if not settings.public_demo_mode:
        return
    if not settings.gemini_demo_enabled:
        raise ValueError("Public Gemini demo is disabled")
    if settings.provider != "gemini" or settings.gemini_model != PUBLIC_DEMO_MODEL:
        raise ValueError("Public Gemini demo configuration is invalid")

import pytest
from pathlib import Path

import config
from config import load_settings, validate_public_gemini_request


def enable_marker(monkeypatch, tmp_path):
    fake_config = tmp_path / "config.py"
    monkeypatch.setattr(config, "__file__", str(fake_config))
    fake_config.with_name("PUBLIC_DEMO_MODE").write_text("public snapshot\n")


def test_defaults_select_local_and_keep_bounded_limits(monkeypatch):
    for name in ("LLM_PROVIDER", "GEMINI_MODEL", "TOP_K", "CHUNK_SIZE",
                 "CHUNK_OVERLAP", "HISTORY_MESSAGES", "MAX_FILE_MB", "MAX_PAGES",
                 "MAX_CHUNKS", "MAX_GENERATION_TOKENS", "PUBLIC_DEMO_MODE",
                 "GEMINI_DEMO_ENABLED"):
        monkeypatch.delenv(name, raising=False)

    settings = load_settings()
    assert settings.gemini_model == "gemini-3.5-flash-lite"
    assert settings.gemini_demo_enabled is False
    if Path(__file__).resolve().parents[1].joinpath("PUBLIC_DEMO_MODE").is_file():
        assert settings.provider == "gemini"
        assert settings.public_demo_mode is True
        assert settings.top_k == 4
    else:
        assert settings.provider == "local"
        assert settings.public_demo_mode is False
        assert settings.top_k == 5


def test_public_demo_applies_fixed_limits_and_defaults_to_gemini(monkeypatch, tmp_path):
    enable_marker(monkeypatch, tmp_path)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setenv("MAX_FILE_MB", "10")
    monkeypatch.setenv("MAX_PAGES", "100")
    monkeypatch.setenv("MAX_CHUNKS", "500")
    monkeypatch.setenv("TOP_K", "20")
    monkeypatch.setenv("HISTORY_MESSAGES", "20")
    monkeypatch.setenv("MAX_GENERATION_TOKENS", "2048")

    settings = load_settings()

    assert settings.public_demo_mode is True
    assert settings.provider == "gemini"
    assert settings.gemini_demo_enabled is False
    assert settings.max_file_mb == 5
    assert settings.max_pages == 25
    assert settings.max_text_chars == 100_000
    assert settings.max_chunks == 100
    assert settings.max_question_chars == 1_000
    assert settings.top_k == 4
    assert settings.history_messages == 4
    assert settings.max_generation_tokens == 256


@pytest.mark.parametrize("value", [None, "false", "invalid", "", "2"])
def test_public_demo_enable_flag_fails_closed(value, monkeypatch, tmp_path):
    enable_marker(monkeypatch, tmp_path)
    if value is None:
        monkeypatch.delenv("GEMINI_DEMO_ENABLED", raising=False)
    else:
        monkeypatch.setenv("GEMINI_DEMO_ENABLED", value)
    assert load_settings().gemini_demo_enabled is False


def test_public_demo_accepts_only_gemini_and_fixed_model(monkeypatch, tmp_path):
    enable_marker(monkeypatch, tmp_path)
    monkeypatch.setenv("LLM_PROVIDER", "local")
    with pytest.raises(ValueError, match="requires the Gemini provider"):
        load_settings()

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_MODEL", "another-model")
    with pytest.raises(ValueError, match="requires the approved Gemini model"):
        load_settings()


@pytest.mark.parametrize("name,value", [
    ("LLM_PROVIDER", "remote"),
    ("TOP_K", "0"),
    ("CHUNK_SIZE", "99"),
    ("CHUNK_OVERLAP", "800"),
    ("HISTORY_MESSAGES", "21"),
    ("MAX_FILE_MB", "11"),
])
def test_invalid_configuration_is_rejected(name, value, monkeypatch):
    monkeypatch.setenv(name, value)
    public_snapshot = Path(__file__).resolve().parents[1].joinpath("PUBLIC_DEMO_MODE").is_file()
    if public_snapshot and name in {"TOP_K", "HISTORY_MESSAGES", "MAX_FILE_MB"}:
        settings = load_settings()
        assert settings.top_k == 4
        assert settings.history_messages == 4
        assert settings.max_file_mb == 5
        return
    with pytest.raises(ValueError):
        load_settings()


def test_gemini_mode_configuration_does_not_require_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", " GEMINI ")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert load_settings().provider == "gemini"


def test_public_gemini_request_validation_is_fail_closed(monkeypatch, tmp_path):
    enable_marker(monkeypatch, tmp_path)
    monkeypatch.setenv("GEMINI_DEMO_ENABLED", "true")
    settings = load_settings()
    validate_public_gemini_request(settings)

    monkeypatch.setenv("GEMINI_DEMO_ENABLED", "not-a-boolean")
    disabled = load_settings()
    with pytest.raises(ValueError, match="disabled"):
        validate_public_gemini_request(disabled)


def test_public_mode_requires_generated_marker_and_ignores_environment(monkeypatch, tmp_path):
    fake_config = tmp_path / "config.py"
    monkeypatch.setattr(config, "__file__", str(fake_config))
    monkeypatch.setenv("PUBLIC_DEMO_MODE", "true")

    absent = load_settings()
    assert absent.public_demo_mode is False
    assert absent.provider == "local"

    fake_config.with_name("PUBLIC_DEMO_MODE").write_text("public snapshot\n")
    monkeypatch.setenv("PUBLIC_DEMO_MODE", "false")
    present = load_settings()
    assert present.public_demo_mode is True
    assert present.provider == "gemini"


def test_public_mode_marker_absence_ignores_truthy_environment_override(monkeypatch, tmp_path):
    fake_config = tmp_path / "config.py"
    monkeypatch.setattr(config, "__file__", str(fake_config))
    monkeypatch.setenv("PUBLIC_DEMO_MODE", "yes")

    settings = load_settings()

    assert settings.public_demo_mode is False
    assert settings.provider == "local"

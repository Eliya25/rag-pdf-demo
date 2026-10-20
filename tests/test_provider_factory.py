from dataclasses import replace
from pathlib import Path

import pytest

import config
from config import load_settings
from provider_factory import create_provider


def public_settings(monkeypatch, tmp_path, enabled):
    fake_config = tmp_path / "config.py"
    monkeypatch.setattr(config, "__file__", str(fake_config))
    fake_config.with_name("PUBLIC_DEMO_MODE").write_text("public snapshot\n")
    monkeypatch.setenv("GEMINI_DEMO_ENABLED", enabled)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    return load_settings()


@pytest.mark.parametrize("flag", ["false", "invalid", ""])
def test_disabled_or_invalid_public_mode_never_reads_key_or_constructs_client(
    flag, monkeypatch, tmp_path
):
    settings = public_settings(monkeypatch, tmp_path, flag)
    key_reads = []
    client_constructions = []

    def fail_key_read():
        key_reads.append(True)
        raise AssertionError("key access must not occur")

    def fail_client(**kwargs):
        client_constructions.append(kwargs)
        raise AssertionError("client construction must not occur")

    from google import genai

    monkeypatch.setattr(genai, "Client", fail_client)
    with pytest.raises(ValueError, match="disabled"):
        create_provider(settings, fail_key_read, lambda: object())
    assert key_reads == []
    assert client_constructions == []


def test_public_provider_uses_fixed_model_and_never_calls_local_loader(monkeypatch, tmp_path):
    settings = public_settings(monkeypatch, tmp_path, "true")
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    from google import genai

    monkeypatch.setattr(genai, "Client", FakeClient)
    provider = create_provider(settings, lambda: "test-placeholder", lambda: pytest.fail("local loader called"))

    assert provider.model_name == "gemini-3.5-flash-lite"
    assert captured["http_options"].retry_options.attempts == 1


def test_private_default_calls_existing_local_provider_loader(monkeypatch):
    fake_config = Path(__file__).parent / "_no_public_marker" / "config.py"
    monkeypatch.setattr(config, "__file__", str(fake_config))
    monkeypatch.delenv("PUBLIC_DEMO_MODE", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    settings = load_settings()
    sentinel = object()
    assert create_provider(settings, lambda: pytest.fail("API key read"), lambda: sentinel) is sentinel


@pytest.mark.parametrize(
    "overrides", [{"provider": "local"}, {"gemini_model": "other-model"}]
)
def test_invalid_public_provider_or_model_never_reads_key(overrides, monkeypatch, tmp_path):
    settings = replace(public_settings(monkeypatch, tmp_path, "true"), **overrides)
    from google import genai

    monkeypatch.setattr(
        genai, "Client", lambda **kwargs: pytest.fail("Gemini client constructed")
    )
    with pytest.raises(ValueError):
        create_provider(
            settings,
            lambda: pytest.fail("API key read"),
            lambda: pytest.fail("local provider constructed"),
        )

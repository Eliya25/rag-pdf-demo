from prompts import build_messages


class FakeProvider:
    model_name = "fake"

    def __init__(self):
        self.calls = []

    def stream(self, messages, max_tokens):
        self.calls.append((messages, max_tokens))
        yield "answer"


def test_provider_receives_retrieved_context_without_pdf_bytes():
    provider = FakeProvider()
    messages = build_messages("question", "[Page 1, Chunk 1] relevant text")
    assert "".join(provider.stream(messages, 128)) == "answer"
    assert provider.calls[0][1] == 128
    assert "relevant text" in provider.calls[0][0][-1]["content"]


def test_gemini_stream_uses_system_instruction_and_retrieved_context():
    from gemini_llm import GeminiLLMProvider

    class Models:
        def __init__(self):
            self.kwargs = None

        def generate_content_stream(self, **kwargs):
            self.kwargs = kwargs
            return [type("Chunk", (), {"text": "reply"})()]

    provider = GeminiLLMProvider.__new__(GeminiLLMProvider)
    provider.model_name = "fake-gemini"
    models = Models()
    provider.client = type("Client", (), {"models": models})()
    messages = build_messages("question", "document excerpt")
    assert list(provider.stream(messages, 100)) == ["reply"]
    assert models.kwargs["config"].system_instruction == messages[0]["content"]
    assert "document excerpt" in models.kwargs["contents"][-1].parts[0].text


def test_gemini_requires_credentials_without_disclosing_them():
    from gemini_llm import GeminiLLMProvider
    import pytest

    with pytest.raises(ValueError, match="GEMINI_API_KEY is required"):
        GeminiLLMProvider("")


def test_gemini_client_is_configured_for_one_attempt(monkeypatch):
    from google import genai
    from gemini_llm import GeminiLLMProvider

    captured = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(genai, "Client", fake_client)
    GeminiLLMProvider("test-only-placeholder", disable_retries=True)

    assert captured["http_options"].retry_options.attempts == 1


def test_gemini_default_mode_keeps_existing_client_options(monkeypatch):
    from google import genai
    from gemini_llm import GeminiLLMProvider

    captured = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(genai, "Client", fake_client)
    GeminiLLMProvider("test-only-placeholder")

    assert captured == {"api_key": "test-only-placeholder"}


def test_gemini_request_includes_selected_history_and_question():
    from gemini_llm import GeminiLLMProvider

    class Models:
        def generate_content_stream(self, **kwargs):
            self.kwargs = kwargs
            return []

    provider = GeminiLLMProvider.__new__(GeminiLLMProvider)
    provider.model_name = "fake"
    models = Models()
    provider.client = type("Client", (), {"models": models})()
    messages = build_messages("current question", "retrieved excerpt", [
        {"role": "user", "content": "selected prior question"},
        {"role": "assistant", "content": "selected prior response"},
    ])
    list(provider.stream(messages, 80))
    sent = " ".join(part.text for turn in models.kwargs["contents"]
                    for part in turn.parts)
    assert "selected prior question" in sent
    assert "selected prior response" in sent
    assert "retrieved excerpt" in sent
    assert "current question" in sent
    assert "%PDF" not in sent


def test_gemini_stream_propagates_failure_after_partial_chunk():
    from gemini_llm import GeminiLLMProvider

    class Models:
        def generate_content_stream(self, **kwargs):
            yield type("Chunk", (), {"text": "incomplete"})()
            raise RuntimeError("provider disconnected")

    provider = GeminiLLMProvider.__new__(GeminiLLMProvider)
    provider.model_name = "fake"
    provider.client = type("Client", (), {"models": Models()})()
    stream = provider.stream([{"role": "system", "content": "s"}], 10)
    assert next(stream) == "incomplete"
    import pytest
    with pytest.raises(RuntimeError, match="disconnected"):
        next(stream)

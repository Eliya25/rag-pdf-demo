import logging

from ui_errors import (GENERATION_ERROR_MESSAGE,
                       PUBLIC_DEMO_GENERATION_ERROR_MESSAGE,
                       safe_value_error_message, stream_answer)


class Placeholder:
    def __init__(self):
        self.cleared = False
        self.rendered = []

    def write_stream(self, chunks):
        result = ""
        for chunk in chunks:
            self.rendered.append(chunk)
            result += chunk
        return result

    def empty(self):
        self.cleared = True


class FakeProvider:
    def __init__(self, failure=None):
        self.failure = failure

    def stream(self, messages, max_tokens):
        yield "partial answer"
        if self.failure:
            raise self.failure
        yield " complete"


def test_completed_stream_returns_answer_without_clearing(caplog):
    placeholder = Placeholder()
    errors = []
    answer = stream_answer(FakeProvider(), [], 32, placeholder, errors.append,
                           logging.getLogger("test-ui"))
    assert answer == "partial answer complete"
    assert not placeholder.cleared
    assert errors == []


def test_rate_limit_failure_clears_partial_answer_and_hides_exception(caplog):
    secret = "fake-api-key-never-log"
    document_text = "private synthetic PDF excerpt"
    placeholder = Placeholder()
    errors = []
    with caplog.at_level(logging.ERROR, logger="test-ui-errors"):
        answer = stream_answer(
            FakeProvider(RuntimeError(f"429 rate limit; {secret}; {document_text}")),
            [], 32, placeholder, errors.append, logging.getLogger("test-ui-errors"),
        )

    assert answer is None
    assert placeholder.cleared
    assert errors == [GENERATION_ERROR_MESSAGE]
    assert "429" not in errors[0]
    assert secret not in caplog.text and document_text not in caplog.text
    assert "Traceback" not in caplog.text
    assert "Answer generation failed" in caplog.text


def test_interrupted_stream_clears_partial_answer_and_reports_safe_message():
    placeholder = Placeholder()
    errors = []
    answer = stream_answer(FakeProvider(ConnectionError("stream interrupted")), [], 32,
                           placeholder, errors.append, logging.getLogger("test-ui"))
    assert answer is None
    assert placeholder.cleared
    assert errors == [GENERATION_ERROR_MESSAGE]


def test_public_demo_provider_error_uses_safe_non_quota_specific_message():
    placeholder = Placeholder()
    errors = []
    answer = stream_answer(
        FakeProvider(RuntimeError("provider response with sensitive detail")),
        [], 32, placeholder, errors.append, logging.getLogger("test-ui"),
        error_message=PUBLIC_DEMO_GENERATION_ERROR_MESSAGE,
    )
    assert answer is None
    assert placeholder.cleared
    assert errors == [PUBLIC_DEMO_GENERATION_ERROR_MESSAGE]
    assert "sensitive detail" not in errors[0]


def test_unknown_validation_exception_text_is_not_shown():
    private_detail = "private document parser detail and fake-key"
    assert safe_value_error_message(
        ValueError(private_detail), {"known safe message"}, "Could not continue."
    ) == "Could not continue."
    assert safe_value_error_message(
        ValueError("known safe message"), {"known safe message"}, "Could not continue."
    ) == "known safe message"

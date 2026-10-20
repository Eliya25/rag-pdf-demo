"""Small UI boundary for failed, possibly partial provider streams."""

GENERATION_ERROR_MESSAGE = (
    "Could not generate an answer. Check the model settings or API quota and try again."
)
PUBLIC_DEMO_GENERATION_ERROR_MESSAGE = (
    "Could not generate an answer. The demo may be temporarily unavailable; try again later."
)


def stream_answer(
    provider,
    messages,
    max_tokens,
    placeholder,
    show_error,
    logger,
    error_message=GENERATION_ERROR_MESSAGE,
):
    """Render a completed answer or clear partial output and show a safe error."""
    try:
        return placeholder.write_stream(provider.stream(messages, max_tokens))
    except Exception:
        # Provider exceptions can contain request data, response bodies, or secrets.
        # Keep diagnostics free of exception text and traceback.
        placeholder.empty()
        logger.error("Answer generation failed")
        show_error(error_message)
        return None


def safe_value_error_message(error, allowed_messages, fallback):
    """Return only a known application message, never arbitrary exception text."""
    message = str(error)
    return message if message in allowed_messages else fallback

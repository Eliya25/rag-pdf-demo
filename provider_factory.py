"""Provider selection with public-demo checks before secret access."""

from config import Settings, validate_public_gemini_request


def create_provider(settings: Settings, api_key_loader, local_provider_loader):
    if settings.public_demo_mode:
        validate_public_gemini_request(settings)
    if settings.provider == "local":
        if settings.public_demo_mode:
            raise ValueError("Public demo mode requires Gemini")
        return local_provider_loader()

    from gemini_llm import GeminiLLMProvider

    return GeminiLLMProvider(
        api_key_loader(),
        settings.gemini_model,
        disable_retries=settings.public_demo_mode,
    )

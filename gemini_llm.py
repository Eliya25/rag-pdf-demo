"""Gemini generation; retrieval and document processing stay local."""
from typing import Iterator


class GeminiLLMProvider:
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3.5-flash-lite",
        disable_retries: bool = False,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini mode")
        from google import genai

        self.model_name = model_name
        client_options = {"api_key": api_key}
        if disable_retries:
            from google.genai import types

            client_options["http_options"] = types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1),
            )
        self.client = genai.Client(**client_options)

    def stream(self, messages: list[dict], max_tokens: int) -> Iterator[str]:
        from google.genai import types

        system = messages[0]["content"]
        turns = [types.Content(role="model" if m["role"] == "assistant" else "user",
                               parts=[types.Part.from_text(text=m["content"])]) for m in messages[1:]]
        responses = self.client.models.generate_content_stream(
            model=self.model_name,
            contents=turns,
            config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=max_tokens),
        )
        for response in responses:
            if response.text:
                yield response.text

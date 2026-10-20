"""Minimal streaming provider contract."""
from typing import Iterator, Protocol


class LLMProvider(Protocol):
    model_name: str

    def stream(self, messages: list[dict], max_tokens: int) -> Iterator[str]: ...

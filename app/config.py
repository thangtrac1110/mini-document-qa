"""Application settings with explicit, documented defaults.

The API key is deliberately *not* a setting: it is read from ``ANTHROPIC_API_KEY`` only
where the LLM client is built, so it never appears in a settings repr or log line.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Chunk size in words (whitespace-separated). Chosen a priori, not tuned on the eval set.
    chunk_size_words: int = 100
    max_upload_bytes: int = 20 * 1024 * 1024

    llm_model: str = "claude-opus-5-5"
    llm_effort: str = "medium"  # low | medium | high | xhigh | max
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2  # the SDK retries 429, 5xx and connection errors
    llm_max_tokens: int = 16000  # includes the model's thinking tokens

    @classmethod
    def from_env(cls) -> "Settings":
        defaults = cls()
        return cls(
            llm_model=os.environ.get("LLM_MODEL", defaults.llm_model),
            llm_effort=os.environ.get("LLM_EFFORT", defaults.llm_effort),
            llm_timeout_seconds=float(os.environ.get("LLM_TIMEOUT_SECONDS", defaults.llm_timeout_seconds)),
        )

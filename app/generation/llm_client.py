"""Provider wrapper. All Anthropic-specific code lives here; the rest of the app only
sees the ``LLMClient`` protocol and the ``LLMError`` subclasses in ``app.errors``.
"""

import json
from typing import Protocol

import anthropic

from app.config import Settings
from app.errors import (
    LLMConfigurationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMTimeoutError,
    MalformedLLMResponseError,
)


class LLMClient(Protocol):
    def complete_json(self, *, system: str, user: str, schema: dict) -> dict:
        """Return the model's reply parsed as a JSON object matching ``schema``."""
        ...


class AnthropicLLMClient:
    # Server-side refusal fallback: if a safety classifier declines, the API retries on
    # the model Anthropic recommends for that refusal category within the same call.
    _FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(self, api_key: str, settings: Settings):
        self._settings = settings
        self._client = anthropic.Anthropic(
            api_key=api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    def complete_json(self, *, system: str, user: str, schema: dict) -> dict:
        try:
            response = self._client.beta.messages.create(
                model=self._settings.llm_model,
                max_tokens=self._settings.llm_max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={
                    "effort": self._settings.llm_effort,
                    "format": {"type": "json_schema", "schema": schema},
                },
                betas=[self._FALLBACK_BETA],
                fallbacks="default",
            )
        # Most specific first: APITimeoutError subclasses APIConnectionError, and the
        # status-specific errors subclass APIStatusError. The SDK has already retried
        # 429, 5xx and connection errors by the time these reach us.
        except anthropic.APITimeoutError as exc:
            raise LLMTimeoutError("The LLM provider timed out") from exc
        except anthropic.RateLimitError as exc:
            raise LLMRateLimitError("The LLM provider is rate limiting requests; try again later") from exc
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise LLMConfigurationError("The LLM provider rejected the configured credentials") from exc
        except anthropic.APIStatusError as exc:
            raise LLMProviderError(f"The LLM provider returned HTTP {exc.status_code}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMProviderError("Could not reach the LLM provider") from exc

        if response.stop_reason == "refusal":
            raise LLMRefusalError("The model declined to answer this request")
        if response.stop_reason == "max_tokens":
            raise MalformedLLMResponseError("The model output was truncated")

        # Thinking (and fallback) blocks may precede the answer; take the text block.
        text = next((block.text for block in response.content if block.type == "text"), None)
        if text is None:
            raise MalformedLLMResponseError("The model returned no text")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MalformedLLMResponseError("The model output is not valid JSON") from exc
        if not isinstance(data, dict):
            raise MalformedLLMResponseError("The model output is not a JSON object")
        return data

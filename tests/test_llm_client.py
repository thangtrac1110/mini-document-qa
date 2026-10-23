"""AnthropicLLMClient tests with the SDK call replaced: no network, no API key."""

import json
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from app.config import Settings
from app.errors import (
    LLMConfigurationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMTimeoutError,
    MalformedLLMResponseError,
)
from app.generation.llm_client import AnthropicLLMClient

SCHEMA = {"type": "object"}
REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def status_error(cls, code: int):
    return cls("error", response=httpx2.Response(code, request=REQUEST), body=None)


def text_response(text: str, stop_reason: str = "end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
    )


def make_client(result) -> tuple[AnthropicLLMClient, list[dict]]:
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if isinstance(result, Exception):
            raise result
        return result

    client = AnthropicLLMClient("test-key", Settings(llm_model="test-model", llm_effort="low"))
    client._client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    return client, calls


def test_returns_parsed_json_and_sends_expected_request():
    payload = {"answer": "x", "supporting_source_ids": ["S1"], "insufficient_information": False}
    client, calls = make_client(text_response(json.dumps(payload)))

    result = client.complete_json(system="sys", user="question", schema=SCHEMA)

    assert result == payload
    (request,) = calls
    assert request["model"] == "test-model"
    assert request["system"] == "sys"
    assert request["messages"] == [{"role": "user", "content": "question"}]
    assert request["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}}
    assert request["fallbacks"] == "default"
    assert request["betas"] == ["server-side-fallback-2026-07-01"]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (anthropic.APITimeoutError(request=REQUEST), LLMTimeoutError),
        (status_error(anthropic.RateLimitError, 429), LLMRateLimitError),
        (status_error(anthropic.AuthenticationError, 401), LLMConfigurationError),
        (status_error(anthropic.PermissionDeniedError, 403), LLMConfigurationError),
        (status_error(anthropic.InternalServerError, 529), LLMProviderError),
        (status_error(anthropic.BadRequestError, 400), LLMProviderError),
        (anthropic.APIConnectionError(request=REQUEST), LLMProviderError),
    ],
)
def test_provider_errors_become_app_errors(error, expected):
    client, _ = make_client(error)

    with pytest.raises(expected):
        client.complete_json(system="s", user="u", schema=SCHEMA)


def test_error_messages_do_not_leak_the_api_key():
    client, _ = make_client(status_error(anthropic.AuthenticationError, 401))

    with pytest.raises(LLMConfigurationError) as excinfo:
        client.complete_json(system="s", user="u", schema=SCHEMA)

    assert "test-key" not in str(excinfo.value)


def test_refusal_is_reported():
    client, _ = make_client(SimpleNamespace(stop_reason="refusal", content=[]))

    with pytest.raises(LLMRefusalError):
        client.complete_json(system="s", user="u", schema=SCHEMA)


def client_over_transport(handler) -> AnthropicLLMClient:
    """Real SDK client whose HTTP layer is a local handler: exercises request
    serialization and response parsing without network access."""
    client = AnthropicLLMClient("test-key", Settings(llm_model="claude-opus-5-5"))
    client._client = anthropic.Anthropic(
        api_key="test-key", max_retries=0, http_client=httpx2.Client(transport=httpx2.MockTransport(handler))
    )
    return client


def test_real_sdk_request_and_response_round_trip():
    payload = {"answer": "In 1998.", "supporting_source_ids": ["S2"], "insufficient_information": False}
    seen = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["url"] = str(request.url)
        seen["beta"] = request.headers.get("anthropic-beta")
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json={
            "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [
                {"type": "thinking", "thinking": "", "signature": "sig"},
                {"type": "text", "text": json.dumps(payload)},
            ],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 50, "output_tokens": 20},
        })

    result = client_over_transport(handler).complete_json(system="sys", user="q", schema=SCHEMA)

    assert result == payload
    assert seen["url"].endswith("/v1/messages?beta=true")
    assert "server-side-fallback-2026-07-01" in seen["beta"]
    body = seen["body"]
    assert body["model"] == "claude-opus-5-5"
    assert body["fallbacks"] == "default"
    assert body["output_config"]["format"] == {"type": "json_schema", "schema": SCHEMA}
    assert body["output_config"]["effort"] == "medium"
    assert "thinking" not in body  # Opus 5.5 always thinks adaptively; omit the parameter


def test_real_sdk_http_429_maps_to_rate_limit_error():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(429, json={"type": "error", "error": {"type": "rate_limit_error", "message": "slow"}})

    with pytest.raises(LLMRateLimitError):
        client_over_transport(handler).complete_json(system="s", user="u", schema=SCHEMA)


@pytest.mark.parametrize(
    "response",
    [
        text_response('{"answer": "cut off', stop_reason="max_tokens"),
        text_response("not json"),
        text_response('["a", "list"]'),
        SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="thinking", thinking="")]),
    ],
)
def test_malformed_output_raises(response):
    client, _ = make_client(response)

    with pytest.raises(MalformedLLMResponseError):
        client.complete_json(system="s", user="u", schema=SCHEMA)

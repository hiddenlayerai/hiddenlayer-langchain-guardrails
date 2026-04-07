"""Integration tests for the asynchronous AsyncHiddenLayerGuardrail."""
from unittest.mock import AsyncMock, Mock, patch

import pytest

from hiddenlayer_langchain_guardrails.middleware import (
    AsyncHiddenLayerGuardrail,
    HiddenLayerParams,
    InputBlockedError,
)

pytestmark = pytest.mark.asyncio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALLOW_INPUT = {"messages": [{"role": "user", "content": "hello"}]}
BLOCK_INPUT = {"choices": [{"message": {"content": "blocked"}, "finish_reason": "stop"}]}


def _allow_output(content: str) -> dict:
    """Return an output evaluation response that echoes the same content (allow)."""
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}


def _redact_input(new_content: str) -> dict:
    return {"messages": [{"role": "user", "content": new_content}]}


def _redact_output(new_content: str) -> dict:
    return {"choices": [{"message": {"content": new_content}, "finish_reason": "stop"}]}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def async_client():
    client = Mock()
    client.post = AsyncMock()
    return client


@pytest.fixture
def guardrail(async_client):
    params = HiddenLayerParams(model="gpt-4", project_id="proj-1", requester_id="test-user")
    return AsyncHiddenLayerGuardrail(params=params, client=async_client)


@pytest.fixture(autouse=True)
def mock_get_config():
    with patch("hiddenlayer_langchain_guardrails.middleware.get_config") as m:
        m.return_value = {}
        yield m


@pytest.fixture
def make_request(dummy_request_classes):
    Msg, Req = dummy_request_classes

    def _make(content="hello"):
        return Req([Msg("system msg", type="system"), Msg(content, type="human")])

    return _make


@pytest.fixture
def make_response(dummy_response_class):
    return dummy_response_class


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_awrap_model_call_allow_passthrough(
    guardrail, async_client, make_request, make_response, make_http_response
):
    async_client.post.side_effect = [
        make_http_response(ALLOW_INPUT),
        make_http_response(_allow_output("world")),
    ]

    req = make_request("hello")

    async def handler(r):
        return make_response("world")

    resp = await guardrail.awrap_model_call(req, handler)
    assert resp.message.content == "world"
    assert async_client.post.call_count == 2


@pytest.mark.asyncio
async def test_async_awrap_model_call_block_input_raises(
    guardrail, async_client, make_request, make_http_response
):
    async_client.post.side_effect = [make_http_response(BLOCK_INPUT)]

    req = make_request("hello")

    async def handler(_):
        raise AssertionError("handler should not be called")

    with pytest.raises(InputBlockedError):
        await guardrail.awrap_model_call(req, handler)

    assert async_client.post.call_count == 1


@pytest.mark.asyncio
async def test_async_awrap_model_call_redact_input_replaces_last_message(
    guardrail, async_client, make_request, make_response, make_http_response
):
    async_client.post.side_effect = [
        make_http_response(_redact_input("REDACTED_IN")),
        make_http_response(_allow_output("ok")),
    ]

    req = make_request("SECRET")
    received = []

    async def handler(r):
        received.append(r.messages[-1].content)
        return make_response("ok")

    await guardrail.awrap_model_call(req, handler)
    assert received == ["REDACTED_IN"]


@pytest.mark.asyncio
async def test_async_awrap_model_call_redact_output_mutates_response(
    guardrail, async_client, make_request, make_response, make_http_response
):
    async_client.post.side_effect = [
        make_http_response(ALLOW_INPUT),
        make_http_response(_redact_output("REDACTED_OUT")),
    ]

    req = make_request("hello")

    async def handler(_):
        return make_response("SECRET_OUT")

    resp = await guardrail.awrap_model_call(req, handler)
    assert resp.message.content == "REDACTED_OUT"


@pytest.mark.asyncio
async def test_async_awrap_model_call_empty_messages_skips_input_analysis(
    guardrail, async_client, make_response, make_http_response, dummy_request_classes
):
    _, Req = dummy_request_classes
    req = Req([])  # no messages → body["messages"] is [] → falsy → skip input POST

    async_client.post.side_effect = [make_http_response(_allow_output("ok"))]

    async def handler(_):
        return make_response("ok")

    resp = await guardrail.awrap_model_call(req, handler)
    assert resp.message.content == "ok"
    assert async_client.post.call_count == 1


@pytest.mark.asyncio
async def test_async_awrap_model_call_no_output_content_skips_output_analysis(
    guardrail, async_client, make_request, make_http_response, dummy_response_class
):
    async_client.post.side_effect = [make_http_response(ALLOW_INPUT)]

    req = make_request("hello")

    async def handler(_):
        return dummy_response_class(content=None)

    resp = await guardrail.awrap_model_call(req, handler)
    assert resp.message.content is None
    assert async_client.post.call_count == 1


@pytest.mark.asyncio
async def test_async_awrap_model_call_session_id_from_config(
    guardrail, async_client, make_request, make_response, make_http_response, mock_get_config
):
    mock_get_config.return_value = {"metadata": {"thread_id": "sess-xyz"}}

    async_client.post.side_effect = [
        make_http_response(ALLOW_INPUT),
        make_http_response(_allow_output("world")),
    ]

    req = make_request("hello")

    async def handler(_):
        return make_response("world")

    await guardrail.awrap_model_call(req, handler)

    assert async_client.post.call_count == 2


@pytest.mark.asyncio
async def test_async_awrap_model_call_with_tools_in_request_body(
    guardrail, async_client, make_response, make_http_response, dummy_request_classes
):
    Msg, Req = dummy_request_classes
    tool = {"name": "search", "description": "searches things", "parameters": {"type": "object", "properties": {}}}
    req = Req([Msg("hello", type="human")], tools=[tool])

    async_client.post.side_effect = [
        make_http_response(ALLOW_INPUT),
        make_http_response(_allow_output("result")),
    ]

    async def handler(_):
        return make_response("result")

    resp = await guardrail.awrap_model_call(req, handler)
    assert resp.message.content == "result"

    call_kwargs = async_client.post.call_args_list[0][1]
    assert "tools" in call_kwargs["body"]
    assert call_kwargs["body"]["tools"][0]["function"]["name"] == "search"


@pytest.mark.asyncio
async def test_async_awrap_model_call_with_system_message(
    guardrail, async_client, make_response, make_http_response, dummy_request_classes
):
    Msg, Req = dummy_request_classes
    sys_msg = Msg("Be concise", type="system")
    req = Req([Msg("hi", type="human")], system_message=sys_msg)

    async_client.post.side_effect = [
        make_http_response(ALLOW_INPUT),
        make_http_response(_allow_output("ok")),
    ]

    async def handler(_):
        return make_response("ok")

    await guardrail.awrap_model_call(req, handler)

    call_kwargs = async_client.post.call_args_list[0][1]
    messages = call_kwargs["body"]["messages"]
    assert messages[0] == {"role": "system", "content": "Be concise"}

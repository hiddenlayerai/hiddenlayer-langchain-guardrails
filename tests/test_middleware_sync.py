"""Integration tests for the synchronous HiddenLayerGuardrail."""
from unittest.mock import Mock, patch

import pytest

from hiddenlayer_langchain_guardrails.middleware import (
    HiddenLayerGuardrail,
    HiddenLayerParams,
    InputBlockedError,
)

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
def sync_client():
    client = Mock()
    client.post = Mock()
    return client


@pytest.fixture
def guardrail(sync_client):
    params = HiddenLayerParams(model="gpt-4", project_id="proj-1", requester_id="test-user")
    return HiddenLayerGuardrail(params=params, client=sync_client)


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


def test_sync_wrap_model_call_allow_passthrough(guardrail, sync_client, make_request, make_response, make_http_response):
    sync_client.post.side_effect = [
        make_http_response(ALLOW_INPUT),
        make_http_response(_allow_output("world")),
    ]

    req = make_request("hello")

    def handler(r):
        return make_response("world")

    resp = guardrail.wrap_model_call(req, handler)
    assert resp.message.content == "world"
    assert sync_client.post.call_count == 2


def test_sync_wrap_model_call_block_input_raises(guardrail, sync_client, make_request, make_http_response):
    sync_client.post.side_effect = [make_http_response(BLOCK_INPUT)]

    req = make_request("hello")

    def handler(_):
        raise AssertionError("handler should not be called")

    with pytest.raises(InputBlockedError):
        guardrail.wrap_model_call(req, handler)

    assert sync_client.post.call_count == 1


def test_sync_wrap_model_call_redact_input_replaces_last_message(
    guardrail, sync_client, make_request, make_response, make_http_response
):
    sync_client.post.side_effect = [
        make_http_response(_redact_input("REDACTED_IN")),
        make_http_response(_allow_output("ok")),
    ]

    req = make_request("SECRET")
    received = []

    def handler(r):
        received.append(r.messages[-1].content)
        return make_response("ok")

    guardrail.wrap_model_call(req, handler)
    assert received == ["REDACTED_IN"]


def test_sync_wrap_model_call_redact_output_mutates_response(
    guardrail, sync_client, make_request, make_response, make_http_response
):
    sync_client.post.side_effect = [
        make_http_response(ALLOW_INPUT),
        make_http_response(_redact_output("REDACTED_OUT")),
    ]

    req = make_request("hello")

    def handler(_):
        return make_response("SECRET_OUT")

    resp = guardrail.wrap_model_call(req, handler)
    assert resp.message.content == "REDACTED_OUT"


def test_sync_wrap_model_call_empty_messages_skips_input_analysis(
    guardrail, sync_client, make_response, make_http_response, dummy_request_classes
):
    _, Req = dummy_request_classes
    req = Req([])  # no messages → body["messages"] is [] → falsy → skip input POST

    sync_client.post.side_effect = [make_http_response(_allow_output("ok"))]

    def handler(_):
        return make_response("ok")

    resp = guardrail.wrap_model_call(req, handler)
    assert resp.message.content == "ok"
    assert sync_client.post.call_count == 1


def test_sync_wrap_model_call_no_output_content_skips_output_analysis(
    guardrail, sync_client, make_request, make_http_response, dummy_response_class
):
    sync_client.post.side_effect = [make_http_response(ALLOW_INPUT)]

    req = make_request("hello")

    def handler(_):
        return dummy_response_class(content=None)

    resp = guardrail.wrap_model_call(req, handler)
    assert resp.message.content is None
    assert sync_client.post.call_count == 1


def test_sync_wrap_model_call_session_id_from_config(
    guardrail, sync_client, make_request, make_response, make_http_response, mock_get_config
):
    mock_get_config.return_value = {"metadata": {"thread_id": "sess-abc"}}

    sync_client.post.side_effect = [
        make_http_response(ALLOW_INPUT),
        make_http_response(_allow_output("world")),
    ]

    req = make_request("hello")
    guardrail.wrap_model_call(req, lambda _: make_response("world"))

    assert sync_client.post.call_count == 2


def test_sync_wrap_model_call_with_tools_in_request_body(
    guardrail, sync_client, make_response, make_http_response, dummy_request_classes
):
    Msg, Req = dummy_request_classes
    tool = {"name": "calculator", "description": "does math", "parameters": {"type": "object", "properties": {}}}
    req = Req([Msg("hello", type="human")], tools=[tool])

    sync_client.post.side_effect = [
        make_http_response(ALLOW_INPUT),
        make_http_response(_allow_output("42")),
    ]

    def handler(_):
        return make_response("42")

    resp = guardrail.wrap_model_call(req, handler)
    assert resp.message.content == "42"

    call_kwargs = sync_client.post.call_args_list[0][1]
    assert "tools" in call_kwargs["body"]
    assert call_kwargs["body"]["tools"][0]["function"]["name"] == "calculator"


def test_sync_wrap_model_call_with_system_message(
    guardrail, sync_client, make_response, make_http_response, dummy_request_classes
):
    Msg, Req = dummy_request_classes
    sys_msg = Msg("You are helpful", type="system")
    req = Req([Msg("hi", type="human")], system_message=sys_msg)

    sync_client.post.side_effect = [
        make_http_response(ALLOW_INPUT),
        make_http_response(_allow_output("ok")),
    ]

    guardrail.wrap_model_call(req, lambda _: make_response("ok"))

    call_kwargs = sync_client.post.call_args_list[0][1]
    messages = call_kwargs["body"]["messages"]
    assert messages[0] == {"role": "system", "content": "You are helpful"}

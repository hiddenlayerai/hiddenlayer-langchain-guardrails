import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from hiddenlayer_langchain_guardrails.middleware import (
    AsyncHiddenLayerGuardrail,
    HiddenLayerActions,
    HiddenLayerParams,
    InputBlockedError,
    OutputBlockedError,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def async_client_mock():
    client = Mock()
    client.interactions = Mock()
    client.interactions.analyze = AsyncMock()
    return client


@pytest.fixture
def guardrail_async(async_client_mock):
    params = HiddenLayerParams(model="m", project_id="p", requester_id="r")
    return AsyncHiddenLayerGuardrail(params=params, client=async_client_mock)


@pytest.fixture
def make_request(dummy_request_classes):
    Msg, Req = dummy_request_classes

    def _make(last_content):
        return Req([Msg("system"), Msg(last_content)])

    return _make


@pytest.fixture
def make_response(dummy_response_class):
    Resp = dummy_response_class
    return Resp


@pytest.mark.asyncio
async def test_async_awrap_model_call_allow_passthrough(
    guardrail_async, async_client_mock, make_hl_response, make_request, make_response
):
    async_client_mock.interactions.analyze.side_effect = [
        make_hl_response(action=None, role="user"),
        make_hl_response(action=None, role="assistant"),
    ]

    req = make_request("hello")

    async def handler(r):
        return make_response("world")

    resp = await guardrail_async.awrap_model_call(req, handler)
    assert resp.message.content == "world"
    assert async_client_mock.interactions.analyze.call_count == 2


@pytest.mark.asyncio
async def test_async_awrap_model_call_block_input_raises(
    guardrail_async, async_client_mock, make_hl_response, make_request
):
    async_client_mock.interactions.analyze.side_effect = [
        make_hl_response(action=HiddenLayerActions.BLOCK, role="user")
    ]

    req = make_request("hello")

    async def handler(_):
        raise AssertionError("handler should not be called")

    with pytest.raises(InputBlockedError):
        await guardrail_async.awrap_model_call(req, handler)


@pytest.mark.asyncio
async def test_async_awrap_model_call_redact_input_replaces_last_message(
    guardrail_async, async_client_mock, make_hl_response, make_request, make_response
):
    async_client_mock.interactions.analyze.side_effect = [
        make_hl_response(action=HiddenLayerActions.REDACT, role="user", redacted_text="REDACTED_IN"),
        make_hl_response(action=None, role="assistant"),
    ]

    req = make_request("SECRET")

    async def handler(r):
        assert r.messages[-1].content == "REDACTED_IN"
        return make_response("ok")

    resp = await guardrail_async.awrap_model_call(req, handler)
    assert resp.message.content == "ok"


@pytest.mark.asyncio
async def test_async_awrap_model_call_block_output_raises(
    guardrail_async, async_client_mock, make_hl_response, make_request, make_response
):
    async_client_mock.interactions.analyze.side_effect = [
        make_hl_response(action=None, role="user"),
        make_hl_response(action=HiddenLayerActions.BLOCK, role="assistant"),
    ]

    req = make_request("hello")

    async def handler(_):
        return make_response("bad")

    with pytest.raises(OutputBlockedError):
        await guardrail_async.awrap_model_call(req, handler)


@pytest.mark.asyncio
async def test_async_awrap_model_call_redact_output_mutates_response(
    guardrail_async, async_client_mock, make_hl_response, make_request, make_response
):
    async_client_mock.interactions.analyze.side_effect = [
        make_hl_response(action=None, role="user"),
        make_hl_response(action=HiddenLayerActions.REDACT, role="assistant", redacted_text="REDACTED_OUT"),
    ]

    req = make_request("hello")

    async def handler(_):
        return make_response("SECRET_OUT")

    resp = await guardrail_async.awrap_model_call(req, handler)
    assert resp.message.content == "REDACTED_OUT"


@pytest.mark.asyncio
async def test_async_awrap_model_call_no_input_content_skips_input_analysis(
    guardrail_async, async_client_mock, make_hl_response, dummy_request_classes, make_response
):
    Msg, Req = dummy_request_classes
    req = Req([Msg("system"), Msg(None)])

    async_client_mock.interactions.analyze.side_effect = [make_hl_response(action=None, role="assistant")]

    async def handler(_):
        return make_response("ok")

    resp = await guardrail_async.awrap_model_call(req, handler)
    assert resp.message.content == "ok"
    assert async_client_mock.interactions.analyze.call_count == 1


@pytest.mark.asyncio
async def test_async_awrap_model_call_no_output_content_skips_output_analysis(
    guardrail_async, async_client_mock, make_hl_response, make_request
):
    async_client_mock.interactions.analyze.side_effect = [make_hl_response(action=None, role="user")]

    req = make_request("hello")

    async def handler(_):
        return SimpleNamespace(message=SimpleNamespace(content=None))

    resp = await guardrail_async.awrap_model_call(req, handler)
    assert resp.message.content is None
    assert async_client_mock.interactions.analyze.call_count == 1


@pytest.mark.asyncio
async def test_async_awrap_tool_call_allow_passthrough(
    guardrail_async, async_client_mock, make_hl_response, dummy_tool_request_class
):
    async_client_mock.interactions.analyze.side_effect = [
        make_hl_response(action=None, role="user"),
        make_hl_response(action=None, role="assistant"),
    ]

    ToolReq = dummy_tool_request_class
    req = ToolReq({"name": "t", "args": {"x": 1}})

    async def handler(r):
        assert r is req
        return {"ok": True}

    out = await guardrail_async.awrap_tool_call(req, handler)
    assert out == {"ok": True}


@pytest.mark.asyncio
async def test_async_awrap_tool_call_block_input_raises(
    guardrail_async, async_client_mock, make_hl_response, dummy_tool_request_class
):
    async_client_mock.interactions.analyze.side_effect = [
        make_hl_response(action=HiddenLayerActions.BLOCK, role="user")
    ]

    ToolReq = dummy_tool_request_class
    req = ToolReq({"name": "t", "args": {"x": "secret"}})

    async def handler(_):
        raise AssertionError("handler should not run")

    with pytest.raises(InputBlockedError):
        await guardrail_async.awrap_tool_call(req, handler)


@pytest.mark.asyncio
async def test_async_awrap_tool_call_redact_input_applies_args_to_request(
    guardrail_async, async_client_mock, make_hl_response, dummy_tool_request_class
):
    async_client_mock.interactions.analyze.side_effect = [
        make_hl_response(
            action=HiddenLayerActions.REDACT,
            role="user",
            redacted_text=json.dumps({"args": {"x": "REDACTED"}}),
        ),
        make_hl_response(action=None, role="assistant"),
    ]

    ToolReq = dummy_tool_request_class
    req = ToolReq({"name": "t", "args": {"x": "secret"}})

    async def handler(r):
        assert r.tool_call["args"] == {"x": "REDACTED"}
        return "OK"

    out = await guardrail_async.awrap_tool_call(req, handler)
    assert out == "OK"


@pytest.mark.asyncio
async def test_async_awrap_tool_call_block_output_raises(
    guardrail_async, async_client_mock, make_hl_response, dummy_tool_request_class
):
    async_client_mock.interactions.analyze.side_effect = [
        make_hl_response(action=None, role="user"),
        make_hl_response(action=HiddenLayerActions.BLOCK, role="assistant"),
    ]

    ToolReq = dummy_tool_request_class
    req = ToolReq({"name": "t", "args": {"x": 1}})

    async def handler(_):
        return "BAD"

    with pytest.raises(OutputBlockedError):
        await guardrail_async.awrap_tool_call(req, handler)


@pytest.mark.asyncio
async def test_async_awrap_tool_call_redact_output_returns_redacted_content(
    guardrail_async, async_client_mock, make_hl_response, dummy_tool_request_class
):
    async_client_mock.interactions.analyze.side_effect = [
        make_hl_response(action=None, role="user"),
        make_hl_response(action=HiddenLayerActions.REDACT, role="user", redacted_text="REDACTED_TOOL_OUT"),
    ]

    ToolReq = dummy_tool_request_class
    req = ToolReq({"name": "t", "args": {"x": 1}})

    async def handler(_):
        return "SECRET_TOOL_OUT"

    out = await guardrail_async.awrap_tool_call(req, handler)
    assert out == "REDACTED_TOOL_OUT"


@pytest.mark.asyncio
async def test_async_awrap_tool_call_missing_tool_call_fields_are_safe(
    guardrail_async, async_client_mock, make_hl_response, dummy_tool_request_class
):
    async_client_mock.interactions.analyze.side_effect = [
        make_hl_response(action=None, role="user"),
        make_hl_response(action=None, role="assistant"),
    ]

    ToolReq = dummy_tool_request_class
    req = ToolReq(tool_call={})

    async def handler(_):
        return "OK"

    out = await guardrail_async.awrap_tool_call(req, handler)
    assert out == "OK"

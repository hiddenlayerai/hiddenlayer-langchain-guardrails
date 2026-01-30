import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hiddenlayer_langchain_guardrails.middleware import (
    HiddenLayerActions,
    HiddenLayerGuardrail,
    HiddenLayerParams,
    InputBlockedError,
    OutputBlockedError,
)


@pytest.fixture
def sync_client_mock():
    client = Mock()
    client.interactions = Mock()
    client.interactions.analyze = Mock()
    return client


@pytest.fixture
def guardrail_sync(sync_client_mock):
    params = HiddenLayerParams(model="m", project_id="p", requester_id="r")
    return HiddenLayerGuardrail(params=params, client=sync_client_mock)


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


def test_sync_wrap_model_call_allow_passthrough(
    guardrail_sync, sync_client_mock, make_hl_response, make_request, make_response
):
    sync_client_mock.interactions.analyze.side_effect = [
        make_hl_response(action=None, role="user"),
        make_hl_response(action=None, role="assistant"),
    ]

    req = make_request("hello")

    def handler(r):
        return make_response("world")

    resp = guardrail_sync.wrap_model_call(req, handler)
    assert resp.message.content == "world"
    assert sync_client_mock.interactions.analyze.call_count == 2


def test_sync_wrap_model_call_block_input_raises(guardrail_sync, sync_client_mock, make_hl_response, make_request):
    sync_client_mock.interactions.analyze.side_effect = [make_hl_response(action=HiddenLayerActions.BLOCK, role="user")]

    req = make_request("hello")

    def handler(_):
        raise AssertionError("handler should not be called")

    with pytest.raises(InputBlockedError):
        guardrail_sync.wrap_model_call(req, handler)


def test_sync_wrap_model_call_redact_input_replaces_last_message(
    guardrail_sync, sync_client_mock, make_hl_response, make_request, make_response
):
    sync_client_mock.interactions.analyze.side_effect = [
        make_hl_response(action=HiddenLayerActions.REDACT, role="user", redacted_text="REDACTED_IN"),
        make_hl_response(action=None, role="assistant"),
    ]

    req = make_request("SECRET")

    def handler(r):
        # handler should see redacted last message content
        assert r.messages[-1].content == "REDACTED_IN"
        return make_response("ok")

    resp = guardrail_sync.wrap_model_call(req, handler)
    assert resp.message.content == "ok"


def test_sync_wrap_model_call_block_output_raises(
    guardrail_sync, sync_client_mock, make_hl_response, make_request, make_response
):
    sync_client_mock.interactions.analyze.side_effect = [
        make_hl_response(action=None, role="user"),
        make_hl_response(action=HiddenLayerActions.BLOCK, role="assistant"),
    ]

    req = make_request("hello")

    def handler(_):
        return make_response("bad")

    with pytest.raises(OutputBlockedError):
        guardrail_sync.wrap_model_call(req, handler)


def test_sync_wrap_model_call_redact_output_mutates_response(
    guardrail_sync, sync_client_mock, make_hl_response, make_request, make_response
):
    sync_client_mock.interactions.analyze.side_effect = [
        make_hl_response(action=None, role="user"),
        make_hl_response(action=HiddenLayerActions.REDACT, role="assistant", redacted_text="REDACTED_OUT"),
    ]

    req = make_request("hello")

    def handler(_):
        return make_response("SECRET_OUT")

    resp = guardrail_sync.wrap_model_call(req, handler)
    assert resp.message.content == "REDACTED_OUT"


def test_sync_wrap_model_call_no_input_content_skips_input_analysis(
    guardrail_sync, sync_client_mock, make_hl_response, dummy_request_classes, make_response
):
    Msg, Req = dummy_request_classes
    req = Req([Msg("system"), Msg(None)])  # last content None

    sync_client_mock.interactions.analyze.side_effect = [make_hl_response(action=None, role="assistant")]

    def handler(_):
        return make_response("ok")

    resp = guardrail_sync.wrap_model_call(req, handler)
    assert resp.message.content == "ok"
    assert sync_client_mock.interactions.analyze.call_count == 1  # only output


def test_sync_wrap_model_call_no_output_content_skips_output_analysis(
    guardrail_sync, sync_client_mock, make_hl_response, make_request
):
    sync_client_mock.interactions.analyze.side_effect = [make_hl_response(action=None, role="user")]

    req = make_request("hello")

    def handler(_):
        return SimpleNamespace(message=SimpleNamespace(content=None))

    resp = guardrail_sync.wrap_model_call(req, handler)
    assert resp.message.content is None
    assert sync_client_mock.interactions.analyze.call_count == 1  # only input


def test_sync_wrap_tool_call_allow_passthrough(
    guardrail_sync, sync_client_mock, make_hl_response, dummy_tool_request_class
):
    sync_client_mock.interactions.analyze.side_effect = [
        make_hl_response(action=None, role="user"),
        make_hl_response(action=None, role="assistant"),
    ]

    ToolReq = dummy_tool_request_class
    req = ToolReq({"name": "echo", "args": {"x": 1}})

    def handler(r):
        assert r is req
        return {"ok": True}

    out = guardrail_sync.wrap_tool_call(req, handler)
    assert out == {"ok": True}


def test_sync_wrap_tool_call_block_input_raises(
    guardrail_sync, sync_client_mock, make_hl_response, dummy_tool_request_class
):
    sync_client_mock.interactions.analyze.side_effect = [make_hl_response(action=HiddenLayerActions.BLOCK, role="user")]

    ToolReq = dummy_tool_request_class
    req = ToolReq({"name": "t", "args": {"x": "secret"}})

    def handler(_):
        raise AssertionError("handler should not run")

    with pytest.raises(InputBlockedError):
        guardrail_sync.wrap_tool_call(req, handler)


def test_sync_wrap_tool_call_redact_input_applies_args_to_request(
    guardrail_sync, sync_client_mock, make_hl_response, dummy_tool_request_class
):
    # input redaction returns JSON payload {"args": {...}}
    sync_client_mock.interactions.analyze.side_effect = [
        make_hl_response(
            action=HiddenLayerActions.REDACT, role="user", redacted_text=json.dumps({"args": {"x": "REDACTED"}})
        ),
        make_hl_response(action=None, role="assistant"),
    ]

    ToolReq = dummy_tool_request_class
    req = ToolReq({"name": "t", "args": {"x": "secret"}})

    def handler(r):
        assert r.tool_call["args"] == {"x": "REDACTED"}
        return "OK"

    out = guardrail_sync.wrap_tool_call(req, handler)
    assert out == "OK"


def test_sync_wrap_tool_call_block_output_raises(
    guardrail_sync, sync_client_mock, make_hl_response, dummy_tool_request_class
):
    sync_client_mock.interactions.analyze.side_effect = [
        make_hl_response(action=None, role="user"),
        make_hl_response(action=HiddenLayerActions.BLOCK, role="assistant"),
    ]

    ToolReq = dummy_tool_request_class
    req = ToolReq({"name": "t", "args": {"x": 1}})

    def handler(_):
        return "BAD"

    with pytest.raises(OutputBlockedError):
        guardrail_sync.wrap_tool_call(req, handler)


def test_sync_wrap_tool_call_redact_output_returns_redacted_content(
    guardrail_sync, sync_client_mock, make_hl_response, dummy_tool_request_class
):
    sync_client_mock.interactions.analyze.side_effect = [
        make_hl_response(action=None, role="user"),
        make_hl_response(action=HiddenLayerActions.REDACT, role="assistant", redacted_text="REDACTED_TOOL_OUT"),
    ]

    ToolReq = dummy_tool_request_class
    req = ToolReq({"name": "t", "args": {"x": 1}})

    def handler(_):
        return "SECRET_TOOL_OUT"

    out = guardrail_sync.wrap_tool_call(req, handler)
    assert out == "REDACTED_TOOL_OUT"


def test_sync_wrap_tool_call_missing_tool_call_fields_are_safe(
    guardrail_sync, sync_client_mock, make_hl_response, dummy_tool_request_class
):
    sync_client_mock.interactions.analyze.side_effect = [
        make_hl_response(action=None, role="user"),
        make_hl_response(action=None, role="assistant"),
    ]

    ToolReq = dummy_tool_request_class
    req = ToolReq(tool_call={})  # no name/args

    def handler(_):
        return "OK"

    out = guardrail_sync.wrap_tool_call(req, handler)
    assert out == "OK"

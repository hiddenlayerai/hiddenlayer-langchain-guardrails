import json
from types import SimpleNamespace

import pytest

from hiddenlayer_langchain_guardrails.middleware import (
    HiddenLayerActions,
    HiddenLayerParams,
    _apply_tool_input_redaction,
    _build_analyze_kwargs,
    _extract_result,
    _get_request_last_content,
    _get_response_content,
    _replace_last_message,
    _set_response_content,
)


def test_build_analyze_kwargs_user_includes_input_and_metadata():
    params = HiddenLayerParams(model="m1", project_id="p1", requester_id="r1")
    out = _build_analyze_kwargs("hello", "user", params)

    assert out["metadata"] == {"model": "m1", "requester_id": "r1"}
    assert out["hl_project_id"] == "p1"
    assert "input" in out and "output" not in out
    assert out["input"]["messages"][-1] == {"role": "user", "content": "hello"}


def test_build_analyze_kwargs_assistant_includes_output_and_no_project_id_when_none():
    params = HiddenLayerParams(model=None, project_id=None, requester_id="r1")
    out = _build_analyze_kwargs("ok", "assistant", params)

    assert out["metadata"] == {"model": None, "requester_id": "r1"}
    assert "hl_project_id" not in out
    assert "output" in out and "input" not in out
    assert out["output"]["messages"][-1] == {"role": "assistant", "content": "ok"}


def test_extract_result_allow_no_modified(make_hl_response):
    resp = make_hl_response(action=None, role="user")
    res = _extract_result(resp, "user")
    assert res.block is False
    assert res.redact is False
    assert res.redacted_content is None


def test_extract_result_block(make_hl_response):
    resp = make_hl_response(action=HiddenLayerActions.BLOCK, role="user")
    res = _extract_result(resp, "user")
    assert res.block is True
    assert res.redact is False
    assert res.redacted_content is None


def test_extract_result_redact_user_pulls_modified_input(make_hl_response):
    resp = make_hl_response(action=HiddenLayerActions.REDACT, role="user", redacted_text="X")
    res = _extract_result(resp, "user")
    assert res.block is False
    assert res.redact is True
    assert res.redacted_content == "X"


def test_extract_result_redact_assistant_pulls_modified_output(make_hl_response):
    resp = make_hl_response(action=HiddenLayerActions.REDACT, role="assistant", redacted_text="Y")
    res = _extract_result(resp, "assistant")
    assert res.redact is True
    assert res.redacted_content == "Y"


def test_extract_result_redact_missing_modified_data_is_safe():
    resp = SimpleNamespace(evaluation=SimpleNamespace(action=HiddenLayerActions.REDACT), modified_data=None)
    res = _extract_result(resp, "user")
    assert res.redact is True
    assert res.redacted_content is None


def test_get_request_last_content_handles_empty_and_non_string(dummy_request_classes):
    Msg, Req = dummy_request_classes

    assert _get_request_last_content(Req([])) is None
    assert _get_request_last_content(Req([Msg(None)])) is None
    assert _get_request_last_content(Req([Msg(123)])) is None
    assert _get_request_last_content(Req([Msg("")])) is None
    assert _get_request_last_content(Req([Msg("hi")])) == "hi"


def test_replace_last_message_replaces_only_last(dummy_request_classes):
    Msg, Req = dummy_request_classes
    req = Req([Msg("a"), Msg("b")])

    new_req = _replace_last_message(req, "X")
    assert [m.content for m in req.messages] == ["a", "b"]  # original unchanged
    assert [m.content for m in new_req.messages] == ["a", "X"]


def test_get_and_set_response_content(dummy_response_class):
    Resp = dummy_response_class
    resp = Resp("hello")

    assert _get_response_content(resp) == "hello"
    _set_response_content(resp, "X")
    assert _get_response_content(resp) == "X"

    # If message is None, setter should be safe
    resp2 = SimpleNamespace(message=None)
    _set_response_content(resp2, "Y")  # no error


def test_apply_tool_input_redaction_valid_json_updates_args(dummy_tool_request_class):
    ToolReq = dummy_tool_request_class
    req = ToolReq({"name": "t", "args": {"a": "secret"}})

    redacted = json.dumps({"args": {"a": "REDACTED"}}, ensure_ascii=False)
    out = _apply_tool_input_redaction(req, redacted)

    assert out is req  # in-place
    assert req.tool_call["args"] == {"a": "REDACTED"}


@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        json.dumps(["args", {"a": 1}]),
        json.dumps({"noargs": {}}),
        json.dumps({"args": "not-a-dict"}),
    ],
)
def test_apply_tool_input_redaction_invalid_payload_no_change(dummy_tool_request_class, payload):
    ToolReq = dummy_tool_request_class
    req = ToolReq({"name": "t", "args": {"a": "secret"}})

    out = _apply_tool_input_redaction(req, payload)
    assert out is req
    assert req.tool_call["args"] == {"a": "secret"}


def test_apply_tool_input_redaction_non_dict_tool_call_no_change(dummy_tool_request_class):
    ToolReq = dummy_tool_request_class
    req = ToolReq(tool_call=None)

    redacted = json.dumps({"args": {"a": "REDACTED"}})
    out = _apply_tool_input_redaction(req, redacted)
    assert out is req

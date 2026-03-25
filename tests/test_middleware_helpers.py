"""Unit tests for pure helper functions in middleware.py."""
import json
from types import SimpleNamespace

import pytest

from hiddenlayer_langchain_guardrails.middleware import (
    AnalysisResult,
    _extract_input_result,
    _extract_output_result,
    _format_tool_call,
    _get_response_output,
    _to_openai_messages,
    _tool_to_openai,
)


# ---------------------------------------------------------------------------
# _extract_input_result
# ---------------------------------------------------------------------------


def test_extract_input_result_block_when_choices_present():
    data = {"choices": [{"message": {"content": "Sorry, blocked."}, "finish_reason": "stop"}]}
    result = _extract_input_result(data, original_messages=[{"role": "user", "content": "hi"}])
    assert result == AnalysisResult(block=True, redact=False, redacted_content=None)


def test_extract_input_result_allow_when_content_unchanged():
    original = [{"role": "user", "content": "hello"}]
    result = _extract_input_result({"messages": original}, original_messages=original)
    assert result == AnalysisResult(block=False, redact=False, redacted_content=None)


def test_extract_input_result_redact_when_last_message_content_differs():
    original = [{"role": "user", "content": "hello"}]
    returned = [{"role": "user", "content": "REDACTED"}]
    result = _extract_input_result({"messages": returned}, original_messages=original)
    assert result == AnalysisResult(block=False, redact=True, redacted_content="REDACTED")


def test_extract_input_result_allow_without_original_messages():
    data = {"messages": [{"role": "user", "content": "hi"}]}
    result = _extract_input_result(data)
    assert result == AnalysisResult(block=False, redact=False, redacted_content=None)


def test_extract_input_result_allow_when_messages_empty():
    result = _extract_input_result({})
    assert result == AnalysisResult(block=False, redact=False, redacted_content=None)


# ---------------------------------------------------------------------------
# _extract_output_result
# ---------------------------------------------------------------------------


def test_extract_output_result_allow_no_choices():
    result = _extract_output_result({})
    assert result == AnalysisResult(block=False, redact=False, redacted_content=None)


def test_extract_output_result_allow_when_content_matches():
    data = {"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]}
    result = _extract_output_result(data, original_content="hello")
    assert result == AnalysisResult(block=False, redact=False, redacted_content=None)


def test_extract_output_result_redact_when_content_differs():
    data = {"choices": [{"message": {"content": "REDACTED"}, "finish_reason": "stop"}]}
    result = _extract_output_result(data, original_content="hello")
    assert result == AnalysisResult(block=False, redact=True, redacted_content="REDACTED")


def test_extract_output_result_allow_no_original_content():
    data = {"choices": [{"message": {"content": "something"}, "finish_reason": "stop"}]}
    result = _extract_output_result(data, original_content=None)
    assert result == AnalysisResult(block=False, redact=False, redacted_content=None)


# ---------------------------------------------------------------------------
# _format_tool_call
# ---------------------------------------------------------------------------


def test_format_tool_call_basic():
    tc = {"id": "call_1", "name": "my_tool", "args": {"x": 1}}
    out = _format_tool_call(tc)
    assert out == {
        "id": "call_1",
        "type": "function",
        "function": {"name": "my_tool", "arguments": json.dumps({"x": 1})},
    }


def test_format_tool_call_missing_fields_use_defaults():
    out = _format_tool_call({})
    assert out["id"] == ""
    assert out["function"]["name"] == ""
    assert out["function"]["arguments"] == "{}"


# ---------------------------------------------------------------------------
# _to_openai_messages
# ---------------------------------------------------------------------------


def _msg(type_, content, tool_calls=None, tool_call_id=None):
    ns = SimpleNamespace(type=type_, content=content, tool_calls=tool_calls, tool_call_id=tool_call_id)
    return ns


def test_to_openai_messages_human():
    msgs = [_msg("human", "hello")]
    out = _to_openai_messages(msgs)
    assert out == [{"role": "user", "content": "hello"}]


def test_to_openai_messages_ai():
    msgs = [_msg("ai", "world")]
    out = _to_openai_messages(msgs)
    assert out == [{"role": "assistant", "content": "world"}]


def test_to_openai_messages_system():
    msgs = [_msg("system", "You are a bot")]
    out = _to_openai_messages(msgs)
    assert out == [{"role": "system", "content": "You are a bot"}]


def test_to_openai_messages_tool():
    msgs = [_msg("tool", "result text", tool_call_id="call_1")]
    out = _to_openai_messages(msgs)
    assert out == [{"role": "tool", "content": "result text", "tool_call_id": "call_1"}]


def test_to_openai_messages_assistant_with_tool_calls():
    tc = {"id": "c1", "name": "fn", "args": {"a": 1}}
    msgs = [_msg("ai", None, tool_calls=[tc])]
    out = _to_openai_messages(msgs)
    assert len(out) == 1
    assert out[0]["role"] == "assistant"
    assert out[0]["content"] is None
    assert out[0]["tool_calls"][0]["function"]["name"] == "fn"


def test_to_openai_messages_non_string_content_serialized():
    msgs = [_msg("human", {"key": "val"})]
    out = _to_openai_messages(msgs)
    assert out[0]["content"] == json.dumps({"key": "val"})


# ---------------------------------------------------------------------------
# _tool_to_openai
# ---------------------------------------------------------------------------


def test_tool_to_openai_from_dict():
    tool = {
        "name": "my_tool",
        "description": "does stuff",
        "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}},
    }
    out = _tool_to_openai(tool)
    assert out["type"] == "function"
    assert out["function"]["name"] == "my_tool"
    assert out["function"]["description"] == "does stuff"
    assert out["function"]["parameters"]["properties"]["x"] == {"type": "integer"}


def test_tool_to_openai_dict_missing_fields_uses_defaults():
    out = _tool_to_openai({})
    assert out["function"]["name"] == ""
    assert out["function"]["description"] == ""
    assert out["function"]["parameters"] == {"type": "object", "properties": {}}


# ---------------------------------------------------------------------------
# _get_response_output
# ---------------------------------------------------------------------------


def test_get_response_output_text_content():
    msg = SimpleNamespace(content="hello", tool_calls=None)
    resp = SimpleNamespace(message=msg)
    content, tool_calls = _get_response_output(resp)
    assert content == "hello"
    assert tool_calls is None


def test_get_response_output_empty_content_returns_none():
    msg = SimpleNamespace(content="", tool_calls=None)
    resp = SimpleNamespace(message=msg)
    content, _ = _get_response_output(resp)
    assert content is None


def test_get_response_output_tool_calls():
    tc = {"id": "c1", "name": "fn", "args": {}}
    msg = SimpleNamespace(content=None, tool_calls=[tc])
    resp = SimpleNamespace(message=msg)
    content, tool_calls = _get_response_output(resp)
    assert content is None
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "fn"

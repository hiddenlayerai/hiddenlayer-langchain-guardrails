from unittest.mock import Mock

import pytest


@pytest.fixture
def make_http_response():
    """Return a callable that builds a mock httpx.Response with a .json() method."""

    def _make(data: dict):
        resp = Mock()
        resp.json.return_value = data
        return resp

    return _make


@pytest.fixture
def dummy_request_classes():
    """
    Minimal stand-ins for LangChain message objects and ModelRequest:
      - Msg: .type, .content, .tool_calls, .tool_call_id, .model_copy(update={...})
      - Req: .messages, .system_message, .tools, .override(messages=[...])
    """

    class Msg:
        def __init__(self, content=None, type="human", tool_calls=None, tool_call_id=None):
            self.content = content
            self.type = type
            self.tool_calls = tool_calls
            self.tool_call_id = tool_call_id

        def model_copy(self, *, update=None):
            update = update or {}
            return Msg(
                content=update.get("content", self.content),
                type=update.get("type", self.type),
                tool_calls=update.get("tool_calls", self.tool_calls),
                tool_call_id=update.get("tool_call_id", self.tool_call_id),
            )

    class Req:
        def __init__(self, messages, system_message=None, tools=None):
            self.messages = messages
            self.system_message = system_message
            self.tools = tools

        def override(self, *, messages):
            return Req(messages, system_message=self.system_message, tools=self.tools)

    return Msg, Req


@pytest.fixture
def dummy_response_class():
    """
    Minimal stand-in for ModelResponse:
      - Resp(content): .message.content, .message.tool_calls
    """

    class MsgOut:
        def __init__(self, content=None, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls

    class Resp:
        def __init__(self, content=None, tool_calls=None):
            self.message = MsgOut(content, tool_calls)

    return Resp

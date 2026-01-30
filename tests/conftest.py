from types import SimpleNamespace

import pytest


@pytest.fixture
def make_hl_response():
    """
    Build a lightweight HL-like response object:
      resp.evaluation.action
      resp.modified_data.(input|output).messages[-1].content
    """

    def _make(*, action=None, role=None, redacted_text="REDACTED"):
        evaluation = SimpleNamespace(action=action)

        modified_data = None
        if action == "Redact" and role in ("user", "assistant"):
            container = SimpleNamespace(messages=[SimpleNamespace(content=redacted_text)])
            modified_data = SimpleNamespace(
                input=container if role == "user" else None,
                output=container if role == "assistant" else None,
            )

        return SimpleNamespace(evaluation=evaluation, modified_data=modified_data)

    return _make


@pytest.fixture
def dummy_request_classes():
    """
    Provide minimal stand-ins for ModelRequest and messages with the required API:
      - request.messages list of message objects with .content
      - request.override(messages=[...]) returns new request
    """

    class Msg:
        def __init__(self, content=None):
            self.content = content

    class Req:
        def __init__(self, messages):
            self.messages = messages

        def override(self, *, messages):
            return Req(messages)

    return Msg, Req


@pytest.fixture
def dummy_response_class():
    """
    Provide minimal stand-in for ModelResponse with required API:
      - response.message with .content
    """

    class Msg:
        def __init__(self, content=None):
            self.content = content

    class Resp:
        def __init__(self, content=None):
            self.message = Msg(content)

    return Resp


@pytest.fixture
def dummy_tool_request_class():
    """
    Provide minimal ToolCallRequest-like object with .tool_call dict.
    """

    class ToolReq:
        def __init__(self, tool_call):
            self.tool_call = tool_call

    return ToolReq

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Optional

import pytest

import hiddenlayer_langchain_guardrails.middleware as m

# -------------------------
# Test-only shapes (provided via fixtures ONLY)
# -------------------------


@dataclass
class Msg:
    content: str


@dataclass
class ModelReq:
    messages: list[Msg]

    def override(self, *, messages: list[Msg]) -> "ModelReq":
        return ModelReq(messages=messages)


@dataclass
class ModelResp:
    message: Msg


@dataclass
class ToolReq:
    tool_call: dict[str, Any]


# -------------------------
# Fixtures: HiddenLayer response builders
# -------------------------


@pytest.fixture()
def hl_response_builder() -> Callable[..., Any]:
    """Return a helper to build minimal HiddenLayer-like response objects."""

    def _build(
        *,
        action: Optional[str],
        role: str,
        redacted: Optional[str] = None,
    ) -> Any:
        evaluation = SimpleNamespace(action=action)

        modified_data = None
        if action == "Redact" and redacted is not None:
            msg_obj = SimpleNamespace(content=redacted)
            container = SimpleNamespace(messages=[msg_obj])
            modified_data = SimpleNamespace(
                input=container if role == "user" else None,
                output=container if role == "assistant" else None,
            )

        return SimpleNamespace(evaluation=evaluation, modified_data=modified_data)

    return _build


# -------------------------
# Fixtures: Params + guardrail
# -------------------------


@pytest.fixture()
def hl_params() -> m.HiddenLayerParams:
    return m.HiddenLayerParams(model="gpt-4o", project_id=None, requester_id="tester")


@pytest.fixture()
def guardrail(hl_params: m.HiddenLayerParams) -> m.HiddenLayerGuardrail:
    return m.HiddenLayerGuardrail(hl_params)


# -------------------------
# Fixtures: Requests
# -------------------------


@pytest.fixture()
def model_request() -> ModelReq:
    return ModelReq(messages=[Msg("original user message")])


@pytest.fixture()
def tool_request() -> ToolReq:
    return ToolReq(tool_call={"name": "echo", "args": {"text": "hello"}})


# -------------------------
# Fixtures: Handlers
# -------------------------


@pytest.fixture()
def model_handler_ok() -> Callable[[ModelReq], ModelResp]:
    def _handler(_req: ModelReq) -> ModelResp:
        return ModelResp(message=Msg("ok output"))

    return _handler


@pytest.fixture()
def model_handler_echo_input() -> Callable[[ModelReq], ModelResp]:
    """Echo the last user message (useful to validate input redaction)."""

    def _handler(req: ModelReq) -> ModelResp:
        return ModelResp(message=Msg(req.messages[-1].content))

    return _handler


@pytest.fixture()
def tool_handler_ok() -> Callable[[ToolReq], str]:
    def _handler(_req: ToolReq) -> str:
        return "tool ok"

    return _handler


# -------------------------
# Fixtures: Patch HiddenLayer analyze scenarios
#   Each fixture patches m._analyze_content and returns None
# -------------------------


@pytest.fixture()
def patch_hl_block_input(monkeypatch: pytest.MonkeyPatch, hl_response_builder) -> None:
    async def _fake_analyze_content(content: str, role: str, hiddenlayer_params: Any) -> Any:
        return hl_response_builder(action="Block", role="user")

    monkeypatch.setattr(m, "_analyze_content", _fake_analyze_content)


@pytest.fixture()
def patch_hl_redact_input(monkeypatch: pytest.MonkeyPatch, hl_response_builder) -> None:
    async def _fake_analyze_content(content: str, role: str, hiddenlayer_params: Any) -> Any:
        if role == "user":
            return hl_response_builder(action="Redact", role="user", redacted="REDACTED INPUT")
        return hl_response_builder(action=None, role="assistant")

    monkeypatch.setattr(m, "_analyze_content", _fake_analyze_content)


@pytest.fixture()
def patch_hl_block_output(monkeypatch: pytest.MonkeyPatch, hl_response_builder) -> None:
    async def _fake_analyze_content(content: str, role: str, hiddenlayer_params: Any) -> Any:
        if role == "user":
            return hl_response_builder(action=None, role="user")
        return hl_response_builder(action="Block", role="assistant")

    monkeypatch.setattr(m, "_analyze_content", _fake_analyze_content)


@pytest.fixture()
def patch_hl_redact_output(monkeypatch: pytest.MonkeyPatch, hl_response_builder) -> None:
    async def _fake_analyze_content(content: str, role: str, hiddenlayer_params: Any) -> Any:
        if role == "user":
            return hl_response_builder(action=None, role="user")
        return hl_response_builder(action="Redact", role="assistant", redacted="REDACTED OUTPUT")

    monkeypatch.setattr(m, "_analyze_content", _fake_analyze_content)


@pytest.fixture()
def patch_hl_block_tool_input(monkeypatch: pytest.MonkeyPatch, hl_response_builder) -> None:
    async def _fake_analyze_content(content: str, role: str, hiddenlayer_params: Any) -> Any:
        # tool input is treated as role="user" in your implementation
        return hl_response_builder(action="Block", role="user")

    monkeypatch.setattr(m, "_analyze_content", _fake_analyze_content)


@pytest.fixture()
def patch_hl_block_tool_output(monkeypatch: pytest.MonkeyPatch, hl_response_builder) -> None:
    async def _fake_analyze_content(content: str, role: str, hiddenlayer_params: Any) -> Any:
        if role == "user":
            return hl_response_builder(action=None, role="user")
        return hl_response_builder(action="Block", role="assistant")

    monkeypatch.setattr(m, "_analyze_content", _fake_analyze_content)


@pytest.fixture()
def patch_hl_redact_tool_output(monkeypatch: pytest.MonkeyPatch, hl_response_builder) -> None:
    async def _fake_analyze_content(content: str, role: str, hiddenlayer_params: Any) -> Any:
        if role == "user":
            return hl_response_builder(action=None, role="user")
        return hl_response_builder(action="Redact", role="assistant", redacted="REDACTED TOOL OUT")

    monkeypatch.setattr(m, "_analyze_content", _fake_analyze_content)


# -------------------------
# Tests (NO inline patching)
# -------------------------


def test_block_input_sync(
    patch_hl_block_input: None,
    guardrail: m.HiddenLayerGuardrail,
    model_request: ModelReq,
    model_handler_ok: Callable[[ModelReq], ModelResp],
) -> None:
    with pytest.raises(m.InputBlockedError):
        guardrail.wrap_model_call(model_request, model_handler_ok)


def test_redact_input_sync(
    patch_hl_redact_input: None,
    guardrail: m.HiddenLayerGuardrail,
    model_request: ModelReq,
    model_handler_echo_input: Callable[[ModelReq], ModelResp],
) -> None:
    resp = guardrail.wrap_model_call(model_request, model_handler_echo_input)
    assert resp.message.content == "REDACTED INPUT"


def test_block_output_sync(
    patch_hl_block_output: None,
    guardrail: m.HiddenLayerGuardrail,
    model_request: ModelReq,
    model_handler_ok: Callable[[ModelReq], ModelResp],
) -> None:
    with pytest.raises(m.OutputBlockedError):
        guardrail.wrap_model_call(model_request, model_handler_ok)


def test_redact_output_sync(
    patch_hl_redact_output: None,
    guardrail: m.HiddenLayerGuardrail,
    model_request: ModelReq,
    model_handler_ok: Callable[[ModelReq], ModelResp],
) -> None:
    resp = guardrail.wrap_model_call(model_request, model_handler_ok)
    assert resp.message.content == "REDACTED OUTPUT"


def test_block_tool_input_sync(
    patch_hl_block_tool_input: None,
    guardrail: m.HiddenLayerGuardrail,
    tool_request: ToolReq,
    tool_handler_ok: Callable[[ToolReq], str],
) -> None:
    with pytest.raises(m.InputBlockedError):
        guardrail.wrap_tool_call(tool_request, tool_handler_ok)


def test_block_tool_output_sync(
    patch_hl_block_tool_output: None,
    guardrail: m.HiddenLayerGuardrail,
    tool_request: ToolReq,
    tool_handler_ok: Callable[[ToolReq], str],
) -> None:
    with pytest.raises(m.OutputBlockedError):
        guardrail.wrap_tool_call(tool_request, tool_handler_ok)


def test_redact_tool_output_sync(
    patch_hl_redact_tool_output: None,
    guardrail: m.HiddenLayerGuardrail,
    tool_request: ToolReq,
    tool_handler_ok: Callable[[ToolReq], str],
) -> None:
    out = guardrail.wrap_tool_call(tool_request, tool_handler_ok)
    assert out == "REDACTED TOOL OUT"


@pytest.mark.asyncio
async def test_block_input_async(
    patch_hl_block_input: None,
    guardrail: m.HiddenLayerGuardrail,
    model_request: ModelReq,
) -> None:
    async def handler(_req: ModelReq) -> ModelResp:
        return ModelResp(message=Msg("should not run"))

    with pytest.raises(m.InputBlockedError):
        await guardrail.awrap_model_call(model_request, handler)

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Literal

from hiddenlayer import AsyncHiddenLayer, HiddenLayer
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.tools.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from pydantic import BaseModel

logger = logging.getLogger(__name__)

Role = Literal["user", "assistant"]


class HiddenLayerParams(BaseModel):
    """Configuration parameters used for HiddenLayer analysis calls.

    Attributes:
        model: The model name associated with the request (for metadata).
        project_id: Optional HiddenLayer project ID used to apply a specific policy.
        requester_id: Identifier for the end-user or calling service (for metadata/audit).
    """

    model: str | None
    project_id: str | None
    requester_id: str | None


class HiddenLayerActions(str, Enum):
    """HiddenLayer evaluation actions supported by this guardrail.

    Rule set / policy of the project determines the response to detections
    e.g., 'Alert', 'Allow', 'Block' and 'Redact' (and redaction type).
    Allow and Alert actions will not interrupt execution.
    """

    BLOCK = "Block"
    REDACT = "Redact"


class InputBlockedError(Exception):
    """Raised when HiddenLayer blocks the input."""


class OutputBlockedError(Exception):
    """Raised when HiddenLayer blocks the output."""


@dataclass
class AnalysisResult:
    """Normalized result derived from a HiddenLayer analysis response."""

    block: bool
    redact: bool
    redacted_content: str | None


def _build_analyze_kwargs(content: str, role: Role, params: HiddenLayerParams) -> dict[str, Any]:
    metadata = {"model": params.model, "requester_id": params.requester_id}
    message = {"messages": [{"role": role, "content": content}]}

    kwargs: dict[str, Any] = {"metadata": metadata}
    if params.project_id:
        kwargs["hl_project_id"] = params.project_id

    if role == "user":
        kwargs["input"] = message
    else:
        kwargs["output"] = message

    return kwargs


def _extract_result(response: Any, role: Role) -> AnalysisResult:
    action = getattr(getattr(response, "evaluation", None), "action", None)
    logger.debug("HiddenLayer evaluation.action=%r response=%r", action, response)

    block = action == HiddenLayerActions.BLOCK
    redact = action == HiddenLayerActions.REDACT

    redacted_content: str | None = None
    modified = getattr(response, "modified_data", None)
    if redact and modified:
        container = modified.input if role == "user" else modified.output
        msgs = getattr(container, "messages", None)
        if msgs:
            redacted_content = msgs[-1].content

    return AnalysisResult(block=block, redact=redact, redacted_content=redacted_content)


def _get_request_last_content(request: ModelRequest) -> str | None:
    if not request.messages:
        return None
    content = getattr(request.messages[-1], "content", None)
    return content if isinstance(content, str) and content else None


def _replace_last_message(request: ModelRequest, text: str) -> ModelRequest:
    last = request.messages[-1]
    new_last = last.__class__(content=text)
    return request.override(messages=[*request.messages[:-1], new_last])


def _get_response_content(response: ModelResponse) -> str | None:
    msg = getattr(response, "message", None)
    content = getattr(msg, "content", None)
    return content if isinstance(content, str) and content else None


def _set_response_content(response: ModelResponse, text: str) -> None:
    msg = getattr(response, "message", None)
    if msg is not None:
        msg.content = text


def _apply_tool_input_redaction(request: ToolCallRequest, redacted_payload: str) -> ToolCallRequest:
    """
    Apply redacted tool input payload to ToolCallRequest. If redacted_content,
    overwrite request.tool_call["args"] with redacted content.
    """
    try:
        parsed = json.loads(redacted_payload)
    except Exception:
        logger.warning("Failed to parse redacted tool payload as JSON: %r", redacted_payload)
        return request

    if not isinstance(parsed, dict) or "args" not in parsed or not isinstance(parsed["args"], dict):
        logger.warning("Redacted tool payload missing expected 'args' dict: %r", parsed)
        return request

    tool_call = getattr(request, "tool_call", None)
    if not isinstance(tool_call, dict):
        return request

    tool_call["args"] = parsed["args"]
    return request


class HiddenLayerGuardrailBase(AgentMiddleware):
    """Base class for sync and async HiddenLayer guardrails."""

    def __init__(self, params: HiddenLayerParams | None = None):
        super().__init__()
        self.params = params

    def _on_input_result(self, request: ModelRequest, result: AnalysisResult) -> ModelRequest:
        if result.block:
            raise InputBlockedError("Input blocked by HiddenLayer")
        if result.redact and result.redacted_content:
            return _replace_last_message(request, result.redacted_content)
        return request

    def _on_output_result(self, response: ModelResponse, result: AnalysisResult) -> ModelResponse:
        if result.block:
            raise OutputBlockedError("Output blocked by HiddenLayer")
        if result.redact and result.redacted_content:
            _set_response_content(response, result.redacted_content)
        return response


class AsyncHiddenLayerGuardrail(HiddenLayerGuardrailBase):
    """
    Supports asynchronous agent execution with `ainvoke` and `astream`.
    Uses AsyncHiddenLayer client and async middleware hooks.

    The middleware analyzes:
      - model input (user messages) and output (assistant responses)
      - tool input (tool args JSON) and tool output (stringified result)
    """

    def __init__(self, params: HiddenLayerParams | None = None, client: AsyncHiddenLayer | None = None):
        super().__init__(params=params)
        self.client = client or AsyncHiddenLayer()

    async def analyze(self, *, content: str, role: Role) -> AnalysisResult:
        kwargs = _build_analyze_kwargs(content, role, self.params)
        resp = await self.client.interactions.analyze(**kwargs)
        return _extract_result(resp, role)

    @AgentMiddleware.abefore_model
    async def check_model_input(self, state: AgentState, runtime: Runtime, request: ModelRequest) -> ModelRequest:
        content = _get_request_last_content(request)
        if not content:
            return request
        result = await self.analyze(content=content, role="user")
        return self._on_input_result(request, result)

    @AgentMiddleware.aafter_model
    async def check_model_output(
        self, state: AgentState, runtime: Runtime, request: ModelRequest, response: ModelResponse
    ) -> ModelResponse:
        content = _get_response_content(response)
        if not content:
            return response
        result = await self.analyze(content=content, role="assistant")
        return self._on_output_result(response, result)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        tool_call = getattr(request, "tool_call", {}) or {}
        tool_name = tool_call.get("name", "<unknown>")
        tool_args = tool_call.get("args", {}) or {}

        tool_payload = json.dumps({"args": tool_args}, ensure_ascii=False)
        in_res = await self.analyze(content=tool_payload, role="user")
        if in_res.block:
            raise InputBlockedError(f"Tool input for {tool_name} blocked by HiddenLayer")

        if in_res.redact and in_res.redacted_content:
            request = _apply_tool_input_redaction(request, in_res.redacted_content)

        output = await handler(request)

        out_res = await self.analyze(content=str(output), role="assistant")
        if out_res.block:
            raise OutputBlockedError(f"Tool output from {tool_name} blocked by HiddenLayer")

        return out_res.redacted_content if (out_res.redact and out_res.redacted_content) else output


class HiddenLayerGuardrail(HiddenLayerGuardrailBase):
    """Supports synchronous agent execution with `invoke` and `stream`.
       Use AsyncHiddenLayerGuardrail to take advantage of LangChain's async support.

    This middleware analyzes:
      - model input (user messages) and output (assistant responses)
      - tool input (tool args JSON) and tool output (stringified result)
    """

    def __init__(self, params: HiddenLayerParams | None = None, client: HiddenLayer | None = None):
        super().__init__(params=params)
        self.client = client or HiddenLayer()

    def analyze(self, *, content: str, role: Role) -> AnalysisResult:
        kwargs = _build_analyze_kwargs(content, role, self.params)
        resp = self.client.interactions.analyze(**kwargs)
        return _extract_result(resp, role)

    @AgentMiddleware.before_model
    def check_model_input(self, state: AgentState, runtime: Runtime, request: ModelRequest) -> ModelRequest:
        content = _get_request_last_content(request)
        if not content:
            return request
        result = self.analyze(content=content, role="user")
        return self._on_input_result(request, result)

    @AgentMiddleware.after_model
    def check_model_output(
        self, state: AgentState, runtime: Runtime, request: ModelRequest, response: ModelResponse
    ) -> ModelResponse:
        content = _get_response_content(response)
        if not content:
            return response
        result = self.analyze(content=content, role="assistant")
        return self._on_output_result(response, result)

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Any]) -> Any:
        tool_call = getattr(request, "tool_call", {}) or {}
        tool_name = tool_call.get("name", "<unknown>")
        tool_args = tool_call.get("args", {}) or {}

        tool_payload = json.dumps({"args": tool_args}, ensure_ascii=False)
        in_res = self.analyze(content=tool_payload, role="user")
        if in_res.block:
            raise InputBlockedError(f"Tool input for {tool_name} blocked by HiddenLayer")

        if in_res.redact and in_res.redacted_content:
            request = _apply_tool_input_redaction(request, in_res.redacted_content)

        output = handler(request)

        out_res = self.analyze(content=str(output), role="assistant")
        if out_res.block:
            raise OutputBlockedError(f"Tool output from {tool_name} blocked by HiddenLayer")

        return out_res.redacted_content if (out_res.redact and out_res.redacted_content) else output

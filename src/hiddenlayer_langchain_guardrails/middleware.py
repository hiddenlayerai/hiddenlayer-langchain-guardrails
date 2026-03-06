from __future__ import annotations
from langchain_core.tools.base import BaseTool

import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Awaitable, Callable, Iterator, Literal, TypeVar

from hiddenlayer import AsyncHiddenLayer, HiddenLayer
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools.tool_node import ToolCallRequest
from pydantic import BaseModel

logger = logging.getLogger(__name__)

Role = Literal["user", "assistant"]

T = TypeVar("T")


class HiddenLayerParams(BaseModel):
    """HiddenLayer request metadata and policy routing parameters."""

    model: str | None = None
    project_id: str | None = os.getenv("HIDDENLAYER_PROJECT_ID")
    requester_id: str = os.getenv("HIDDENLAYER_REQUESTER_ID", "hiddenlayer-langchain-integration")


class HiddenLayerActions(str, Enum):
    """HiddenLayer evaluation actions supported by this middleware."""

    BLOCK = "Block"
    REDACT = "Redact"


class InputBlockedError(Exception):
    """Raised when HiddenLayer blocks the input."""


class OutputBlockedError(Exception):
    """Raised when HiddenLayer blocks the output."""


@dataclass
class AnalysisResult:
    """Normalized evaluation outcome for a single analysis call."""

    block: bool
    redact: bool
    redacted_content: str | None


def _build_analyze_kwargs(content: str, role: Role, params: HiddenLayerParams) -> dict[str, Any]:
    """Build HiddenLayer `interactions.analyze` kwargs for a single message and role."""
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
    """Extract block/redact decisions (and redacted content if present) from a HiddenLayer response."""
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
    """Return last request message content if it is a non-empty string."""
    if not request.messages:
        return None
    content = getattr(request.messages[-1], "content", None)
    return content if isinstance(content, str) and content else None


def _replace_last_message(request: ModelRequest, text: str) -> ModelRequest:
    """Return a copy of the request with the last message content replaced."""
    last = request.messages[-1]
    new_last = last.__class__(content=text)
    return request.override(messages=[*request.messages[:-1], new_last])


def _get_response_content(response: ModelResponse) -> str | None:
    """Return response message content if it is a non-empty string."""
    msg = getattr(response, "message", None) or getattr(response, "result")

    if isinstance(msg, list):
        msg = msg[-1]

    content = getattr(msg, "content", None)

    # If a model responds saying to run a tool, the content gets parsed into a tool calls field.
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        content = json.dumps(tool_calls)

    return content if isinstance(content, str) and content else None


def _set_response_content(response: ModelResponse, text: str) -> None:
    """Set response message content in place when present."""
    msg = getattr(response, "message", None)
    if msg is not None:
        msg.content = text


def _apply_tool_input_redaction(request: ToolCallRequest, redacted_payload: str) -> ToolCallRequest:
    """
    Apply tool-argument redactions from a JSON payload.

    The payload must be a JSON object with an `args` dict (e.g., {"args": {...}}).
    If parsing fails or the shape is unexpected, the request is returned unchanged.
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


def _extract_text_from_event(event: Any) -> str:
    """Extract text content from a LangGraph stream event.

    Supports two LangGraph streaming modes:
    - ``stream_mode="messages"``: ``(AIMessageChunk, metadata)`` tuple
    - ``stream_mode="updates"``: ``{"node_name": {"messages": [AIMessage(...)]}}`` dict

    Returns an empty string when no text can be extracted.
    """
    # stream_mode="messages": (AIMessageChunk, metadata)
    if isinstance(event, tuple) and event:
        chunk = event[0]
    # stream_mode="updates": {"node_name": {"messages": [AIMessage(...)]}}
    elif isinstance(event, dict):
        chunk = None
        for value in event.values():
            if isinstance(value, dict):
                messages = value.get("messages")
                if messages and isinstance(messages, list):
                    chunk = messages[-1]
                    break
    else:
        return ""

    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block
            if isinstance(block, str)
            else block.get("text", "")
            if isinstance(block, dict) and block.get("type") == "text"
            else ""
            for block in content
        )

    return ""


class HiddenLayerGuardrailBase(AgentMiddleware):
    """Shared logic for sync/async guardrails (params normalization and block/redact handling)."""

    def __init__(self, params: HiddenLayerParams | None = None):
        super().__init__()
        # Always keep a concrete params instance so helper functions never see None.
        self.params: HiddenLayerParams = params or HiddenLayerParams()

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
    """Async guardrail that wraps model/tool calls and enforces HiddenLayer decisions."""

    def __init__(self, params: HiddenLayerParams | None = None, client: AsyncHiddenLayer | None = None):
        super().__init__(params=params)
        self.client = client or AsyncHiddenLayer()

    async def analyze(self, *, content: str, role: Role) -> AnalysisResult:
        kwargs = _build_analyze_kwargs(content, role, self.params)
        resp = await self.client.interactions.analyze(**kwargs)
        return _extract_result(resp, role)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:

        content_pieces = []
        if request.tools:
            tools = [{"name": tool.name, "description": tool.description} for tool in request.tools]
            content_pieces.append(json.dumps(tools))

        last_content = _get_request_last_content(request)
        if last_content:
            content_pieces.append(last_content)

        if content_pieces:
            in_res = await self.analyze(content="\n".join(content_pieces), role="user")
            request = self._on_input_result(request, in_res)

        response = await handler(request)

        out_content = _get_response_content(response)
        if out_content:
            out_res = await self.analyze(content=out_content, role="assistant")
            response = self._on_output_result(response, out_res)

        return response

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        tool_call = getattr(request, "tool_call", {}) or {}
        tool_name = tool_call.get("name", "<unknown>")
        tool_args = tool_call.get("args", {}) or {}
        tool_description = tool_call.get("description", "")
        tool_payload = {"name": tool_name, "description": tool_description, "args": tool_args}
        in_res = await self.analyze(content=json.dumps(tool_payload), role="user")
        if in_res.block:
            raise InputBlockedError(f"Tool input for {tool_name} blocked by HiddenLayer")

        if in_res.redact and in_res.redacted_content:
            request = _apply_tool_input_redaction(request, in_res.redacted_content)

        output = await handler(request)

        out_res = await self.analyze(content=str(output), role="user")
        if out_res.block:
            raise OutputBlockedError(f"Tool output from {tool_name} blocked by HiddenLayer")

        return out_res.redacted_content if (out_res.redact and out_res.redacted_content) else output

    async def safe_stream(self, stream: AsyncIterator[T]) -> AsyncIterator[T]:
        """Wrap an asynchronous output stream, forwarding every event and scanning once complete.

        Each event from *stream* is yielded immediately.  Text content is
        extracted from every event and accumulated.  After the stream is
        exhausted the full text is submitted to HiddenLayer for output
        scanning.  This operates in **alert-only** mode: detected issues are
        logged but the stream is never blocked or modified.
        """
        buffer: list[str] = []
        async for event in stream:
            text = _extract_text_from_event(event)
            if text:
                buffer.append(text)
            yield event

        full_text = "".join(buffer)
        if full_text:
            try:
                await self.analyze(content=full_text, role="assistant")
            except Exception:
                logger.exception("HiddenLayer output stream scan failed")


class HiddenLayerGuardrail(HiddenLayerGuardrailBase):
    """Sync guardrail that wraps model/tool calls and enforces HiddenLayer decisions."""

    def __init__(self, params: HiddenLayerParams | None = None, client: HiddenLayer | None = None):
        super().__init__(params=params)
        self.client = client or HiddenLayer()

    def analyze(self, *, content: str, role: Role) -> AnalysisResult:
        kwargs = _build_analyze_kwargs(content, role, self.params)
        resp = self.client.interactions.analyze(**kwargs)
        return _extract_result(resp, role)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:

        content_pieces = []
        if request.tools:
            tools = [{"name": tool.name, "description": tool.description} for tool in request.tools]
            content_pieces.append(json.dumps(tools))

        last_content = _get_request_last_content(request)
        if last_content:
            content_pieces.append(last_content)

        if content_pieces:
            in_res = self.analyze(content="\n".join(content_pieces), role="user")
            request = self._on_input_result(request, in_res)

        response = handler(request)

        out_content = _get_response_content(response)
        if out_content:
            out_res = self.analyze(content=out_content, role="assistant")
            response = self._on_output_result(response, out_res)

        return response

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        tool_call = getattr(request, "tool_call", {}) or {}
        tool_name = tool_call.get("name", "<unknown>")
        tool_args = tool_call.get("args", {}) or {}
        tool_description = tool_call.get("description", "")

        tool_payload = {"name": tool_name, "description": tool_description, "args": tool_args}
        in_res = self.analyze(content=json.dumps(tool_payload), role="user")
        if in_res.block:
            raise InputBlockedError(f"Tool input for {tool_name} blocked by HiddenLayer")

        if in_res.redact and in_res.redacted_content:
            request = _apply_tool_input_redaction(request, in_res.redacted_content)

        output = handler(request)

        out_res = self.analyze(content=str(output), role="user")
        if out_res.block:
            raise OutputBlockedError(f"Tool output from {tool_name} blocked by HiddenLayer")

        return out_res.redacted_content if (out_res.redact and out_res.redacted_content) else output

    def safe_stream(self, stream: Iterator[T]) -> Iterator[T]:
        """Wrap a synchronous output stream, forwarding every event and scanning once complete.

        Each event from *stream* is yielded immediately.  Text content is
        extracted from every event and accumulated.  After the stream is
        exhausted the full text is submitted to HiddenLayer for output
        scanning.  This operates in **alert-only** mode: detected issues are
        logged but the stream is never blocked or modified.
        """
        buffer: list[str] = []
        for event in stream:
            text = _extract_text_from_event(event)
            if text:
                buffer.append(text)
            yield event

        full_text = "".join(buffer)
        if full_text:
            try:
                self.analyze(content=full_text, role="assistant")
            except Exception:
                logger.exception("HiddenLayer output stream scan failed")

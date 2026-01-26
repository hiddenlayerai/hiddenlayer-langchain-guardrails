from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Coroutine, Literal, TypeVar

from hiddenlayer import AsyncHiddenLayer
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools.tool_node import ToolCallRequest
from pydantic import BaseModel

client = AsyncHiddenLayer()
logger = logging.getLogger(__name__)

T = TypeVar("T")


class HiddenLayerParams(BaseModel):
    """Configuration parameters used for HiddenLayer analysis calls.

    Attributes:
        model: The model name associated with the request (for metadata).
        project_id: Optional HiddenLayer project ID used to apply a specific policy.
        requester_id: Identifier for the end-user or calling service (for metadata/audit).
    """

    model: str
    project_id: str | None
    requester_id: str


class HiddenLayerActions(str, Enum):
    """HiddenLayer evaluation actions supported by this guardrail.
       Rule set / policy of the project determines the response to detections
       i.e., 'Alert' vs. 'Block' and redaction type.

    Attributes:
        BLOCK: The request/response should be blocked.
        REDACT: The request/response should be redacted (rewritten).
    """

    BLOCK = "Block"
    REDACT = "Redact"


class InputBlockedError(Exception):
    """Raised when HiddenLayer blocks the input."""


class OutputBlockedError(Exception):
    """Raised when HiddenLayer blocks the output."""


@dataclass
class AnalysisResult:
    """Normalized result derived from a HiddenLayer analysis response.

    Attributes:
        block: True if HiddenLayer returned action == "Block".
        redact: True if HiddenLayer returned action == "Redact".
        redacted_content: Redacted text returned by HiddenLayer (if available).
    """

    block: bool
    redact: bool
    redacted_content: str | None


async def _analyze_content(
    content: str,
    role: Literal["user", "assistant"],
    hiddenlayer_params: HiddenLayerParams,
) -> Any:
    """Call HiddenLayer Interactions Analyze API for a single message.

    Args:
        content: Message content to analyze.
        role: Message role. Use "user" to analyze input; "assistant" to analyze output.
        hiddenlayer_params: HiddenLayer configuration parameters.

    Returns:
        Raw HiddenLayer SDK response object.
    """
    metadata = {"model": hiddenlayer_params.model, "requester_id": hiddenlayer_params.requester_id}
    message = {"messages": [{"role": role, "content": content}]}

    kwargs: dict[str, Any] = {"metadata": metadata}
    if hiddenlayer_params.project_id:
        kwargs["hl_project_id"] = hiddenlayer_params.project_id

    if role == "user":
        kwargs["input"] = message
    else:
        kwargs["output"] = message

    return await client.interactions.analyze(**kwargs)


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine from synchronous code.

    This is used to allow `agent.invoke()` / `stream()` (sync code paths) to
    call the async HiddenLayer client.

    Notes:
        - If no running event loop exists, uses `asyncio.run`.
        - If a loop is running (e.g., notebooks), schedules the coroutine using
          `asyncio.run_coroutine_threadsafe`.

    Args:
        coro: The coroutine to execute.

    Returns:
        The coroutine result.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        return asyncio.run_coroutine_threadsafe(coro, loop).result()
    return asyncio.run(coro)


async def _analyze_and_extract(
    *,
    content: str,
    role: Literal["user", "assistant"],
    hiddenlayer_params: HiddenLayerParams,
) -> AnalysisResult:
    """Analyze content and normalize HiddenLayer block/redact decisions.

    Args:
        content: Text to analyze.
        role: "user" for input analysis or "assistant" for output analysis.
        hiddenlayer_params: HiddenLayer configuration parameters.

    Returns:
        A normalized `AnalysisResult` describing block/redact decisions and any
        redacted content.
    """
    response = await _analyze_content(content, role, hiddenlayer_params)

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


def _default_params_from_env() -> HiddenLayerParams:
    """Load HiddenLayer params from environment variables.

    Environment variables:
        HL_MODEL: Model name to attach to metadata (default: "gpt-4o").
        HL_REQUESTER_ID: Requester identifier (default: "unknown").
        HL_PROJECT_ID: Optional project ID to apply a specific policy.

    Returns:
        A `HiddenLayerParams` instance.
    """
    return HiddenLayerParams(
        model=os.getenv("HL_MODEL", "gpt-4o"),
        requester_id=os.getenv("HL_REQUESTER_ID", "unknown"),
        project_id=os.getenv("HL_PROJECT_ID") or None,
    )


class HiddenLayerGuardrail(AgentMiddleware):
    """Custom LangChain agent middleware that implements HiddenLayer runtime security as a guardrail.

    This middleware intercepts:
      - model input (user messages) and output (assistant responses)
      - tool input (tool args JSON) and tool output (stringified result)

    It uses the async HiddenLayer client under the hood but supports both:
      - sync agent execution (`invoke`, `stream`)
      - async agent execution (`ainvoke`, `astream`)

    Args:
        params: Optional `HiddenLayerParams`. If not provided, values are loaded
            from environment using `_default_params_from_env()`.
    """

    def __init__(self, params: HiddenLayerParams | None = None):
        super().__init__()
        self.params = params or _default_params_from_env()

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        """Synchronous wrapper for model calls.

        Args:
            request: LangChain model request.
            handler: Callable that performs the actual model call.

        Returns:
            ModelResponse after optional redaction.

        Raises:
            InputBlockedError: If HiddenLayer blocks input.
            OutputBlockedError: If HiddenLayer blocks output.
        """
        if request.messages:
            last = request.messages[-1]
            content = getattr(last, "content", None)

            if isinstance(content, str) and content:
                res_in = _run_async(_analyze_and_extract(content=content, role="user", hiddenlayer_params=self.params))
                if res_in.block:
                    raise InputBlockedError("Input blocked by HiddenLayer")
                if res_in.redact and res_in.redacted_content:
                    new_last = last.__class__(content=res_in.redacted_content)
                    request = request.override(messages=[*request.messages[:-1], new_last])

        response = handler(request)

        msg = getattr(response, "message", None)
        out = getattr(msg, "content", None)

        if isinstance(out, str) and out:
            res_out = _run_async(_analyze_and_extract(content=out, role="assistant", hiddenlayer_params=self.params))
            if res_out.block:
                raise OutputBlockedError("Output blocked by HiddenLayer")
            if res_out.redact and res_out.redacted_content and msg is not None:
                msg.content = res_out.redacted_content

        return response

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Any]) -> Any:
        """Synchronous wrapper for tool calls.

        Args:
            request: Tool call request (contains tool name + args).
            handler: Callable that executes the tool.

        Returns:
            Tool output, optionally redacted if HiddenLayer requests redaction.

        Raises:
            InputBlockedError: If HiddenLayer blocks tool input.
            OutputBlockedError: If HiddenLayer blocks tool output.
        """
        tool_call = getattr(request, "tool_call", {}) or {}
        tool_name = tool_call.get("name", "<unknown>")
        tool_args = tool_call.get("args", {}) or {}

        tool_payload = json.dumps({"args": tool_args}, ensure_ascii=False)
        res_in = _run_async(_analyze_and_extract(content=tool_payload, role="user", hiddenlayer_params=self.params))
        if res_in.block:
            raise InputBlockedError(f"Tool input for {tool_name} blocked by HiddenLayer")

        output = handler(request)

        res_out = _run_async(
            _analyze_and_extract(content=str(output), role="assistant", hiddenlayer_params=self.params)
        )
        if res_out.block:
            raise OutputBlockedError(f"Tool output from {tool_name} blocked by HiddenLayer")

        return res_out.redacted_content if (res_out.redact and res_out.redacted_content) else output

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async wrapper for model calls.

        Raises:
            InputBlockedError: If HiddenLayer blocks input.
            OutputBlockedError: If HiddenLayer blocks output.
        """
        if request.messages:
            last = request.messages[-1]
            content = getattr(last, "content", None)

            if isinstance(content, str) and content:
                res_in = await _analyze_and_extract(content=content, role="user", hiddenlayer_params=self.params)
                if res_in.block:
                    raise InputBlockedError("Input blocked by HiddenLayer")
                if res_in.redact and res_in.redacted_content:
                    new_last = last.__class__(content=res_in.redacted_content)
                    request = request.override(messages=[*request.messages[:-1], new_last])

        response = await handler(request)

        msg = getattr(response, "message", None)
        out = getattr(msg, "content", None)
        if isinstance(out, str) and out:
            res_out = await _analyze_and_extract(content=out, role="assistant", hiddenlayer_params=self.params)
            if res_out.block:
                raise OutputBlockedError("Output blocked by HiddenLayer")
            if res_out.redact and res_out.redacted_content and msg is not None:
                msg.content = res_out.redacted_content

        return response

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        """Async wrapper for tool calls."""
        tool_call = getattr(request, "tool_call", {}) or {}
        tool_name = tool_call.get("name", "<unknown>")
        tool_args = tool_call.get("args", {}) or {}

        tool_payload = json.dumps({"args": tool_args}, ensure_ascii=False)
        res_in = await _analyze_and_extract(content=tool_payload, role="user", hiddenlayer_params=self.params)
        if res_in.block:
            raise InputBlockedError(f"Tool input for {tool_name} blocked by HiddenLayer")

        output = await handler(request)

        res_out = await _analyze_and_extract(content=str(output), role="assistant", hiddenlayer_params=self.params)
        if res_out.block:
            raise OutputBlockedError(f"Tool output from {tool_name} blocked by HiddenLayer")

        return res_out.redacted_content if (res_out.redact and res_out.redacted_content) else output

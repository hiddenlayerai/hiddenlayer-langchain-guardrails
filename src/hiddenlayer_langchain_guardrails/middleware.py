from __future__ import annotations

import httpx
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Awaitable, Callable, Iterator, TypeVar

from hiddenlayer import AsyncHiddenLayer, HiddenLayer
from hiddenlayer._base_client import make_request_options
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.tools.base import BaseTool
from pydantic import BaseModel
from uuid import uuid4

logger = logging.getLogger(__name__)

T = TypeVar("T")

_LANGCHAIN_ROLE_MAP = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "function",
}

REQUEST_EVALUATIONS_PATH = "/detection/v2/request-evaluations"
RESPONSE_EVALUATIONS_PATH = "/detection/v2/response-evaluations"


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


def _to_openai_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Convert LangChain messages to OpenAI chat format."""
    result = []
    for msg in messages:
        role = _LANGCHAIN_ROLE_MAP.get(getattr(msg, "type", "human"), "user")

        if role == "tool":
            content = getattr(msg, "content", "") or ""
            if not isinstance(content, str):
                content = json.dumps(content)
            entry: dict[str, Any] = {"role": "tool", "content": content}
            tool_call_id = getattr(msg, "tool_call_id", None)
            if tool_call_id:
                entry["tool_call_id"] = tool_call_id
            result.append(entry)
            continue

        if role == "assistant":
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                result.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tc.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": tc.get("name", ""),
                                    "arguments": json.dumps(tc.get("args", {})),
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                )
                continue

        content = getattr(msg, "content", "") or ""
        if not isinstance(content, str):
            content = json.dumps(content)
        result.append({"role": role, "content": content})
    return result


def _tool_to_openai(tool: BaseTool | dict[str, Any]) -> dict[str, Any]:
    """Convert a LangChain tool (BaseTool or dict) to OpenAI function tool format."""
    if isinstance(tool, dict):
        name = tool.get("name", "")
        description = tool.get("description", "")
        parameters = tool.get("parameters", {"type": "object", "properties": {}})
    else:
        name = tool.name
        description = tool.description
        args_schema = getattr(tool, "args_schema", None)
        if args_schema is not None:
            try:
                parameters = args_schema.model_json_schema()
            except AttributeError:
                parameters = args_schema.schema()
        else:
            parameters = {"type": "object", "properties": {}}
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


def _build_request_eval_body(messages: list[dict[str, Any]], params: HiddenLayerParams) -> dict[str, Any]:
    """Build body for the request-evaluations endpoint (OpenAI chat request format)."""
    body: dict[str, Any] = {"messages": messages}
    if params.model:
        body["model"] = params.model
    return body


def _build_response_eval_body(
    content: str | None,
    params: HiddenLayerParams,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build body for the response-evaluations endpoint (OpenAI chat response format)."""
    if tool_calls:
        message: dict[str, Any] = {"role": "assistant", "content": None, "tool_calls": tool_calls}
        finish_reason = "tool_calls"
    else:
        message = {"role": "assistant", "content": content}
        finish_reason = "stop"
    body: dict[str, Any] = {"choices": [{"index": 0, "message": message, "finish_reason": finish_reason}]}
    if params.model:
        body["model"] = params.model
    return body


def _extra_options(params: HiddenLayerParams, roundtrip_id: str) -> dict[str, Any]:
    """Build make_request_options kwargs from params."""
    opts: dict[str, Any] = {"extra_headers": {"HL-RoundTrip-Id": roundtrip_id}}
    opts["extra_headers"]["hl-requester-id"] = params.requester_id

    if params.project_id:
        opts["extra_headers"]["HL-Project-Id"] = params.project_id
    return opts


def _extract_input_result(
    data: dict[str, Any], original_messages: list[dict[str, Any]] | None = None
) -> AnalysisResult:
    """Extract block/redact decisions from a request-evaluations response (OpenAI pass-through format).

    The API returns the provider payload directly:
    - Block: an OpenAI response payload (has ``choices``) instead of a request payload
    - Allow/Redact: the (potentially modified) OpenAI request payload (has ``messages``)
    """
    # Block: API returned an OpenAI response payload (canned block message)
    if "choices" in data:
        logger.debug("HiddenLayer request-evaluations: blocked")
        return AnalysisResult(block=True, redact=False, redacted_content=None)

    # Allow or Redact: API returned the (potentially modified) OpenAI request payload
    messages = data.get("messages") or []
    redacted_content: str | None = None
    redact = False

    if messages and original_messages:
        original_last = original_messages[-1].get("content", "")
        returned_last = messages[-1].get("content", "")
        if original_last != returned_last:
            redact = True
            redacted_content = returned_last
            logger.debug("HiddenLayer request-evaluations: redacted")

    return AnalysisResult(block=False, redact=redact, redacted_content=redacted_content)


def _extract_output_result(data: dict[str, Any], original_content: str | None = None) -> AnalysisResult:
    """Extract block/redact decisions from a response-evaluations response (OpenAI pass-through format).

    The API always returns an OpenAI response payload. Content is compared against the
    original to detect redaction.
    """
    choices = data.get("choices") or []
    if not choices:
        return AnalysisResult(block=False, redact=False, redacted_content=None)

    msg = choices[0].get("message") or {}
    returned_content = msg.get("content")

    redacted_content: str | None = None
    redact = False

    if returned_content is not None and original_content is not None and returned_content != original_content:
        redact = True
        redacted_content = returned_content
        logger.debug("HiddenLayer response-evaluations: redacted")

    return AnalysisResult(block=False, redact=redact, redacted_content=redacted_content)


def _get_response_msg(response: ModelResponse) -> Any:
    """Return the message object from a ModelResponse."""
    msg = getattr(response, "message", None) or getattr(response, "result")
    if isinstance(msg, list):
        msg = msg[-1]
    return msg


def _get_response_content(response: ModelResponse) -> str | None:
    """Return response message text content if it is a non-empty string."""
    content = getattr(_get_response_msg(response), "content", None)
    return content if isinstance(content, str) and content else None


def _get_response_tool_calls(response: ModelResponse) -> list[dict[str, Any]] | None:
    """Return tool calls from the response in OpenAI format, or None."""
    tool_calls = getattr(_get_response_msg(response), "tool_calls", None)
    if not tool_calls:
        return None
    return [
        {
            "id": tc.get("id", ""),
            "type": "function",
            "function": {
                "name": tc.get("name", ""),
                "arguments": json.dumps(tc.get("args", {})),
            },
        }
        for tc in tool_calls
    ]


def _set_response_content(response: ModelResponse, text: str) -> None:
    """Set response message content in place when present."""
    msg = getattr(response, "message", None)
    if msg is not None:
        msg.content = text


def _replace_last_message(request: ModelRequest, text: str) -> ModelRequest:
    """Return a copy of the request with the last message content replaced."""
    last = request.messages[-1]
    new_last = last.model_copy(update={"content": text})
    return request.override(messages=[*request.messages[:-1], new_last])


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

    # async def analyze_input(self, messages: list[dict[str, Any]]) -> AnalysisResult:
    #     body = _build_request_eval_body(messages, self.params)
    #     opts = make_request_options(**_extra_options(self.params))
    #     resp = await self.client.post(
    #         REQUEST_EVALUATIONS_PATH,
    #         cast_to=httpx.Response,
    #         body=body,
    #         options=opts,
    #     )
    #     return _extract_input_result(resp.json(), original_messages=messages)

    # async def analyze_output(self, content: str) -> AnalysisResult:
    #     body = _build_response_eval_body(content, self.params)
    #     opts = make_request_options(**_extra_options(self.params))
    #     resp = await self.client.post(
    #         RESPONSE_EVALUATIONS_PATH,
    #         cast_to=httpx.Response,
    #         body=body,
    #         options=opts,
    #     )
    #     return _extract_output_result(resp.json(), original_content=content)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        roundtrip_id = str(uuid4())
        messages = _to_openai_messages(request.messages)

        if messages:
            body = _build_request_eval_body(messages, self.params)
            if request.tools:
                body["tools"] = [_tool_to_openai(t) for t in request.tools]
            opts = make_request_options(**_extra_options(self.params, roundtrip_id))
            resp = await self.client.post(
                REQUEST_EVALUATIONS_PATH,
                cast_to=httpx.Response,
                body=body,
                options=opts,
            )
            in_res = _extract_input_result(resp.json(), original_messages=messages)
            request = self._on_input_result(request, in_res)

        response = await handler(request)

        out_content = _get_response_content(response)
        out_tool_calls = _get_response_tool_calls(response)
        if out_content or out_tool_calls:
            body = _build_response_eval_body(out_content, self.params, tool_calls=out_tool_calls)
            opts = make_request_options(**_extra_options(self.params, roundtrip_id))
            resp = await self.client.post(
                RESPONSE_EVALUATIONS_PATH,
                cast_to=httpx.Response,
                body=body,
                options=opts,
            )
            out_res = _extract_output_result(resp.json(), original_content=out_content)
            response = self._on_output_result(response, out_res)

        return response

    # async def awrap_tool_call(
    #     self,
    #     request: ToolCallRequest,
    #     handler: Callable[[ToolCallRequest], Awaitable[Any]],
    # ) -> Any:
    #     tool_call = getattr(request, "tool_call", {}) or {}
    #     tool_name = tool_call.get("name", "<unknown>")
    #     tool_args = tool_call.get("args", {}) or {}
    #     tool_description = tool_call.get("description", "")
    #     tool_payload = {"name": tool_name, "description": tool_description, "args": tool_args}

    #     in_messages = [{"role": "user", "content": json.dumps(tool_payload)}]
    #     in_res = await self.analyze_input(in_messages)
    #     if in_res.block:
    #         raise InputBlockedError(f"Tool input for {tool_name} blocked by HiddenLayer")

    #     if in_res.redact and in_res.redacted_content:
    #         request = _apply_tool_input_redaction(request, in_res.redacted_content)

    #     output = await handler(request)

    #     out_messages = [{"role": "user", "content": str(output)}]
    #     out_res = await self.analyze_input(out_messages)
    #     if out_res.block:
    #         raise OutputBlockedError(f"Tool output from {tool_name} blocked by HiddenLayer")

    #     return out_res.redacted_content if (out_res.redact and out_res.redacted_content) else output

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
                await self.analyze_output(full_text)
            except Exception:
                logger.exception("HiddenLayer output stream scan failed")


class HiddenLayerGuardrail(HiddenLayerGuardrailBase):
    """Sync guardrail that wraps model/tool calls and enforces HiddenLayer decisions."""

    def __init__(self, params: HiddenLayerParams | None = None, client: HiddenLayer | None = None):
        super().__init__(params=params)
        self.client = client or HiddenLayer()

    # def analyze_input(self, messages: list[dict[str, Any]]) -> AnalysisResult:
    #     body = _build_request_eval_body(messages, self.params)
    #     opts = make_request_options(**_extra_options(self.params))
    #     resp = self.client.post(
    #         REQUEST_EVALUATIONS_PATH,
    #         cast_to=httpx.Response,
    #         body=body,
    #         options=opts,
    #     )
    #     return _extract_input_result(resp.json(), original_messages=messages)

    # def analyze_output(self, content: str) -> AnalysisResult:
    #     body = _build_response_eval_body(content, self.params)
    #     opts = make_request_options(**_extra_options(self.params))
    #     resp = self.client.post(
    #         RESPONSE_EVALUATIONS_PATH,
    #         cast_to=httpx.Response,
    #         body=body,
    #         options=opts,
    #     )
    #     return _extract_output_result(resp.json(), original_content=content)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        roundtrip_id = str(uuid4())
        messages = _to_openai_messages(request.messages)

        if messages:
            body = _build_request_eval_body(messages, self.params)
            if request.tools:
                body["tools"] = [_tool_to_openai(t) for t in request.tools]
            opts = make_request_options(**_extra_options(self.params, roundtrip_id))
            resp = self.client.post(
                REQUEST_EVALUATIONS_PATH,
                cast_to=httpx.Response,
                body=body,
                options=opts,
            )

            in_res = _extract_input_result(resp.json(), original_messages=messages)
            request = self._on_input_result(request, in_res)

        response = handler(request)

        out_content = _get_response_content(response)
        out_tool_calls = _get_response_tool_calls(response)
        if out_content or out_tool_calls:
            body = _build_response_eval_body(out_content, self.params, tool_calls=out_tool_calls)
            opts = make_request_options(**_extra_options(self.params, roundtrip_id))
            resp = self.client.post(
                RESPONSE_EVALUATIONS_PATH,
                cast_to=httpx.Response,
                body=body,
                options=opts,
            )
            out_res = _extract_output_result(resp.json(), original_content=out_content)
            response = self._on_output_result(response, out_res)

        return response

    # def wrap_tool_call(
    #     self,
    #     request: ToolCallRequest,
    #     handler: Callable[[ToolCallRequest], Any],
    # ) -> Any:
    #     tool_call = getattr(request, "tool_call", {}) or {}
    #     tool_name = tool_call.get("name", "<unknown>")
    #     tool_args = tool_call.get("args", {}) or {}
    #     tool_description = tool_call.get("description", "")

    #     tool_payload = {"name": tool_name, "description": tool_description, "args": tool_args}
    #     in_messages = [{"role": "user", "content": json.dumps(tool_payload)}]
    #     in_res = self.analyze_input(in_messages)
    #     if in_res.block:
    #         raise InputBlockedError(f"Tool input for {tool_name} blocked by HiddenLayer")

    #     if in_res.redact and in_res.redacted_content:
    #         request = _apply_tool_input_redaction(request, in_res.redacted_content)

    #     output = handler(request)

    #     out_messages = [{"role": "user", "content": str(output)}]
    #     out_res = self.analyze_input(out_messages)
    #     if out_res.block:
    #         raise OutputBlockedError(f"Tool output from {tool_name} blocked by HiddenLayer")

    #     return out_res.redacted_content if (out_res.redact and out_res.redacted_content) else output

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
                self.analyze_output(full_text)
            except Exception:
                logger.exception("HiddenLayer output stream scan failed")

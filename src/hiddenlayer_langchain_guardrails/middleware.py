from __future__ import annotations

import httpx
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from hiddenlayer import AsyncHiddenLayer, HiddenLayer
from hiddenlayer._base_client import make_request_options
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.tools.base import BaseTool
from pydantic import BaseModel
from uuid import uuid4

logger = logging.getLogger(__name__)

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


def _format_tool_call(tc: dict[str, Any]) -> dict[str, Any]:
    """Convert a tool call dict to OpenAI function format."""
    return {
        "id": tc.get("id", ""),
        "type": "function",
        "function": {
            "name": tc.get("name", ""),
            "arguments": json.dumps(tc.get("args", {})),
        },
    }


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
                        "tool_calls": [_format_tool_call(tc) for tc in tool_calls],
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
            if isinstance(args_schema, dict):
                parameters = args_schema
            else:
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
    headers: dict[str, str] = {
        "HL-RoundTrip-Id": roundtrip_id,
        "hl-requester-id": params.requester_id,
    }
    if params.project_id:
        headers["HL-Project-Id"] = params.project_id
    return {"extra_headers": headers}


def _extract_input_result(
    data: dict[str, Any], original_messages: list[dict[str, Any]] | None = None
) -> AnalysisResult:
    """Extract block/redact decisions from a request-evaluations response (OpenAI pass-through format).

    The API returns the provider payload directly:
    - Block: an OpenAI response payload (has ``choices``) instead of a request payload
    - Allow/Redact: the (potentially modified) OpenAI request payload (has ``messages``)
    """
    if "choices" in data:
        logger.debug("HiddenLayer request-evaluations: blocked")
        return AnalysisResult(block=True, redact=False, redacted_content=None)

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


def _get_response_output(response: ModelResponse) -> tuple[str | None, list[dict[str, Any]] | None]:
    """Extract text content and tool calls from a ModelResponse in a single pass."""
    msg = getattr(response, "message", None) or getattr(response, "result")
    if isinstance(msg, list):
        msg = msg[-1]

    content = getattr(msg, "content", None)
    content = content if isinstance(content, str) and content else None

    raw_tool_calls = getattr(msg, "tool_calls", None)
    tool_calls = [_format_tool_call(tc) for tc in raw_tool_calls] if raw_tool_calls else None

    return content, tool_calls


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


class HiddenLayerGuardrailBase(AgentMiddleware):
    """Shared logic for sync/async guardrails (params normalization and block/redact handling)."""

    def __init__(self, params: HiddenLayerParams | None = None):
        super().__init__()
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

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        roundtrip_id = str(uuid4())
        opts = make_request_options(**_extra_options(self.params, roundtrip_id))
        messages = _to_openai_messages(request.messages)

        if messages:
            body = _build_request_eval_body(messages, self.params)
            if request.tools:
                body["tools"] = [_tool_to_openai(t) for t in request.tools]
            resp = await self.client.post(REQUEST_EVALUATIONS_PATH, cast_to=httpx.Response, body=body, options=opts)
            request = self._on_input_result(request, _extract_input_result(resp.json(), original_messages=messages))

        response = await handler(request)

        out_content, out_tool_calls = _get_response_output(response)
        if out_content or out_tool_calls:
            body = _build_response_eval_body(out_content, self.params, tool_calls=out_tool_calls)
            resp = await self.client.post(RESPONSE_EVALUATIONS_PATH, cast_to=httpx.Response, body=body, options=opts)
            response = self._on_output_result(
                response, _extract_output_result(resp.json(), original_content=out_content)
            )

        return response


class HiddenLayerGuardrail(HiddenLayerGuardrailBase):
    """Sync guardrail that wraps model/tool calls and enforces HiddenLayer decisions."""

    def __init__(self, params: HiddenLayerParams | None = None, client: HiddenLayer | None = None):
        super().__init__(params=params)
        self.client = client or HiddenLayer()

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        roundtrip_id = str(uuid4())
        opts = make_request_options(**_extra_options(self.params, roundtrip_id))
        messages = _to_openai_messages(request.messages)

        if messages:
            body = _build_request_eval_body(messages, self.params)
            if request.tools:
                body["tools"] = [_tool_to_openai(t) for t in request.tools]
            resp = self.client.post(REQUEST_EVALUATIONS_PATH, cast_to=httpx.Response, body=body, options=opts)
            request = self._on_input_result(request, _extract_input_result(resp.json(), original_messages=messages))

        response = handler(request)

        out_content, out_tool_calls = _get_response_output(response)
        if out_content or out_tool_calls:
            body = _build_response_eval_body(out_content, self.params, tool_calls=out_tool_calls)
            resp = self.client.post(RESPONSE_EVALUATIONS_PATH, cast_to=httpx.Response, body=body, options=opts)
            response = self._on_output_result(
                response, _extract_output_result(resp.json(), original_content=out_content)
            )

        return response

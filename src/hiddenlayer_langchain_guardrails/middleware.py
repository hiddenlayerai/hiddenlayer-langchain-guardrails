from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from uuid import uuid4

import httpx
from hiddenlayer import AsyncHiddenLayer, HiddenLayer
from hiddenlayer._base_client import make_request_options
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.tools.base import BaseTool
from pydantic import BaseModel

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


def _get_response_output(response: ModelResponse) -> tuple[str | None, list[dict[str, Any]] | None]:
    """Extract text content and tool calls from a ModelResponse."""
    msg = getattr(response, "message", None) or getattr(response, "result")
    if isinstance(msg, list):
        msg = msg[-1]
    content = getattr(msg, "content", None)
    content = content if isinstance(content, str) and content else None
    raw_tool_calls = getattr(msg, "tool_calls", None)
    tool_calls = [_format_tool_call(tc) for tc in raw_tool_calls] if raw_tool_calls else None
    return content, tool_calls


def _extract_input_result(
    data: dict[str, Any], original_messages: list[dict[str, Any]] | None = None
) -> AnalysisResult:
    """Parse a request-evaluations response.

    Block:  API returns an OpenAI *response* payload (has ``choices``) instead of echoing the request.
    Redact: API echoes back the request with the last message content modified.
    Allow:  API echoes back the request unchanged.
    """
    if "choices" in data:
        logger.debug("HiddenLayer request-evaluations: blocked")
        return AnalysisResult(block=True, redact=False, redacted_content=None)

    messages = data.get("messages") or []
    if messages and original_messages:
        original_last = original_messages[-1].get("content", "")
        returned_last = messages[-1].get("content", "")
        if original_last != returned_last:
            logger.debug("HiddenLayer request-evaluations: redacted")
            return AnalysisResult(block=False, redact=True, redacted_content=returned_last)

    return AnalysisResult(block=False, redact=False, redacted_content=None)


def _extract_output_result(data: dict[str, Any], original_content: str | None = None) -> AnalysisResult:
    """Parse a response-evaluations response.

    The API always returns an OpenAI response payload. Content is compared
    against the original to detect redaction.
    """
    choices = data.get("choices") or []
    if not choices:
        return AnalysisResult(block=False, redact=False, redacted_content=None)

    returned_content = (choices[0].get("message") or {}).get("content")
    if returned_content is not None and original_content is not None and returned_content != original_content:
        logger.debug("HiddenLayer response-evaluations: redacted")
        return AnalysisResult(block=False, redact=True, redacted_content=returned_content)

    return AnalysisResult(block=False, redact=False, redacted_content=None)


class HiddenLayerGuardrailBase(AgentMiddleware):
    """Shared logic for sync/async guardrails (params normalization and block/redact handling)."""

    def __init__(self, params: HiddenLayerParams | None = None):
        super().__init__()
        self.params: HiddenLayerParams = params or HiddenLayerParams()

    def _make_request_options(self, roundtrip_id: str) -> Any:
        headers: dict[str, str] = {
            "HL-RoundTrip-Id": roundtrip_id,
            "hl-requester-id": self.params.requester_id,
        }
        if self.params.project_id:
            headers["HL-Project-Id"] = self.params.project_id
        return make_request_options(extra_headers=headers)

    def _make_request_eval_body(self, messages: list[dict[str, Any]], tools: list[Any] | None) -> dict[str, Any]:
        body: dict[str, Any] = {"messages": messages}
        if self.params.model:
            body["model"] = self.params.model
        if tools:
            body["tools"] = [_tool_to_openai(t) for t in tools]
        return body

    def _make_response_eval_body(
        self, out_content: str | None, out_tool_calls: list[dict[str, Any]] | None
    ) -> dict[str, Any]:
        if out_tool_calls:
            message: dict[str, Any] = {"role": "assistant", "content": None, "tool_calls": out_tool_calls}
            finish_reason = "tool_calls"
        else:
            message = {"role": "assistant", "content": out_content}
            finish_reason = "stop"
        body: dict[str, Any] = {"choices": [{"index": 0, "message": message, "finish_reason": finish_reason}]}
        if self.params.model:
            body["model"] = self.params.model
        return body

    def _on_input_result(self, request: ModelRequest, result: AnalysisResult) -> ModelRequest:
        if result.block:
            raise InputBlockedError("Input blocked by HiddenLayer")
        if result.redact and result.redacted_content:
            new_last = request.messages[-1].model_copy(update={"content": result.redacted_content})
            return request.override(messages=[*request.messages[:-1], new_last])
        return request

    def _on_output_result(self, response: ModelResponse, result: AnalysisResult) -> ModelResponse:
        if result.block:
            raise OutputBlockedError("Output blocked by HiddenLayer")
        if result.redact and result.redacted_content:
            msg = getattr(response, "message", None)
            if msg is not None:
                msg.content = result.redacted_content
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
        opts = self._make_request_options(str(uuid4()))

        messages = _to_openai_messages(request.messages)
        if messages:
            resp = await self.client.post(
                REQUEST_EVALUATIONS_PATH,
                cast_to=httpx.Response,
                body=self._make_request_eval_body(messages, request.tools),
                options=opts,
            )
            request = self._on_input_result(request, _extract_input_result(resp.json(), original_messages=messages))

        response = await handler(request)

        out_content, out_tool_calls = _get_response_output(response)
        if out_content or out_tool_calls:
            resp = await self.client.post(
                RESPONSE_EVALUATIONS_PATH,
                cast_to=httpx.Response,
                body=self._make_response_eval_body(out_content, out_tool_calls),
                options=opts,
            )
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
        opts = self._make_request_options(str(uuid4()))

        messages = _to_openai_messages(request.messages)
        if messages:
            resp = self.client.post(
                REQUEST_EVALUATIONS_PATH,
                cast_to=httpx.Response,
                body=self._make_request_eval_body(messages, request.tools),
                options=opts,
            )
            request = self._on_input_result(request, _extract_input_result(resp.json(), original_messages=messages))

        response = handler(request)

        out_content, out_tool_calls = _get_response_output(response)
        if out_content or out_tool_calls:
            resp = self.client.post(
                RESPONSE_EVALUATIONS_PATH,
                cast_to=httpx.Response,
                body=self._make_response_eval_body(out_content, out_tool_calls),
                options=opts,
            )
            response = self._on_output_result(
                response, _extract_output_result(resp.json(), original_content=out_content)
            )

        return response

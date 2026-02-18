from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from hiddenlayer_langchain_guardrails.middleware import (
    AsyncHiddenLayerGuardrail,
    HiddenLayerGuardrail,
    HiddenLayerParams,
    _extract_text_from_event,
)


# ---------------------------------------------------------------------------
# _extract_text_from_event tests
# ---------------------------------------------------------------------------


class TestExtractTextFromEvent:
    def test_plain_string(self):
        assert _extract_text_from_event("hello") == "hello"

    def test_empty_string(self):
        assert _extract_text_from_event("") == ""

    def test_object_with_string_content(self):
        event = SimpleNamespace(content="chunk text")
        assert _extract_text_from_event(event) == "chunk text"

    def test_object_with_list_content_strings(self):
        event = SimpleNamespace(content=["hello", " world"])
        assert _extract_text_from_event(event) == "hello world"

    def test_object_with_list_content_text_blocks(self):
        event = SimpleNamespace(content=[{"type": "text", "text": "foo"}, {"type": "text", "text": "bar"}])
        assert _extract_text_from_event(event) == "foobar"

    def test_object_with_list_content_mixed(self):
        event = SimpleNamespace(content=["plain", {"type": "text", "text": " block"}])
        assert _extract_text_from_event(event) == "plain block"

    def test_object_with_non_text_block_ignored(self):
        event = SimpleNamespace(content=[{"type": "image", "url": "x"}])
        assert _extract_text_from_event(event) == ""

    def test_dict_stream_event_with_chunk_string_content(self):
        chunk = SimpleNamespace(content="token")
        event = {"event": "on_chat_model_stream", "data": {"chunk": chunk}}
        assert _extract_text_from_event(event) == "token"

    def test_dict_stream_event_with_chunk_list_content(self):
        chunk = SimpleNamespace(content=["a", {"type": "text", "text": "b"}])
        event = {"event": "on_chat_model_stream", "data": {"chunk": chunk}}
        assert _extract_text_from_event(event) == "ab"

    def test_dict_stream_event_with_string_chunk(self):
        event = {"event": "on_chain_stream", "data": {"chunk": "raw text"}}
        assert _extract_text_from_event(event) == "raw text"

    def test_dict_stream_event_no_chunk(self):
        event = {"event": "on_chain_start", "data": {"input": "x"}}
        assert _extract_text_from_event(event) == ""

    def test_dict_stream_event_empty_data(self):
        event = {"event": "on_chain_start", "data": {}}
        assert _extract_text_from_event(event) == ""

    def test_none_returns_empty(self):
        assert _extract_text_from_event(None) == ""

    def test_int_returns_empty(self):
        assert _extract_text_from_event(42) == ""

    def test_object_with_none_content(self):
        event = SimpleNamespace(content=None)
        assert _extract_text_from_event(event) == ""


# ---------------------------------------------------------------------------
# Sync scan_output_stream tests
# ---------------------------------------------------------------------------


@pytest.fixture
def sync_client_mock():
    client = Mock()
    client.interactions = Mock()
    client.interactions.analyze = Mock()
    return client


@pytest.fixture
def guardrail_sync(sync_client_mock):
    params = HiddenLayerParams(model="m", project_id="p", requester_id="r")
    return HiddenLayerGuardrail(params=params, client=sync_client_mock)


class TestSyncScanOutputStream:
    def test_yields_all_events_unchanged(self, guardrail_sync, sync_client_mock, make_hl_response):
        sync_client_mock.interactions.analyze.return_value = make_hl_response(action=None, role="assistant")

        events = ["Hello", " ", "World"]
        collected = list(guardrail_sync.scan_output_stream(iter(events)))

        assert collected == ["Hello", " ", "World"]

    def test_submits_concatenated_text_after_stream(self, guardrail_sync, sync_client_mock, make_hl_response):
        sync_client_mock.interactions.analyze.return_value = make_hl_response(action=None, role="assistant")

        events = ["Hello", " ", "World"]
        list(guardrail_sync.scan_output_stream(iter(events)))

        sync_client_mock.interactions.analyze.assert_called_once()
        call_kwargs = sync_client_mock.interactions.analyze.call_args[1]
        assert call_kwargs["output"]["messages"][0]["content"] == "Hello World"

    def test_skips_analysis_when_no_text(self, guardrail_sync, sync_client_mock):
        events = [42, None, SimpleNamespace(content=None)]
        collected = list(guardrail_sync.scan_output_stream(iter(events)))

        assert len(collected) == 3
        sync_client_mock.interactions.analyze.assert_not_called()

    def test_handles_message_chunk_events(self, guardrail_sync, sync_client_mock, make_hl_response):
        sync_client_mock.interactions.analyze.return_value = make_hl_response(action=None, role="assistant")

        chunk1 = SimpleNamespace(content="Hello")
        chunk2 = SimpleNamespace(content=" World")
        events = [chunk1, chunk2]
        collected = list(guardrail_sync.scan_output_stream(iter(events)))

        assert collected == [chunk1, chunk2]
        call_kwargs = sync_client_mock.interactions.analyze.call_args[1]
        assert call_kwargs["output"]["messages"][0]["content"] == "Hello World"

    def test_handles_stream_event_dicts(self, guardrail_sync, sync_client_mock, make_hl_response):
        sync_client_mock.interactions.analyze.return_value = make_hl_response(action=None, role="assistant")

        events = [
            {"event": "on_chat_model_stream", "data": {"chunk": SimpleNamespace(content="Hi")}},
            {"event": "on_chat_model_stream", "data": {"chunk": SimpleNamespace(content="!")}},
        ]
        collected = list(guardrail_sync.scan_output_stream(iter(events)))

        assert collected == events
        call_kwargs = sync_client_mock.interactions.analyze.call_args[1]
        assert call_kwargs["output"]["messages"][0]["content"] == "Hi!"

    def test_analysis_error_is_swallowed(self, guardrail_sync, sync_client_mock):
        sync_client_mock.interactions.analyze.side_effect = RuntimeError("network error")

        events = ["Hello"]
        collected = list(guardrail_sync.scan_output_stream(iter(events)))

        assert collected == ["Hello"]
        sync_client_mock.interactions.analyze.assert_called_once()

    def test_empty_stream(self, guardrail_sync, sync_client_mock):
        collected = list(guardrail_sync.scan_output_stream(iter([])))

        assert collected == []
        sync_client_mock.interactions.analyze.assert_not_called()

    def test_mixed_event_types(self, guardrail_sync, sync_client_mock, make_hl_response):
        sync_client_mock.interactions.analyze.return_value = make_hl_response(action=None, role="assistant")

        events = [
            "Hello",
            SimpleNamespace(content=" from"),
            {"event": "on_chat_model_stream", "data": {"chunk": SimpleNamespace(content=" LLM")}},
            42,  # non-text event, ignored for buffering
        ]
        collected = list(guardrail_sync.scan_output_stream(iter(events)))

        assert len(collected) == 4
        call_kwargs = sync_client_mock.interactions.analyze.call_args[1]
        assert call_kwargs["output"]["messages"][0]["content"] == "Hello from LLM"


# ---------------------------------------------------------------------------
# Async scan_output_stream tests
# ---------------------------------------------------------------------------


@pytest.fixture
def async_client_mock():
    client = Mock()
    client.interactions = Mock()
    client.interactions.analyze = AsyncMock()
    return client


@pytest.fixture
def guardrail_async(async_client_mock):
    params = HiddenLayerParams(model="m", project_id="p", requester_id="r")
    return AsyncHiddenLayerGuardrail(params=params, client=async_client_mock)


async def _async_iter(items):
    for item in items:
        yield item


@pytest.mark.asyncio
class TestAsyncScanOutputStream:
    @pytest.mark.asyncio
    async def test_yields_all_events_unchanged(self, guardrail_async, async_client_mock, make_hl_response):
        async_client_mock.interactions.analyze.return_value = make_hl_response(action=None, role="assistant")

        events = ["Hello", " ", "World"]
        collected = [e async for e in guardrail_async.scan_output_stream(_async_iter(events))]

        assert collected == ["Hello", " ", "World"]

    @pytest.mark.asyncio
    async def test_submits_concatenated_text_after_stream(
        self, guardrail_async, async_client_mock, make_hl_response
    ):
        async_client_mock.interactions.analyze.return_value = make_hl_response(action=None, role="assistant")

        events = ["Hello", " ", "World"]
        [e async for e in guardrail_async.scan_output_stream(_async_iter(events))]

        async_client_mock.interactions.analyze.assert_called_once()
        call_kwargs = async_client_mock.interactions.analyze.call_args[1]
        assert call_kwargs["output"]["messages"][0]["content"] == "Hello World"

    @pytest.mark.asyncio
    async def test_skips_analysis_when_no_text(self, guardrail_async, async_client_mock):
        events = [42, None, SimpleNamespace(content=None)]
        collected = [e async for e in guardrail_async.scan_output_stream(_async_iter(events))]

        assert len(collected) == 3
        async_client_mock.interactions.analyze.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_message_chunk_events(self, guardrail_async, async_client_mock, make_hl_response):
        async_client_mock.interactions.analyze.return_value = make_hl_response(action=None, role="assistant")

        chunk1 = SimpleNamespace(content="Hello")
        chunk2 = SimpleNamespace(content=" World")
        events = [chunk1, chunk2]
        collected = [e async for e in guardrail_async.scan_output_stream(_async_iter(events))]

        assert collected == [chunk1, chunk2]
        call_kwargs = async_client_mock.interactions.analyze.call_args[1]
        assert call_kwargs["output"]["messages"][0]["content"] == "Hello World"

    @pytest.mark.asyncio
    async def test_handles_stream_event_dicts(self, guardrail_async, async_client_mock, make_hl_response):
        async_client_mock.interactions.analyze.return_value = make_hl_response(action=None, role="assistant")

        events = [
            {"event": "on_chat_model_stream", "data": {"chunk": SimpleNamespace(content="Hi")}},
            {"event": "on_chat_model_stream", "data": {"chunk": SimpleNamespace(content="!")}},
        ]
        collected = [e async for e in guardrail_async.scan_output_stream(_async_iter(events))]

        assert collected == events
        call_kwargs = async_client_mock.interactions.analyze.call_args[1]
        assert call_kwargs["output"]["messages"][0]["content"] == "Hi!"

    @pytest.mark.asyncio
    async def test_analysis_error_is_swallowed(self, guardrail_async, async_client_mock):
        async_client_mock.interactions.analyze.side_effect = RuntimeError("network error")

        events = ["Hello"]
        collected = [e async for e in guardrail_async.scan_output_stream(_async_iter(events))]

        assert collected == ["Hello"]
        async_client_mock.interactions.analyze.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_stream(self, guardrail_async, async_client_mock):
        collected = [e async for e in guardrail_async.scan_output_stream(_async_iter([]))]

        assert collected == []
        async_client_mock.interactions.analyze.assert_not_called()

    @pytest.mark.asyncio
    async def test_mixed_event_types(self, guardrail_async, async_client_mock, make_hl_response):
        async_client_mock.interactions.analyze.return_value = make_hl_response(action=None, role="assistant")

        events = [
            "Hello",
            SimpleNamespace(content=" from"),
            {"event": "on_chat_model_stream", "data": {"chunk": SimpleNamespace(content=" LLM")}},
            42,
        ]
        collected = [e async for e in guardrail_async.scan_output_stream(_async_iter(events))]

        assert len(collected) == 4
        call_kwargs = async_client_mock.interactions.analyze.call_args[1]
        assert call_kwargs["output"]["messages"][0]["content"] == "Hello from LLM"

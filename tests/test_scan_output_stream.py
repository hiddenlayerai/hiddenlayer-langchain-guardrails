from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from hiddenlayer_langchain_guardrails.middleware import (
    AsyncHiddenLayerGuardrail,
    HiddenLayerGuardrail,
    HiddenLayerParams,
    _extract_text_from_event,
)


class TestExtractTextFromEvent:
    def test_tuple_with_string_content(self):
        chunk = SimpleNamespace(content="hello")
        assert _extract_text_from_event((chunk, {})) == "hello"

    def test_tuple_with_empty_string_content(self):
        chunk = SimpleNamespace(content="")
        assert _extract_text_from_event((chunk, {})) == ""

    def test_tuple_with_list_content_text_blocks(self):
        chunk = SimpleNamespace(content=[{"type": "text", "text": "foo"}, {"type": "text", "text": "bar"}])
        assert _extract_text_from_event((chunk, {})) == "foobar"

    def test_tuple_with_list_content_mixed(self):
        chunk = SimpleNamespace(content=["plain", {"type": "text", "text": " block"}])
        assert _extract_text_from_event((chunk, {})) == "plain block"

    def test_tuple_with_non_text_block_ignored(self):
        chunk = SimpleNamespace(content=[{"type": "image", "url": "x"}])
        assert _extract_text_from_event((chunk, {})) == ""

    def test_tuple_with_none_content(self):
        chunk = SimpleNamespace(content=None)
        assert _extract_text_from_event((chunk, {})) == ""

    def test_tuple_with_metadata(self):
        chunk = SimpleNamespace(content="token")
        metadata = {"langgraph_node": "model", "ls_provider": "openai"}
        assert _extract_text_from_event((chunk, metadata)) == "token"

    def test_non_tuple_non_dict_returns_empty(self):
        assert _extract_text_from_event("hello") == ""
        assert _extract_text_from_event(42) == ""
        assert _extract_text_from_event(None) == ""
        assert _extract_text_from_event(SimpleNamespace(content="x")) == ""

    def test_empty_tuple_returns_empty(self):
        assert _extract_text_from_event(()) == ""

    def test_updates_mode_dict(self):
        msg = SimpleNamespace(content="hello from updates")
        event = {"model": {"messages": [msg]}}
        assert _extract_text_from_event(event) == "hello from updates"

    def test_updates_mode_dict_last_message(self):
        msgs = [SimpleNamespace(content="first"), SimpleNamespace(content="last")]
        event = {"model": {"messages": msgs}}
        assert _extract_text_from_event(event) == "last"

    def test_updates_mode_dict_no_messages(self):
        assert _extract_text_from_event({"model": {}}) == ""

    def test_updates_mode_dict_non_dict_value(self):
        assert _extract_text_from_event({"model": "not a dict"}) == ""


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
    def _chunk(self, text):
        return (SimpleNamespace(content=text), {"langgraph_node": "model"})

    def test_yields_all_events_unchanged(self, guardrail_sync, sync_client_mock, make_hl_response):
        sync_client_mock.interactions.analyze.return_value = make_hl_response(action=None, role="assistant")

        events = [self._chunk("Hello"), self._chunk(" "), self._chunk("World")]
        collected = list(guardrail_sync.safe_stream(iter(events)))

        assert collected == events

    def test_submits_concatenated_text_after_stream(self, guardrail_sync, sync_client_mock, make_hl_response):
        sync_client_mock.interactions.analyze.return_value = make_hl_response(action=None, role="assistant")

        events = [self._chunk("Hello"), self._chunk(" "), self._chunk("World")]
        list(guardrail_sync.safe_stream(iter(events)))

        sync_client_mock.interactions.analyze.assert_called_once()
        call_kwargs = sync_client_mock.interactions.analyze.call_args[1]
        assert call_kwargs["output"]["messages"][0]["content"] == "Hello World"

    def test_skips_analysis_when_no_text(self, guardrail_sync, sync_client_mock):
        events = [42, None, (SimpleNamespace(content=None), {})]
        collected = list(guardrail_sync.safe_stream(iter(events)))

        assert len(collected) == 3
        sync_client_mock.interactions.analyze.assert_not_called()

    def test_analysis_error_is_swallowed(self, guardrail_sync, sync_client_mock):
        sync_client_mock.interactions.analyze.side_effect = RuntimeError("network error")

        events = [self._chunk("Hello")]
        collected = list(guardrail_sync.safe_stream(iter(events)))

        assert collected == events
        sync_client_mock.interactions.analyze.assert_called_once()

    def test_empty_stream(self, guardrail_sync, sync_client_mock):
        collected = list(guardrail_sync.safe_stream(iter([])))

        assert collected == []
        sync_client_mock.interactions.analyze.assert_not_called()

    def test_mixed_text_and_non_text_events(self, guardrail_sync, sync_client_mock, make_hl_response):
        sync_client_mock.interactions.analyze.return_value = make_hl_response(action=None, role="assistant")

        non_text = (SimpleNamespace(content=None), {"langgraph_node": "model"})
        events = [self._chunk("Hello"), non_text, self._chunk(" World")]
        collected = list(guardrail_sync.safe_stream(iter(events)))

        assert len(collected) == 3
        call_kwargs = sync_client_mock.interactions.analyze.call_args[1]
        assert call_kwargs["output"]["messages"][0]["content"] == "Hello World"


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
    def _chunk(self, text):
        return (SimpleNamespace(content=text), {"langgraph_node": "model"})

    @pytest.mark.asyncio
    async def test_yields_all_events_unchanged(self, guardrail_async, async_client_mock, make_hl_response):
        async_client_mock.interactions.analyze.return_value = make_hl_response(action=None, role="assistant")

        events = [self._chunk("Hello"), self._chunk(" "), self._chunk("World")]
        collected = [e async for e in guardrail_async.safe_stream(_async_iter(events))]

        assert collected == events

    @pytest.mark.asyncio
    async def test_submits_concatenated_text_after_stream(self, guardrail_async, async_client_mock, make_hl_response):
        async_client_mock.interactions.analyze.return_value = make_hl_response(action=None, role="assistant")

        events = [self._chunk("Hello"), self._chunk(" "), self._chunk("World")]
        [e async for e in guardrail_async.safe_stream(_async_iter(events))]

        async_client_mock.interactions.analyze.assert_called_once()
        call_kwargs = async_client_mock.interactions.analyze.call_args[1]
        assert call_kwargs["output"]["messages"][0]["content"] == "Hello World"

    @pytest.mark.asyncio
    async def test_skips_analysis_when_no_text(self, guardrail_async, async_client_mock):
        events = [42, None, (SimpleNamespace(content=None), {})]
        collected = [e async for e in guardrail_async.safe_stream(_async_iter(events))]

        assert len(collected) == 3
        async_client_mock.interactions.analyze.assert_not_called()

    @pytest.mark.asyncio
    async def test_analysis_error_is_swallowed(self, guardrail_async, async_client_mock):
        async_client_mock.interactions.analyze.side_effect = RuntimeError("network error")

        events = [self._chunk("Hello")]
        collected = [e async for e in guardrail_async.safe_stream(_async_iter(events))]

        assert collected == events
        async_client_mock.interactions.analyze.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_stream(self, guardrail_async, async_client_mock):
        collected = [e async for e in guardrail_async.safe_stream(_async_iter([]))]

        assert collected == []
        async_client_mock.interactions.analyze.assert_not_called()

    @pytest.mark.asyncio
    async def test_mixed_text_and_non_text_events(self, guardrail_async, async_client_mock, make_hl_response):
        async_client_mock.interactions.analyze.return_value = make_hl_response(action=None, role="assistant")

        non_text = (SimpleNamespace(content=None), {"langgraph_node": "model"})
        events = [self._chunk("Hello"), non_text, self._chunk(" World")]
        collected = [e async for e in guardrail_async.safe_stream(_async_iter(events))]

        assert len(collected) == 3
        call_kwargs = async_client_mock.interactions.analyze.call_args[1]
        assert call_kwargs["output"]["messages"][0]["content"] == "Hello World"

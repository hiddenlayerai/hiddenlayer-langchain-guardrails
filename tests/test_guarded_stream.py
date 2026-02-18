"""Tests for HiddenLayerGuardrail.guarded_stream and AsyncHiddenLayerGuardrail.aguarded_stream."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call, patch

import pytest

from hiddenlayer_langchain_guardrails.middleware import (
    AsyncHiddenLayerGuardrail,
    HiddenLayerActions,
    HiddenLayerGuardrail,
    HiddenLayerParams,
    OutputBlockedError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(content):
    """Create a (msg, metadata) tuple mimicking agent.stream output."""
    return (SimpleNamespace(content=content), {"some": "meta"})


def _make_hl_response(*, action=None):
    evaluation = SimpleNamespace(action=action)
    return SimpleNamespace(evaluation=evaluation, modified_data=None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sync_client():
    client = Mock()
    client.interactions = Mock()
    client.interactions.analyze = Mock(return_value=_make_hl_response())
    return client


@pytest.fixture
def sync_guardrail(sync_client):
    params = HiddenLayerParams(model="m", project_id="p", requester_id="r")
    return HiddenLayerGuardrail(params=params, client=sync_client)


@pytest.fixture
def async_client():
    client = AsyncMock()
    client.interactions = AsyncMock()
    client.interactions.analyze = AsyncMock(return_value=_make_hl_response())
    return client


@pytest.fixture
def async_guardrail(async_client):
    params = HiddenLayerParams(model="m", project_id="p", requester_id="r")
    return AsyncHiddenLayerGuardrail(params=params, client=async_client)


# ===========================================================================
# SYNC – guarded_stream
# ===========================================================================

class TestSyncGuardedStreamThresholdZero:
    """threshold=0 (default): buffer all, scan once, then yield."""

    def test_yields_all_events_when_allowed(self, sync_guardrail, sync_client):
        events = [_make_event("Hello "), _make_event("world")]
        result = list(sync_guardrail.guarded_stream(iter(events)))

        assert result == events
        sync_client.interactions.analyze.assert_called_once()

    def test_raises_when_blocked(self, sync_guardrail, sync_client):
        sync_client.interactions.analyze.return_value = _make_hl_response(
            action=HiddenLayerActions.BLOCK,
        )
        events = [_make_event("bad content")]

        with pytest.raises(OutputBlockedError):
            list(sync_guardrail.guarded_stream(iter(events)))

    def test_no_content_skips_scan(self, sync_guardrail, sync_client):
        events = [_make_event(None), _make_event("")]
        result = list(sync_guardrail.guarded_stream(iter(events)))

        assert result == events
        sync_client.interactions.analyze.assert_not_called()

    def test_empty_stream_yields_nothing(self, sync_guardrail, sync_client):
        result = list(sync_guardrail.guarded_stream(iter([])))
        assert result == []
        sync_client.interactions.analyze.assert_not_called()

    def test_accumulates_all_content(self, sync_guardrail, sync_client):
        events = [_make_event("a"), _make_event("b"), _make_event("c")]
        list(sync_guardrail.guarded_stream(iter(events)))

        call_kwargs = sync_client.interactions.analyze.call_args
        # The output message should contain the concatenated content.
        output_msgs = call_kwargs.kwargs.get("output") or call_kwargs[1].get("output")
        assert output_msgs["messages"][0]["content"] == "abc"


class TestSyncGuardedStreamWindowed:
    """threshold > 0: yield immediately, scan in background."""

    def test_yields_events_immediately(self, sync_guardrail, sync_client):
        events = [_make_event("a"), _make_event("b"), _make_event("c")]
        result = list(sync_guardrail.guarded_stream(iter(events), threshold=2))

        assert result == events

    def test_fires_scans_per_window(self, sync_guardrail, sync_client):
        events = [_make_event("a"), _make_event("b"), _make_event("c"), _make_event("d")]

        # Patch _submit_scan_background to capture calls synchronously.
        scans = []
        original = sync_guardrail._submit_scan_background

        def capture(text):
            scans.append(text)

        sync_guardrail._submit_scan_background = capture
        list(sync_guardrail.guarded_stream(iter(events), threshold=2))

        # Window 1: "ab", Window 2: "cd"
        assert scans == ["ab", "cd"]

    def test_overlap_includes_previous_events(self, sync_guardrail, sync_client):
        events = [_make_event("a"), _make_event("b"), _make_event("c"), _make_event("d")]

        scans = []
        sync_guardrail._submit_scan_background = lambda text: scans.append(text)

        list(sync_guardrail.guarded_stream(iter(events), threshold=2, overlap=1))

        # Window 1: "ab" (no carry), Window 2: carry "b" + "cd" = "bcd"
        assert scans == ["ab", "bcd"]

    def test_remainder_flushed(self, sync_guardrail, sync_client):
        events = [_make_event("a"), _make_event("b"), _make_event("c")]

        scans = []
        sync_guardrail._submit_scan_background = lambda text: scans.append(text)

        list(sync_guardrail.guarded_stream(iter(events), threshold=2))

        # Window 1: "ab", remainder: "c"
        assert scans == ["ab", "c"]

    def test_no_content_skips_scan(self, sync_guardrail, sync_client):
        events = [_make_event(None), _make_event("")]

        scans = []
        sync_guardrail._submit_scan_background = lambda text: scans.append(text)

        list(sync_guardrail.guarded_stream(iter(events), threshold=2))
        assert scans == []

    def test_background_thread_is_started(self, sync_guardrail, sync_client):
        events = [_make_event("a"), _make_event("b")]

        with patch.object(threading, "Thread") as mock_thread:
            mock_thread.return_value = Mock()
            list(sync_guardrail.guarded_stream(iter(events), threshold=2))

            mock_thread.assert_called_once()
            mock_thread.return_value.start.assert_called_once()

    def test_handles_non_tuple_events(self, sync_guardrail, sync_client):
        """Events can be plain objects, not just tuples."""
        events = [SimpleNamespace(content="hello"), SimpleNamespace(content="world")]
        result = list(sync_guardrail.guarded_stream(iter(events)))
        assert result == events


# ===========================================================================
# ASYNC – aguarded_stream
# ===========================================================================

class TestAsyncGuardedStreamThresholdZero:
    """threshold=0 (default): buffer all, scan once, then yield."""

    @pytest.mark.asyncio
    async def test_yields_all_events_when_allowed(self, async_guardrail, async_client):
        events = [_make_event("Hello "), _make_event("world")]

        async def _aiter():
            for e in events:
                yield e

        result = [e async for e in async_guardrail.aguarded_stream(_aiter())]

        assert result == events
        async_client.interactions.analyze.assert_called_once()

    @pytest.mark.asyncio
    async def test_raises_when_blocked(self, async_guardrail, async_client):
        async_client.interactions.analyze.return_value = _make_hl_response(
            action=HiddenLayerActions.BLOCK,
        )
        events = [_make_event("bad content")]

        async def _aiter():
            for e in events:
                yield e

        with pytest.raises(OutputBlockedError):
            _ = [e async for e in async_guardrail.aguarded_stream(_aiter())]

    @pytest.mark.asyncio
    async def test_no_content_skips_scan(self, async_guardrail, async_client):
        events = [_make_event(None), _make_event("")]

        async def _aiter():
            for e in events:
                yield e

        result = [e async for e in async_guardrail.aguarded_stream(_aiter())]
        assert result == events
        async_client.interactions.analyze.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_stream(self, async_guardrail, async_client):
        async def _aiter():
            return
            yield  # noqa: unreachable — makes this an async generator

        result = [e async for e in async_guardrail.aguarded_stream(_aiter())]
        assert result == []
        async_client.interactions.analyze.assert_not_called()

    @pytest.mark.asyncio
    async def test_accumulates_all_content(self, async_guardrail, async_client):
        events = [_make_event("x"), _make_event("y"), _make_event("z")]

        async def _aiter():
            for e in events:
                yield e

        _ = [e async for e in async_guardrail.aguarded_stream(_aiter())]

        call_kwargs = async_client.interactions.analyze.call_args
        output_msgs = call_kwargs.kwargs.get("output") or call_kwargs[1].get("output")
        assert output_msgs["messages"][0]["content"] == "xyz"


class TestAsyncGuardedStreamWindowed:
    """threshold > 0: yield immediately, fire scans in background."""

    @pytest.mark.asyncio
    async def test_yields_events_immediately(self, async_guardrail, async_client):
        events = [_make_event("a"), _make_event("b"), _make_event("c")]

        async def _aiter():
            for e in events:
                yield e

        result = [e async for e in async_guardrail.aguarded_stream(_aiter(), threshold=2)]
        assert result == events

    @pytest.mark.asyncio
    async def test_fires_scans_per_window(self, async_guardrail, async_client):
        events = [_make_event("a"), _make_event("b"), _make_event("c"), _make_event("d")]

        scans = []

        async def capture(text):
            scans.append(text)

        async_guardrail._asubmit_scan = capture

        async def _aiter():
            for e in events:
                yield e

        _ = [e async for e in async_guardrail.aguarded_stream(_aiter(), threshold=2)]
        # Let scheduled tasks run.
        await asyncio.sleep(0)
        assert scans == ["ab", "cd"]

    @pytest.mark.asyncio
    async def test_overlap_includes_previous_events(self, async_guardrail, async_client):
        events = [_make_event("a"), _make_event("b"), _make_event("c"), _make_event("d")]

        scans = []

        async def capture(text):
            scans.append(text)

        async_guardrail._asubmit_scan = capture

        async def _aiter():
            for e in events:
                yield e

        _ = [e async for e in async_guardrail.aguarded_stream(_aiter(), threshold=2, overlap=1)]
        await asyncio.sleep(0)
        assert scans == ["ab", "bcd"]

    @pytest.mark.asyncio
    async def test_remainder_flushed(self, async_guardrail, async_client):
        events = [_make_event("a"), _make_event("b"), _make_event("c")]

        scans = []

        async def capture(text):
            scans.append(text)

        async_guardrail._asubmit_scan = capture

        async def _aiter():
            for e in events:
                yield e

        _ = [e async for e in async_guardrail.aguarded_stream(_aiter(), threshold=2)]
        await asyncio.sleep(0)
        assert scans == ["ab", "c"]

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hiddenlayer-langchain-guardrails")
except PackageNotFoundError:
    __version__ = "unknown"

from hiddenlayer_langchain_guardrails.middleware import (
    AsyncHiddenLayerGuardrail,
    HiddenLayerGuardrail,
    HiddenLayerParams,
    InputBlockedError,
    OutputBlockedError,
)

__all__ = [
    "__version__",
    "HiddenLayerParams",
    "HiddenLayerGuardrail",
    "AsyncHiddenLayerGuardrail",
    "InputBlockedError",
    "OutputBlockedError",
]

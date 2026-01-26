## HiddenLayer Guardrails for 🦜🔗 LangChain & LangGraph (Beta)

This package provides a LangChain agent middleware that integrates with the
[HiddenLayer Python SDK](https://github.com/hiddenlayerai/hiddenlayer-sdk-python) to scan and redact or block content before and after the agent executes.

It follows the official LangChain [custom guardrails](https://docs.langchain.com/oss/python/langchain/guardrails#custom-guardrails) middleware pattern and works with agents and tools.

### Installation

```bash
pip install hiddenlayer-langchain-guardrails
```

### Configuration
Set your credentials in your environment variables to authenticate with HiddenLayer via the SDK:

* `HIDDENLAYER_CLIENT_ID`
* `HIDDENLAYER_CLIENT_SECRET`

---

### Usage
#### Basic Agent with Guardrails
```python
from langchain.agents import create_agent
from hiddenlayer_langchain_guardrails import HiddenLayerGuardrail

agent = create_agent(
    model="gpt-4o-mini",
    tools=[get_weather],
    middleware=[HiddenLayerGuardrail()],
)

result = agent.invoke(
    {
        "messages": [
            {"role": "system", "content": "Always respond in haiku form."},
            {"role": "user", "content": "What's the weather in Toronto? Use the get_weather tool."},
        ]
    }
)

# Most agent runtimes return a dict with "messages"
print(result["messages"][-1].content if hasattr(result["messages"][-1], "content") else result["messages"][-1]["content"])
```

### How it works
- `hiddenlayer_langchain_guardrails.middleware.HiddenLayerGuardrail` provides implements `AgentMiddleware` and is configured with:
  - Model-level input/output guardrails that analyze user and assistant messages provided when the agent is invoked
  - Tool-level guardrails that inspect arguments before execution and outputs afterward
  - Personal Identifiable Information (PII) in the input/output at the model- and tool-level is redacted
- Guardrails rely on `AsyncHiddenLayer.interactions.analyze` and will raise when HiddenLayer signals a blocking action.

### Development
Run tests after installing dev deps (`pytest` and `pytest-asyncio`): `pytest tests`
Code lives in [src/hiddenlayer_langchain_guardrails/middleware.py](./src/hiddenlayer_langchain_guardrails/middleware.py); tests are in [tests/middleware.py](./tests/tests_middleware.py).

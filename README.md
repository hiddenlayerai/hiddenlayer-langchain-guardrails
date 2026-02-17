## HiddenLayer Guardrails for 🦜🔗 LangChain & LangGraph (Beta)

This package provides a LangChain agent middleware that integrates with the
[HiddenLayer Python SDK](https://github.com/hiddenlayerai/hiddenlayer-sdk-python) to scan, redact, and/or block content before and after the agent executes.

It follows the official LangChain [custom guardrails](https://docs.langchain.com/oss/python/langchain/guardrails#custom-guardrails) middleware pattern using wrap-style hooks to intercept model and tool request and responses.

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
```python
from langchain.agents import create_agent
from langchain.tools import tool
from hiddenlayer_langchain_guardrails import HiddenLayerGuardrail, HiddenLayerParams

@tool
def get_weather(city: str) -> str:
    """Return simple weather info for the specified city."""
    return f"The weather in {city} is sunny."

agent = create_agent(
    model="gpt-4o-mini",
    tools=[get_weather],
    middleware=[HiddenLayerGuardrail(
    params=HiddenLayerParams(
        model="gpt-4o-mini",
        project_id=None,          # or your HL project id
        requester_id="example",   # optional but recommended
    )
    )],
)

result = agent.invoke(
    {
        "messages": [
            {"role": "system", "content": "Always respond in haiku form."},
            {"role": "user", "content": "What's the weather in Austin? Use the get_weather tool."},
        ]
    }
)

print(result["messages"][-1].content)
```

#### Async Usage
```python
from hiddenlayer_langchain_guardrails import (
    AsyncHiddenLayerGuardrail,
    HiddenLayerParams,
)

@tool
def get_weather(city: str) -> str:
    """Return simple weather info for the specified city."""
    return f"The weather in {city} is sunny."

guardrail = AsyncHiddenLayerGuardrail(
    params=HiddenLayerParams(
        model="gpt-4o-mini",
        project_id=None,          # or your HL project id
        requester_id="example",   # optional but recommended
    )
)

agent = create_agent(
    model="gpt-4o-mini",
    tools=[get_weather],
    middleware=[guardrail],
)

async def main() -> None:
    result = await agent.ainvoke(
        {
            "messages": [
                {"role": "system", "content": "Always respond in haiku form."},
                {
                    "role": "user",
                    "content": "What's the weather in Austin? Use the get_weather tool.",
                },
            ]
        }
    )

    print(result["messages"][-1].content)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

```

### Capability Matrix

| | Alert | Block | Redact |
|---|:---:|:---:|:---:|
| **Input Guardrails** | :x: | :white_check_mark: | :white_check_mark: |
| **Output Guardrails** | :x: | :white_check_mark: | :white_check_mark: |
| **Streaming Output Guardrails** | :x: | :white_check_mark: | :x: |

### How it works
- `hiddenlayer_langchain_guardrails.middleware` provides `AsyncHiddenLayerGuardrail` and `HiddenLayerGuardrail` and is configured with:
  - Model-level input/output guardrails that analyze user and assistant messages provided when the agent is invoked
  - Tool-level guardrails that inspect arguments before execution and outputs afterward
  - Readaction in the input and output at the model- and tool-level
- Guardrails rely on the HiddenLayer REST API and will raise an exception when HiddenLayer signals a blocking action

### Development
Run tests after installing dev deps (`pytest` and `pytest-asyncio`): `pytest tests`
Code lives in [src/hiddenlayer_langchain_guardrails/middleware.py](./src/hiddenlayer_langchain_guardrails/middleware.py); tests are under the [tests](./tests/) directory.

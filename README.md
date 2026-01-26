## HiddenLayer Guardrails for 🦜🔗 LangChain & LangGraph (Beta)

This package provides a LangChain agent middleware that integrates with the
[HiddenLayer Python SDK](https://github.com/hiddenlayerai/hiddenlayer-sdk-python) to scan, redact, and/or block content before and after the agent executes.

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
            {"role": "user", "content": "What's the weather in Austin? Use the get_weather tool."},
        ]
    }
)

# Most agent runtimes return a dict with "messages"
print(result["messages"][-1].content if hasattr(result["messages"][-1], "content") else result["messages"][-1]["content"])
```

#### Basic LangGraph Node with Guardrails
```python
from typing import List, TypedDict

from langgraph.graph import StateGraph, END
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from hiddenlayer_langchain_guardrails import HiddenLayerGuardrail, InputBlockedError, OutputBlockedError

@tool
def get_weather(city: str) -> str:
    """Return a simple weather sentence for the given city."""
    return f"The weather in {city} is sunny."

agent = create_agent(
    model="gpt-4o",
    tools=[get_weather],
    middleware=[HiddenLayerGuardrail()],
)

class MyState(TypedDict):
    messages: List[SystemMessage | HumanMessage | AIMessage]

def agent_node(state: MyState) -> MyState:
    """Graph node that calls the agent and appends the assistant reply to messages.

    The middleware (HiddenLayerGuardrail) will run automatically for:
      - model input (the user's message)
      - model output (assistant response)
      - tool input and tool output when the agent invokes `get_weather`.
    """
    try:
        result = agent.invoke({"messages": state["messages"]})
        returned_messages = result.get("messages", [])
        if returned_messages:
            last = returned_messages[-1]
            # Ensure we have an AIMessage object (some runtimes return dicts)
            assistant_msg = (
                last if isinstance(last, AIMessage)
                else AIMessage(content=getattr(last, "content", str(last)))
            )
            return {"messages": state["messages"] + [assistant_msg]}
        return state

    except InputBlockedError:
        return {"messages": state["messages"] + [AIMessage(content="Blocked by guardrail (input).")]}

    except OutputBlockedError:
        return {"messages": state["messages"] + [AIMessage(content="Blocked by guardrail (output).")]}

graph = StateGraph(MyState)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.add_edge("agent", END)
app = graph.compile()

initial_state: MyState = {
    "messages": [
        SystemMessage(content="You are a helpful assistant. Answer in haiku when asked for weather."),
        HumanMessage(content="What's the weather in Austin? Use the get_weather tool."),
    ]
}

final_state = app.invoke(initial_state)

last_msg = final_state["messages"][-1]
print(getattr(last_msg, "content", str(last_msg)))
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

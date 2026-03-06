from langchain.tools import tool
from langchain.agents import create_agent

from hiddenlayer_langchain_guardrails import HiddenLayerGuardrail, HiddenLayerParams

guardrail = HiddenLayerGuardrail(
    params=HiddenLayerParams(
        model="gpt-4o-mini",
    )
)


@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""

    return f"It's always sunny in {city}!"


agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
    middleware=[
        HiddenLayerGuardrail(
            params=HiddenLayerParams(
                model="gpt-4o-mini",
            )
        )
    ],
)
for chunk in guardrail.safe_stream(
    agent.stream(
        {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
        stream_mode=["messages", "custom"],
    )
):
    continue

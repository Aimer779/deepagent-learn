import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent

# 加载环境变量
load_dotenv()

# 通过 ModelScope 接入模型（兼容 OpenAI 接口）
model = ChatOpenAI(
    model="kimi-k2.6",
    api_key=os.environ["MOONSHOT_API_KEY"],
    base_url=os.environ["MOONSHOT_BASE_URL"],
)


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"The weather in {city} is sunny!"


# 创建 Deep Agent
agent = create_deep_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant.",
)

# 运行 Agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in Beijing?"}]}
)

print(result["messages"][-1].content)

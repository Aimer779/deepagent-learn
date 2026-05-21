import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from deepagents import create_deep_agent

# 加载环境变量
load_dotenv()

# 通过 ModelScope 接入模型（兼容 OpenAI 接口）
model = ChatOpenAI(
    model="inclusionAI/Ring-2.6-1T",
    api_key=os.environ["MODELSCOPE_ACCESS_TOKEN"],
    base_url=os.environ["MODELSCOPE_BASE_URL"],
)


@tool
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

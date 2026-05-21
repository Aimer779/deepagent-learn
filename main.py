import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from deepagents import create_deep_agent
from deepagents.profiles.harness.harness_profiles import (
    HarnessProfile,
    register_harness_profile,
)

# 加载环境变量
load_dotenv()

# 通过 ModelScope 接入模型（兼容 OpenAI 接口）
model = ChatOpenAI(
    model="Qwen/Qwen3.5-27B",
    api_key=os.environ["MODELSCOPE_ACCESS_TOKEN"],
    base_url=os.environ["MODELSCOPE_BASE_URL"],
)


@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"The weather in {city} is sunny!"


# 注册 HarnessProfile，禁用与 ModelScope 不兼容的 SummarizationMiddleware
# 该中间件会导致 API 返回 choices 为 null 的错误
profile = HarnessProfile(
    excluded_middleware=frozenset({"SummarizationMiddleware"}),
)
register_harness_profile("openai", profile)

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

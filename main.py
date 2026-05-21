import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent

# 加载环境变量
load_dotenv()

# 通过 ModelScope 接入模型（兼容 OpenAI 接口）
model = ChatOpenAI(
    model="THUDM/glm-4-9b-chat",  # ModelScope 上支持 Tools 的模型
    api_key=os.environ["MODELSCOPE_ACCESS_TOKEN"],
    base_url=os.environ["MODELSCOPE_BASE_URL"],
)

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_deep_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant.",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "北京今天天气怎么样？"}]}
)

print(result["messages"][-1].content)

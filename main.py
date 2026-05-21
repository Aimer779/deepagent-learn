import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

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


# 绑定工具
model_with_tools = model.bind_tools([get_weather])

# 用户输入
messages = [HumanMessage(content="What is the weather in Beijing?")]

# 第一次调用：模型决定是否需要使用工具
response = model_with_tools.invoke(messages)

# 如果模型调用了工具，执行工具并再次调用模型
if response.tool_calls:
    messages.append(response)
    for tool_call in response.tool_calls:
        if tool_call["name"] == "get_weather":
            tool_result = get_weather.invoke(tool_call)
            messages.append(tool_result)

    # 第二次调用：获取最终回答
    final_response = model_with_tools.invoke(messages)
    print(final_response.content)
else:
    print(response.content)

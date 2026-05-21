"""获取 ModelScope 可用模型列表并更新 models.json"""
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["MODELSCOPE_ACCESS_TOKEN"]
BASE_URL = os.environ["MODELSCOPE_BASE_URL"]


def fetch_models():
    """调用 ModelScope /v1/models 接口获取模型列表"""
    url = f"{BASE_URL.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def parse_models(data: dict):
    """解析 API 返回的模型数据"""
    models = []
    for item in data.get("data", []):
        model_id = item.get("id", "")
        provider = model_id.split("/")[0] if "/" in model_id else "unknown"

        models.append({
            "id": model_id,
            "provider": provider,
            "created": item.get("created"),
        })
    return models


def save_models(models: list, path: str = "models.json"):
    """保存模型列表到 JSON 文件"""
    output = {
        "description": "ModelScope 平台可用模型列表（通过 /v1/models 接口获取）",
        "source": BASE_URL,
        "last_updated": __import__("datetime").datetime.now().isoformat(),
        "total": len(models),
        "models": models,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"已保存 {len(models)} 个模型到 {path}")


def print_summary(models: list):
    """打印模型统计摘要"""
    from collections import Counter

    providers = [m["provider"] for m in models]
    counts = Counter(providers)

    print(f"\n共获取 {len(models)} 个模型\n")
    print("按厂商分布:")
    for provider, count in counts.most_common():
        print(f"  {provider}: {count} 个")


if __name__ == "__main__":
    print("正在从 ModelScope 获取模型列表...")
    data = fetch_models()
    models = parse_models(data)
    print_summary(models)
    save_models(models)

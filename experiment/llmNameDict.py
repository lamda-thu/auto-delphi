from src.models import Qwen7B, Qwen72B, Deepseek_r1, Gpt4o

LLM_NAME_DICT = {
    "qwen2.5:7b": Qwen7B(),
    "qwen2.5:72b": Qwen72B(),
    "deepseek-r1": Deepseek_r1(),
    "gpt-4o": Gpt4o(),
}


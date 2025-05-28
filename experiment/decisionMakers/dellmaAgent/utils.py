"""
Adapted from https://github.com/DeLLMa/DeLLMa/blob/main/utils/prompt_utils.py
- Use local llms with ollama.
- Modify ANALYST_PROMPT to be more general.
"""

import json
from src.models.utils import Qwen7B


def format_query(
    query: str,
    format_instruction: str = "You should format your response as a JSON object.",
):
    return f"{query}\n{format_instruction}"


def inference(
    query: str,
    temperature: float = 0.0,
    language_model = Qwen7B(),
):
    response = language_model.invoke(
        query,
        temperature=temperature,
    )

    try:
        response = json.loads(response.lower())
    except:
        response = response

    return response
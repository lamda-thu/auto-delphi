import logging
from typing import Any
from langchain_core.language_models import LLM
from openai import OpenAI
from pydantic import Field

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)
sys.path.insert(0, os.path.abspath('.'))

# put your own api keys in the file
from APIKEYS.apiKeys import *
#from APIKEYS.qsz_deepseek_key import deepseek_key

# get env info from .env file
from dotenv import load_dotenv
load_dotenv()


class Kimi(LLM):
    # llm 属性可不定义
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "kimillm"

    def _call(self, prompt: str, **kwargs: Any) -> str:
        try:
            client = OpenAI(
                api_key=kimiKey,
                base_url="https://api.moonshot.cn/v1",
            )
            completion = client.chat.completions.create(
                model="moonshot-v1-8k",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in Kimi _call: {e}", exc_info=True)
            raise

class Llama3_1(LLM):
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "llamallm"
    
    def _call(self, prompt: str, **kwargs: Any) -> str:
        try:
            client = OpenAI(
                base_url = 'http://localhost:11434/v1',
                api_key='ollama'
            )
            temperature = kwargs["temperature"] if "temperature" in kwargs.keys() else 0

            completion = client.chat.completions.create(
                model="llama3.1",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in Llama3.1 _call: {e}", exc_info=True)
            raise
    
    
class Llama3(LLM):
    temperature: float = Field(default=0)
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "llamallm"

    def _call(self, prompt: str, **kwargs: Any) -> str:
        try:
            client = OpenAI(
                base_url = 'http://localhost:11434/v1',
                api_key='ollama'
            )
            # Use the instance temperature unless overridden in kwargs
            temperature = kwargs.get("temperature", self.temperature)

            completion = client.chat.completions.create(
                model="llama3",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in LLaMa3.1 _call: {e}", exc_info=True)
            raise

class Llama3_70B(LLM):
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "llamallm"
    
    def _call(self, prompt: str, **kwargs: Any) -> str:
        try:
            client = OpenAI(
                base_url = 'http://localhost:11434/v1',
                api_key='ollama'
            )
            temperature = kwargs["temperature"] if "temperature" in kwargs.keys() else 0

            completion = client.chat.completions.create(
                model="llama3:70b",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in Llama3_70B _call: {e}", exc_info=True)
            raise
    

class Qwen7B(LLM):
    temperature: float = Field(default=0)

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "qwenllm"

    def _call(self, prompt: str, **kwargs: Any) -> str:
        try:
            client = OpenAI(
                base_url = 'http://localhost:11434/v1',
                api_key='ollama'
            )

            temperature = kwargs.get("temperature", self.temperature)
            completion = client.chat.completions.create(
                model="qwen2.5:7b-128k",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in Qwen2.5 _call: {e}", exc_info=True)
            raise

class Qwen72B(LLM):
    temperature: float = Field(default=0)
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "qwenllm"

    def _call(self, prompt: str, **kwargs: Any) -> str:
        try:
            client = OpenAI(
                base_url = 'http://localhost:11434/v1',
                api_key='ollama'
            )
            # Use the instance temperature unless overridden in kwargs
            temperature = kwargs.get("temperature", self.temperature)

            completion = client.chat.completions.create(
                model="qwen2.5:72b-128k",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in Qwen72B _call: {e}", exc_info=True)
            raise
    

class Gemma2(LLM):
    temperature: float = Field(default=0)
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "gemmallm"

    def _call(self, prompt: str, **kwargs: Any) -> str:
        try:
            client = OpenAI(
                base_url = 'http://localhost:11434/v1',
                api_key='ollama'
            )
            # Use the instance temperature unless overridden in kwargs
            temperature = kwargs.get("temperature", self.temperature)

            completion = client.chat.completions.create(
                model="gemma2:27b",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in Gemma2 _call: {e}", exc_info=True)
            raise


class DeepseekCoder(LLM):
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "deepseekllm"

    def _call(self, prompt: str, **kwargs: Any) -> str:
        try:
            client = OpenAI(
                base_url = 'http://localhost:11434/v1',
                api_key='ollama'
            )
            temperature = kwargs["temperature"] if "temperature" in kwargs.keys() else 0

            completion = client.chat.completions.create(
                model="nezahatkorkmaz/deepseek-v3:latest",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in Deepseek-coder-v2 _call: {e}", exc_info=True)
            raise


class Phi4(LLM):
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "phillm"

    def _call(self, prompt: str, **kwargs: Any) -> str:
        try:
            client = OpenAI(
                base_url = 'http://localhost:11434/v1',
                api_key='ollama'
            )
            temperature = kwargs["temperature"] if "temperature" in kwargs.keys() else 0

            completion = client.chat.completions.create(
                model="phi4:latest",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in Phi4 _call: {e}", exc_info=True)
            raise

class Mixtral(LLM):
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "mistralllm"

    def _call(self, prompt: str, **kwargs: Any) -> str:
        try:
            client = OpenAI(
                base_url = 'http://localhost:11434/v1',
                api_key='ollama'
            )
            temperature = kwargs["temperature"] if "temperature" in kwargs.keys() else 0

            completion = client.chat.completions.create(
                model="mixtral:latest",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in Mixtral _call: {e}", exc_info=True)
            raise


class Glm4(LLM):
    def _llm_type(self) -> str:
        return "glmllm"

    def _call(self, prompt: str, **kwargs: Any) -> str:
        try:
            client = OpenAI(
                api_key=zhipuaiKey,
                base_url="https://open.bigmodel.cn/api/paas/v4/"
            )
            temperature = kwargs["temperature"] if "temperature" in kwargs.keys() else 0

            completion = client.chat.completions.create(
                model="glm-4",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in GLM-4 _call: {e}", exc_info=True)
            raise
    

class Glm4AllTools(LLM):
    def _llm_type(self) -> str:
        return "glmllm"

    def _call(self, prompt: str, **kwargs: Any) -> str:
        try:
            client = OpenAI(
                api_key=zhipuaiKey,
                base_url="https://open.bigmodel.cn/api/paas/v4/"
            )
            temperature = kwargs["temperature"] if "temperature" in kwargs.keys() else 0

            completion = client.chat.completions.create(
                model="glm-4-alltools",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in GLM-4-AllTools _call: {e}", exc_info=True)
            raise


class Gpt3_5(LLM):
    def _llm_type(self) -> str:
        return "gptllm"

    def _call(self, prompt: str, **kwargs: Any) -> str:
        try:
            client = OpenAI(
                api_key=openaiKey,
                base_url="https://api.gpt.ge/v1/"
            )
            temperature = kwargs["temperature"] if "temperature" in kwargs.keys() else 0

            completion = client.chat.completions.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in GPT-3.5 _call: {e}", exc_info=True)
            raise


class Gpt4(LLM):
    def _llm_type(self) -> str:
        return "gptllm"

    def _call(self, prompt: str, **kwargs: Any) -> str:
        try:
            client = OpenAI(
                api_key=openaiKey,
                base_url="https://api.gpt.ge/v1/"
            )
            temperature = kwargs["temperature"] if "temperature" in kwargs.keys() else 0

            completion = client.chat.completions.create(
                model="gpt-4-0125-preview",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in GPT-4 _call: {e}", exc_info=True)
            raise


class Gpt4Turbo(LLM):
    temperature: float = Field(default=0)
    
    def _llm_type(self) -> str:
        return "gptllm"

    def _call(self, prompt: str, **kwargs: Any) -> str:        
        try:
            client = OpenAI(
                api_key=openaiKey,
                base_url="https://api.gpt.ge/v1/"
            )

            # Use the instance temperature unless overridden in kwargs
            temperature = kwargs.get("temperature", self.temperature)

            tools = kwargs.get("tools", None)
            completion = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                tools=tools
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in GPT-4-Turbo _call: {e}", exc_info=True)
            raise


class Gpt4o(LLM):
    temperature: float = Field(default=0)
    @property
    def _llm_type(self) -> str:
        return "gptllm"

    def _call(self, prompt: str, **kwargs: Any) -> str:
        import time
        max_retries = 3
        initial_backoff = 1
        
        for retry in range(max_retries):
            try:
                client = OpenAI(
                    api_key=openaiKey,
                    base_url="https://api.gpt.ge/v1/"
                )
                # Use the instance temperature unless overridden in kwargs
                temperature = kwargs.get("temperature", self.temperature)
                
                # Truncate prompt if it's too long
                # This is a simple approach - you might need more sophisticated truncation
                if len(prompt) > 128000:  # Rough estimate to stay under token limit
                    prompt = prompt[:127000] + "\n\n[Content truncated due to length]"
                
                completion = client.chat.completions.create(
                    model="gpt-4o-2024-11-20",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                )
                return completion.choices[0].message.content
            except Exception as e:
                # Check if this is our last retry
                if retry == max_retries - 1:
                    logging.error(f"Error in GPT-4o _call after {max_retries} attempts: {e}", exc_info=True)
                    raise
                
                # For context length errors, truncate the prompt further and retry immediately
                if "context_length_exceeded" in str(e):
                    logging.warning(f"Context length exceeded. Truncating prompt and retrying...")
                    prompt = prompt[:int(len(prompt) * 0.8)]  # Truncate by 20%
                    continue
                
                # For other errors, use exponential backoff
                backoff_time = initial_backoff * (2 ** retry)
                logging.warning(f"Attempt {retry+1}/{max_retries} failed with error: {e}. Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)

    async def _acall(self, prompt: str, **kwargs: Any) -> str:
        """Async version of _call."""
        import asyncio
        import functools
        
        max_retries = 3
        initial_backoff = 1
        
        for retry in range(max_retries):
            try:
                client = OpenAI(
                    api_key=openaiKey,
                    base_url="https://api.gpt.ge/v1/"
                )
                
                # Use the instance temperature unless overridden in kwargs
                temperature = kwargs.get("temperature", self.temperature)
                
                # Truncate prompt if it's too long
                if len(prompt) > 128000:  # Rough estimate to stay under token limit
                    prompt = prompt[:127000] + "\n\n[Content truncated due to length]"
                
                # Run the synchronous OpenAI call in a thread pool
                loop = asyncio.get_running_loop()
                completion = await loop.run_in_executor(
                    None,
                    functools.partial(
                        client.chat.completions.create,
                        model="gpt-4o-2024-11-20",
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature
                    )
                )
                return completion.choices[0].message.content
            except Exception as e:
                # Check if this is our last retry
                if retry == max_retries - 1:
                    logging.error(f"Error in GPT-4o _acall after {max_retries} attempts: {e}", exc_info=True)
                    raise
                
                # For context length errors, truncate the prompt further and retry immediately
                if "context_length_exceeded" in str(e):
                    logging.warning(f"Context length exceeded. Truncating prompt and retrying...")
                    prompt = prompt[:int(len(prompt) * 0.8)]  # Truncate by 20%
                    continue
                
                # For other errors, use exponential backoff
                backoff_time = initial_backoff * (2 ** retry)
                logging.warning(f"Attempt {retry+1}/{max_retries} failed with error: {e}. Retrying in {backoff_time} seconds...")
                await asyncio.sleep(backoff_time)

class O1preview(LLM):
    temperature: float = Field(default=0)
    
    def _llm_type(self) -> str:
        return "gptllm"

    def _call(self, prompt: str, **kwargs: Any) -> str:        
        try:
            client = OpenAI(
                api_key=openaiKey,
                base_url="https://api.gpt.ge/v1/"
            )

            # Use the instance temperature unless overridden in kwargs
            temperature = kwargs.get("temperature", self.temperature)

            tools = kwargs.get("tools", None)
            completion = client.chat.completions.create(
                model="o1-preview",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                tools=tools
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in O1-preview _call: {e}", exc_info=True)
            raise

# class Deepseek_r1_70b(LLM):
#     def _llm_type(self) -> str:
#         return "deepseek-r1-qsz"

#     def _call(self, prompt: str, **kwargs: Any) -> str:        
#         try:
#             client = OpenAI(
#                 api_key=deepseek_key,
#                 base_url="https://gpt.sphenhe.me/api/"
#             )

#             # client = OpenAI(
#             #     api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjb2RlIjoiMTAyNyIsImlhdCI6MTc0MDg5NzAyMiwiZXhwIjoxNzQwOTE4NjIyfQ.EF5c07HLjvFRsRwP3HMKSKRbZbs0u45NBnQoNvpNME4",
#             #     base_url="https://madmodel.cs.tsinghua.edu.cn/v1/"
#             # )

#             temperature = kwargs["temperature"] if "temperature" in kwargs.keys() else 0

#             tools = kwargs["tools"] if "tools" in kwargs.keys() else None
#             completion = client.chat.completions.create(
#                 model="deepseek-r1:70b",
#                 # model="DeepSeek-R1-671B",
#                 # model="DeepSeek-R1-Distill-32B",
#                 messages=[
#                     {"role": "user", "content": prompt}
#                 ],
#                 temperature=temperature,
#                 tools=tools
#             )
#             return completion.choices[0].message.content
#         except Exception as e:
#             logging.error(f"Error in deepseek-r1:70b: {e}", exc_info=True)
#             raise

class Deepseek_r1(LLM):
    def _llm_type(self) -> str:
        return "deepseek-r1"

    def _call(self, prompt: str, **kwargs: Any) -> str:        
        try:
            client = OpenAI(
                api_key=os.getenv("ARK_API_KEY"),
                base_url="https://ark.cn-beijing.volces.com/api/v3"
            )

            temperature = kwargs["temperature"] if "temperature" in kwargs.keys() else 0

            tools = kwargs["tools"] if "tools" in kwargs.keys() else None
            completion = client.chat.completions.create(
                model="ep-20250326102849-5hhvr",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                tools=tools
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in deepseek-r1: {e}", exc_info=True)
            raise
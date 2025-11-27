import os
from typing import Any
from openai import OpenAI

from .base_llm import BaseLLM


class QwenLLM(BaseLLM):
    """通义千问(Qwen)在线模型接口，通过Alibaba Cloud Model Studio"""
    
    def __init__(self, api_key_env_var: str = "QWEN_API_KEY", model_name: str = "qwen-plus") -> None:
        api_key = os.getenv(api_key_env_var)
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env_var} not set.")
        
        # Alibaba Cloud Model Studio使用OpenAI兼容接口
        # 注意：Model Studio的base_url可能与DashScope不同
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"  # 国际版endpoint
        )
        self.model_name = model_name

    def generate(self, prompt: str, **kwargs: Any) -> str:
        temperature = kwargs.pop("temperature", kwargs.get("temp", 0.0))
        top_p = kwargs.pop("top_p", None)
        max_tokens = kwargs.get("max_tokens", 512)

        try:
            request_kwargs = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if top_p is not None:
                request_kwargs["top_p"] = top_p

            response = self.client.chat.completions.create(**request_kwargs)
            
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content.strip()
            return ""
            
        except Exception as e:
            print(f"⚠️ Qwen API错误: {e}")
            return ""




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 提供者模块 - 支持多种 LLM 后端
- OpenAI API (默认)
- 本地模型 (HuggingFace Transformers) - 需要安装 torch/transformers
- vLLM (高性能推理) - 需要安装 vllm
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import os

from logger import get_logger

logger = get_logger()


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None


class BaseLLMProvider(ABC):
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7, **kwargs) -> LLMResponse:
        pass
    
    @abstractmethod
    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: int = 500, temperature: float = 0.7, **kwargs) -> LLMResponse:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass


class OpenAIProvider(BaseLLMProvider):
    
    def __init__(self, api_key: str = None, base_url: str = None, model: str = "gpt-4"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.model = model
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7, **kwargs) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={"prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens, "total_tokens": response.usage.total_tokens} if response.usage else None,
            finish_reason=response.choices[0].finish_reason
        )
    
    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: int = 500, temperature: float = 0.7, **kwargs) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={"prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens, "total_tokens": response.usage.total_tokens} if response.usage else None,
            finish_reason=response.choices[0].finish_reason
        )
    
    def is_available(self) -> bool:
        return bool(self.api_key)


def create_llm_provider(provider_type: str = "openai", **kwargs) -> BaseLLMProvider:
    """
    创建 LLM 提供者实例
    
    Args:
        provider_type: 提供者类型 ("openai", "local", "vllm")
        **kwargs: 提供者特定参数
    
    Returns:
        BaseLLMProvider 实例
    """
    if provider_type == "openai":
        provider = OpenAIProvider(**kwargs)
    elif provider_type == "local":
        try:
            import torch
        except ImportError:
            raise RuntimeError("torch not installed. Run: pip install torch transformers accelerate safetensors")
        
        try:
            from llm_provider_local import LocalModelProvider
            provider = LocalModelProvider(**kwargs)
        except ImportError as e:
            raise RuntimeError(f"Failed to load local model provider: {e}")
            
    elif provider_type == "vllm":
        try:
            from llm_provider_vllm import VLLMProvider
            provider = VLLMProvider(**kwargs)
        except ImportError as e:
            raise RuntimeError(f"vLLM not installed or failed to load: {e}. Run: pip install vllm")
    else:
        raise ValueError(f"Unknown provider type: {provider_type}. Available: ['openai', 'local', 'vllm']")
    
    logger.info(f"Created LLM provider: {provider_type}")
    return provider

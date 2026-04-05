#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 提供者模块 - 支持多种 LLM 后端
- OpenAI API
- 本地模型 (HuggingFace Transformers)
- vLLM (高性能推理)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import os
import json

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
    def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        pass
    
    @abstractmethod
    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass


class OpenAIProvider(BaseLLMProvider):
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = "gpt-4"
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.model = model
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self._client
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
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
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            } if response.usage else None,
            finish_reason=response.choices[0].finish_reason
        )
    
    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
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
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            } if response.usage else None,
            finish_reason=response.choices[0].finish_reason
        )
    
    def is_available(self) -> bool:
        return bool(self.api_key)


class LocalModelProvider(BaseLLMProvider):
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_length: int = 4096
    ):
        self.model_path = model_path
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._pipeline = None
    
    def _load_model(self):
        if self._model is not None:
            return
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            
            logger.info(f"Loading local model from: {self.model_path}")
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            kwargs = {
                "pretrained_model_name_or_path": self.model_path,
                "trust_remote_code": True,
                "device_map": self.device,
            }
            
            if self.load_in_4bit:
                kwargs["load_in_4bit"] = True
            elif self.load_in_8bit:
                kwargs["load_in_8bit"] = True
            else:
                kwargs["torch_dtype"] = torch.float16
            
            self._model = AutoModelForCausalLM.from_pretrained(**kwargs)
            
            self._pipeline = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                max_length=self.max_length,
                device_map=self.device
            )
            
            logger.info(f"Local model loaded successfully: {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise
    
    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer
    
    @property
    def pipeline(self):
        if self._pipeline is None:
            self._load_model()
        return self._pipeline
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device != "cpu":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return LLMResponse(
            content=generated_text,
            model=self.model_path,
            usage={
                "prompt_tokens": inputs["input_ids"].shape[1],
                "completion_tokens": outputs.shape[1] - inputs["input_ids"].shape[1],
                "total_tokens": outputs.shape[1]
            }
        )
    
    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = "\n".join([
                f"{msg['role'].upper()}: {msg['content']}" 
                for msg in messages
            ]) + "\nASSISTANT:"
        
        return self.generate(prompt, max_tokens, temperature, **kwargs)
    
    def is_available(self) -> bool:
        return os.path.exists(self.model_path)


class VLLMProvider(BaseLLMProvider):
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True
    ):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        self._llm = None
        self._sampling_params = None
    
    def _load_model(self):
        if self._llm is not None:
            return
        
        try:
            from vllm import LLM, SamplingParams
            
            logger.info(f"Loading vLLM model from: {self.model_path}")
            
            self._llm = LLM(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                trust_remote_code=self.trust_remote_code
            )
            
            logger.info(f"vLLM model loaded successfully: {self.model_path}")
            
        except ImportError:
            logger.error("vLLM not installed. Please install it with: pip install vllm")
            raise
        except Exception as e:
            logger.error(f"Failed to load vLLM model: {e}")
            raise
    
    @property
    def llm(self):
        if self._llm is None:
            self._load_model()
        return self._llm
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        outputs = self.llm.generate([prompt], sampling_params)
        output = outputs[0]
        
        return LLMResponse(
            content=output.outputs[0].text,
            model=self.model_path,
            usage={
                "prompt_tokens": len(output.prompt_token_ids),
                "completion_tokens": len(output.outputs[0].token_ids),
                "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids)
            }
        )
    
    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        prompt = self.llm.get_tokenizer().apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return self.generate(prompt, max_tokens, temperature, **kwargs)
    
    def is_available(self) -> bool:
        return os.path.exists(self.model_path)


def create_llm_provider(
    provider_type: str = "openai",
    **kwargs
) -> BaseLLMProvider:
    """
    创建 LLM 提供者实例
    
    Args:
        provider_type: 提供者类型 ("openai", "local", "vllm")
        **kwargs: 提供者特定参数
    
    Returns:
        BaseLLMProvider 实例
    """
    providers = {
        "openai": OpenAIProvider,
        "local": LocalModelProvider,
        "vllm": VLLMProvider
    }
    
    if provider_type not in providers:
        raise ValueError(f"Unknown provider type: {provider_type}. "
                        f"Available: {list(providers.keys())}")
    
    provider_class = providers[provider_type]
    provider = provider_class(**kwargs)
    
    if not provider.is_available():
        raise RuntimeError(f"Provider '{provider_type}' is not available. "
                          f"Please check your configuration.")
    
    logger.info(f"Created LLM provider: {provider_type}")
    return provider

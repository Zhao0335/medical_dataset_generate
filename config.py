#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件 - 集中管理所有可配置参数
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class ToMThresholds:
    doctor_info_completeness_threshold: float = 0.80
    patient_gap_coverage_threshold: float = 0.70
    max_safety_turns: int = 15
    max_dialogue_turns: int = 12


@dataclass
class APIConfig:
    api_key: str = ""
    base_url: str = ""
    model: str = "gpt-4"
    max_tokens: int = 2500
    temperature: float = 0.3
    delay: float = 2.0
    max_retries: int = 3


@dataclass
class TaskConfig:
    system_prompt: str
    required_info: List[str]


@dataclass
class Config:
    tom_thresholds: ToMThresholds = field(default_factory=ToMThresholds)
    api: APIConfig = field(default_factory=APIConfig)
    task_configs: Dict[str, TaskConfig] = field(default_factory=dict)
    
    def __post_init__(self):
        self.task_configs = {
            "diagnosis": TaskConfig(
                system_prompt="You are an experienced doctor using Theory of Mind for diagnosis.",
                required_info=["symptoms", "duration", "severity", "medical history", "current medications"]
            ),
            "medrecon": TaskConfig(
                system_prompt="You are a clinical pharmacist using Theory of Mind for medication reconciliation.",
                required_info=["current medications", "dosages", "frequency", "adherence", "side effects"]
            ),
            "prescriptions": TaskConfig(
                system_prompt="You are a physician using Theory of Mind for prescription writing.",
                required_info=["diagnosis", "allergies", "current medications", "patient preferences"]
            )
        }
    
    @classmethod
    def from_env(cls) -> 'Config':
        config = cls()
        config.api.api_key = os.environ.get("OPENAI_API_KEY", "")
        config.api.base_url = os.environ.get("OPENAI_BASE_URL", "")
        return config
    
    @classmethod
    def from_args(cls, api_key: str = None, base_url: str = None, model: str = "gpt-4") -> 'Config':
        config = cls()
        config.api.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        config.api.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
        config.api.model = model
        return config


REQUIRED_INFO_BY_TASK: Dict[str, Dict[str, List[str]]] = {
    "diagnosis": {
        "essential": ["symptoms", "duration", "severity", "medical history"],
        "recommended": ["current medications", "allergies", "family history", "lifestyle factors"]
    },
    "medrecon": {
        "essential": ["current medications", "dosages", "frequency", "adherence"],
        "recommended": ["side effects", "drug interactions", "patient concerns", "allergies"]
    },
    "prescriptions": {
        "essential": ["diagnosis", "allergies", "current medications"],
        "recommended": ["patient preferences", "cost considerations", "lifestyle factors", "contraindications"]
    }
}

EXPLANATION_INDICATORS = [
    'this means', 'let me explain', 'the reason is', 'because', 'due to',
    'in other words', 'simply put', 'what this indicates',
    '意思是', '让我解释', '原因是', '因为', '由于',
    '换句话说', '简单来说', '这表明'
]

KNOWLEDGE_TRANSFER_INDICATORS = [
    'you should know', 'it is important', 'please understand', 'be aware',
    '你需要知道', '重要的是', '请理解', '请注意',
    'treatment plan', 'next steps', 'recommendation', 'advice',
    '治疗方案', '下一步', '建议', '意见'
]

OVER_MENTALIZING_KEYWORDS = [
    'hidden agenda', 'ulterior motive', 'manipulating', 'deceiving',
    'secretly planning', 'pretending', 'lying about',
    '隐藏动机', '欺骗', '假装', '秘密计划'
]

AVOIDANCE_SIGNALS = [
    r"(i don't know|not sure|maybe|i guess|i think so|我不知道|不太清楚|也许|大概)",
    r"(but|however|although|但是|不过|虽然)",
    r"(worried|concerned|afraid|scared|anxious|担心|害怕|顾虑|焦虑)",
    r"(hesitat|uncertain|confus|unclear|犹豫|不确定|困惑|不清楚)",
    r"(actually|to be honest|honestly|其实|说实话|老实说)",
    r"(never mind|forget it|nothing|算了|没事|没什么)"
]

KNOWLEDGE_GAP_INDICATORS = [
    r'\?',
    r'(what|why|how|when|what if|什么|为什么|怎么|何时|如果)',
    r"(i don't understand|i'm confused|can you explain|我不明白|我不懂|能解释)",
    r"(what does that mean|what do you mean|是什么意思|什么意思)",
    r"(is it|will i|do i need|是不是|我会|我需要)"
]

FIRST_ORDER_SIGNALS = [
    'but', 'however', 'worried', 'concerned', 'afraid', 'confused',
    'hesitant', 'not sure', 'i think', 'i feel', 'maybe',
    '但是', '不过', '担心', '害怕', '顾虑', '困惑', '犹豫', '不确定', '觉得', '感觉'
]

NEEDS_TOM_SIGNALS = [
    '?', 'worried', 'concerned', 'afraid', 'confused', "don't know",
    'but', 'however', 'maybe', 'think', 'feel', 'pain', 'hurt',
    '？', '担心', '害怕', '困惑', '不知道', '但是', '觉得', '感觉', '痛', '疼'
]

SIMPLE_ACKNOWLEDGMENT_PATTERNS = [
    r'^(ok|okay|yes|no|sure|alright|好的|是的|没有|行|明白).?$',
    r'^(thank you|thanks|谢谢|感谢).?$',
    r'^.{1,3}$'
]

FORBIDDEN_GENERIC_RESPONSES = [
    "I see. Can you explain more about that?",
    "Okay, I understand.",
    "Thank you for the information.",
    "I'll follow your advice.",
    "That makes sense.",
    "好的，我明白了。",
    "谢谢您的解释。",
    "我会按照您的建议做的。"
]

config = Config()

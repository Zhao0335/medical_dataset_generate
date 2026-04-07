#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ToM推理模块 - 严格落地论文1+2核心方案
双步骤ToM - Step1自主决策、Step2心理推理、自适应DoM
动态时序ToM - 时序链式推理、因果触发链、中间丢失解决
"""

import json
import re
import time
from typing import Dict, List, Any, Optional, Tuple, Union

from tom_models import (
    ToMReasoning,
    MentalState,
    CausalEvent,
    TemporalMentalTrajectory,
    TemporalChainLink,
    MentalBoundary,
    DialogueTurn,
    DoMLevel,
    TaskType
)
from tom_error_detector import ToMErrorDetector
from config import (
    config,
    FIRST_ORDER_SIGNALS,
    NEEDS_TOM_SIGNALS,
    SIMPLE_ACKNOWLEDGMENT_PATTERNS
)
from utils import format_dialogue_history, safe_json_loads, APIError
from logger import get_logger
from llm_provider import BaseLLMProvider

logger = get_logger()


class ToMReasoningModule:
    """
    ToM 推理模块
    
    功能：
    - 实现双步骤 ToM 推理
    - 动态时序心智轨迹追踪
    - 错误检测和修正
    """
    def __init__(self, llm_provider: BaseLLMProvider):
        """
        初始化 ToM 推理模块
        
        参数：
        - llm_provider: LLM 提供者实例
        """
        self.llm_provider = llm_provider
        self.error_detector = ToMErrorDetector()
        self.trajectory_history: List[TemporalMentalTrajectory] = []

    
    def step1_tom_invocation_decision(
        self,
        context: Dict[str, Any],
        dialogue_history: List[DialogueTurn],
        task_type: str
    ) -> Tuple[bool, int, str]:
        """
        Step1 ToM 调用决策 - 使用 LLM 进行决策
        
        参数：
        - context: 上下文信息
        - dialogue_history: 对话历史
        - task_type: 任务类型
        
        返回：
        - Tuple[bool, int, str]: (是否调用ToM, DoM水平, 决策理由)
        """
        # 获取完整的 EHR 数据
        ehr_input = context.get("input_text", "")
        
        # 构建 Step1 决策提示
        prompt = f"""You are performing Step1 Theory of Mind (ToM) invocation decision for medical consultation.

=== PATIENT EHR DATA ===
{ehr_input}

=== CURRENT DIALOGUE ===
{format_dialogue_history(dialogue_history)}

=== YOUR TASK ===
Make a Step1 ToM invocation decision:

1. SHOULD INVOKE TOM: Should the doctor use Theory of Mind in their next response?
   - Yes: If the patient's mental state is complex or requires empathy
   - No: If the patient is simply acknowledging information

2. DYNAMIC DOM LEVEL: Determine the appropriate level of mentalizing depth:
   - 0 (Zero-order): Direct observation, factual information exchange
   - 1 (First-order): Perspective-taking, understanding patient's beliefs and emotions

3. DECISION REASON: Provide a detailed explanation for your decision

OUTPUT FORMAT (JSON):
{
    "should_invoke_tom": true/false,
    "dom_level": 0/1,
    "decision_reason": "Detailed explanation"
}
"""
        
        try:
            response = self.llm_provider.generate_chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=config.llm.max_tokens,
                temperature=config.llm.temperature
            )
            
            result = safe_json_loads(response.content)
            if not result:
                raise APIError("Failed to parse LLM response as JSON")
            
            should_invoke = result.get("should_invoke_tom", True)
            dom_level = result.get("dom_level", 0)
            decision_reason = result.get("decision_reason", "LLM-based ToM invocation decision")
            
            return should_invoke, dom_level, decision_reason
            
        except Exception as e:
            logger.error(f"Step1 ToM invocation decision failed: {e}")
            #  fallback to default decision
            return True, 0, f"Step1 decision error: {str(e)}"
    

    
    def step2_mental_state_inference(
        self,
        context: Dict[str, Any],
        dialogue_history: List[DialogueTurn],
        dom_level: int,
        task_type: str,
        previous_trajectory: Optional[TemporalMentalTrajectory]
    ) -> ToMReasoning:
        """
        Step2 心理状态推理 - 实现动态时序心理轨迹
        
        参数：
        - context: 上下文信息
        - dialogue_history: 对话历史
        - dom_level: DoM 水平
        - task_type: 任务类型
        - previous_trajectory: 之前的心智轨迹
        
        返回：
        - ToMReasoning: ToM 推理结果
        
        异常：
        - APIError: API 错误
        """
        # 获取完整的 EHR 数据
        ehr_input = context.get("input_text", "")
        
        # 构建之前的心智状态信息
        previous_state_info = ""
        if previous_trajectory:
            previous_state_info = f"""
=== PREVIOUS MENTAL STATE ===
- Beliefs: {previous_trajectory.mental_state.beliefs}
- Emotions: {previous_trajectory.mental_state.emotions}
- Intentions: {previous_trajectory.mental_state.intentions}
- Knowledge Gaps: {previous_trajectory.mental_state.knowledge_gaps}
"""
        
        # 构建端到端的 ToM 推理提示
        prompt = f"""You are performing Step2 Theory of Mind (ToM) mental state inference for medical consultation.

=== PATIENT EHR DATA ===
{ehr_input}

=== CURRENT DIALOGUE ===
{format_dialogue_history(dialogue_history)}

{previous_state_info}

=== YOUR TASK ===
Perform comprehensive mental state inference based on the DoM level:

1. MENTAL BOUNDARY SEPARATION (Strict Isolation):
   - DOCTOR's Known Info: What the doctor knows from EHR and dialogue
   - DOCTOR's Unknown Info: What the doctor still needs to find out
   - PATIENT's Known Info: What the patient understands about their condition
   - PATIENT's Knowledge Gaps: What the patient doesn't understand

2. PATIENT's CURRENT MENTAL STATE:
   - Beliefs: What the patient believes about their condition
   - Emotions: What the patient is feeling right now
   - Intentions: What the patient wants to achieve
   - Knowledge Gaps: What the patient doesn't understand

3. DYNAMIC TEMPORAL TRAJECTORY:
   - How the patient's mental state has evolved from previous turn
   - Causal event that triggered the change
   - Temporal chain of mental state changes

4. PATIENT's POTENTIAL INTENTIONS:
   - Primary intentions the patient is pursuing
   - Hidden concerns or fears
   - Information seeking goals

5. NEXT ACTION STRATEGY:
   - Based on mental state analysis
   - Address knowledge gaps
   - Respond to emotions
   - Gather missing information

OUTPUT FORMAT (JSON):
{
    "mental_boundary": {
        "doctor_known": ["confirmed fact 1", "confirmed fact 2"],
        "doctor_unknown": ["needed info 1", "needed info 2"],
        "patient_known": ["patient knows 1", "patient knows 2"],
        "patient_knowledge_gaps": ["gap 1", "gap 2"]
    },
    "patient_mental_state": {
        "beliefs": ["current belief 1", "current belief 2"],
        "emotions": ["current emotion 1", "current emotion 2"],
        "intentions": ["current intention 1", "current intention 2"],
        "knowledge_gaps": ["current gap 1", "current gap 2"]
    },
    "temporal_trajectory": {
        "changes_from_previous": {
            "beliefs": ["belief changes"],
            "emotions": ["emotion changes"],
            "intentions": ["intention changes"],
            "knowledge_gaps": ["knowledge gap changes"]
        },
        "causal_event": {
            "trigger_event": "specific event that caused change",
            "trigger_type": "question|explanation|test result|medication discussion",
            "change_description": "what changed and why"
        },
        "temporal_chain": [
            {
                "turn_number": 1,
                "trigger_input": "what triggered this change",
                "observation": "what was observed",
                "inference": "what mental state was inferred",
                "mental_state_delta": "how mental state changed"
            }
        ]
    },
    "patient_potential_intentions": ["intention 1", "intention 2"],
    "next_action_strategy": "detailed strategy based on analysis"
}
"""
        
        try:
            response = self.llm_provider.generate_chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=config.llm.max_tokens,
                temperature=config.llm.temperature
            )
            
            result = safe_json_loads(response.content)
            if not result:
                raise APIError("Failed to parse LLM response as JSON")
            
            # 构建心智边界
            mental_boundary = MentalBoundary(
                doctor_known=result.get("mental_boundary", {}).get("doctor_known", []),
                doctor_unknown=result.get("mental_boundary", {}).get("doctor_unknown", []),
                patient_known=result.get("mental_boundary", {}).get("patient_known", []),
                patient_knowledge_gaps=result.get("mental_boundary", {}).get("patient_knowledge_gaps", [])
            )
            
            # 构建患者心理状态
            patient_mental_state = MentalState(
                beliefs=result.get("patient_mental_state", {}).get("beliefs", []),
                emotions=result.get("patient_mental_state", {}).get("emotions", []),
                intentions=result.get("patient_mental_state", {}).get("intentions", []),
                knowledge_gaps=result.get("patient_mental_state", {}).get("knowledge_gaps", [])
            )
            
            # 构建时序链
            temporal_chain = []
            for chain_item in result.get("temporal_trajectory", {}).get("temporal_chain", []):
                link = TemporalChainLink(
                    turn_number=chain_item.get("turn_number", 0),
                    trigger_input=chain_item.get("trigger_input", ""),
                    observation=chain_item.get("observation", ""),
                    inference=chain_item.get("inference", ""),
                    mental_state_delta=chain_item.get("mental_state_delta", "")
                )
                temporal_chain.append(link)
            
            # 构建因果事件
            causal_event_data = result.get("temporal_trajectory", {}).get("causal_event", {})
            causal_event = None
            if causal_event_data:
                causal_event = CausalEvent(
                    trigger_event=causal_event_data.get("trigger_event", ""),
                    trigger_type=causal_event_data.get("trigger_type", ""),
                    change_description=causal_event_data.get("change_description", "")
                )
            
            # 构建时序心智轨迹
            temporal_trajectory = TemporalMentalTrajectory(
                turn_number=len(dialogue_history),
                mental_state=patient_mental_state,
                changes_from_previous=result.get("temporal_trajectory", {}).get("changes_from_previous", {}),
                causal_event=causal_event,
                temporal_chain=temporal_chain
            )
            
            # 检测和修正 ToM 错误
            tom_reasoning = ToMReasoning(
                should_invoke_tom=True,
                dom_level=dom_level,
                step1_decision_reason="Step2 mental state inference completed",
                mental_boundary=mental_boundary,
                patient_mental_state=patient_mental_state,
                patient_potential_intentions=result.get("patient_potential_intentions", []),
                next_action_strategy=result.get("next_action_strategy", ""),
                temporal_trajectory=temporal_trajectory,
                temporal_chain_reasoning=temporal_chain
            )
            
            # 添加到轨迹历史
            self.trajectory_history.append(temporal_trajectory)
            
            return tom_reasoning
            
        except APIError as e:
            logger.error(f"API error in mental state inference: {e}")
            raise
        except Exception as e:
            logger.error(f"Mental state inference failed: {e}")
            return ToMReasoning(
                should_invoke_tom=True,
                dom_level=dom_level,
                step1_decision_reason=f"Inference error occurred: {str(e)}",
                mental_boundary=MentalBoundary(),
                patient_mental_state=MentalState()
            )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ToM推理模块 - 严格落地论文1+2核心方案
论文1：双步骤ToM - Step1自主决策、Step2心理推理、自适应DoM
论文2：动态时序ToM - 时序链式推理、因果触发链、中间丢失解决
"""

import json
import re
import time
from typing import Dict, List, Any, Optional, Tuple

from openai import OpenAI

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


class ToMReasoningModule:
    
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
        self.error_detector = ToMErrorDetector()
        self.trajectory_history: List[TemporalMentalTrajectory] = []
    
    def _determine_adaptive_dom(
        self,
        context: Dict[str, Any],
        dialogue_history: List[DialogueTurn],
        task_type: str
    ) -> int:
        """
        论文1要求：自适应DoM选择
        医疗问诊=合作场景→自动选择0阶或1阶，禁止高阶
        基于对话复杂度和患者信号动态决定
        """
        if len(dialogue_history) <= 1:
            return DoMLevel.ZERO_ORDER.value
        
        patient_utterances = [t.content for t in dialogue_history if t.role == "user"]
        if not patient_utterances:
            return DoMLevel.ZERO_ORDER.value
        
        last_utterance = patient_utterances[-1].lower()
        
        first_order_signals = [
            'but', 'however', 'worried', 'concerned', 'afraid', 'confused',
            'hesitant', 'not sure', 'i think', 'i feel', 'maybe',
            '但是', '不过', '担心', '害怕', '顾虑', '困惑', '犹豫', '不确定', '觉得', '感觉'
        ]
        
        has_first_order_signal = any(signal in last_utterance for signal in first_order_signals)
        
        if has_first_order_signal:
            return DoMLevel.FIRST_ORDER.value
        
        question_patterns = ['?', 'what', 'why', 'how', '什么', '为什么', '怎么', '如何']
        has_question = any(p in last_utterance for p in question_patterns)
        
        if has_question and len(last_utterance) > 20:
            return DoMLevel.FIRST_ORDER.value
        
        if task_type == TaskType.MEDRECON.value:
            adherence_signals = ['forgot', 'skip', 'stop', 'side effect', '忘记', '漏服', '停药', '副作用']
            if any(signal in last_utterance for signal in adherence_signals):
                return DoMLevel.FIRST_ORDER.value
        
        return DoMLevel.ZERO_ORDER.value
    
    def step1_tom_invocation_decision(
        self,
        context: Dict[str, Any],
        dialogue_history: List[DialogueTurn],
        task_type: str
    ) -> Tuple[bool, int, str]:
        """
        论文1核心：Step1自主决策是否调用ToM
        必须真实判断，可返回should_invoke_tom=False
        绝不强制永远调用
        """
        if len(dialogue_history) == 0:
            return True, 0, "Initial consultation: ToM required to establish patient baseline mental state"
        
        patient_utterances = [t.content for t in dialogue_history if t.role == "user"]
        if not patient_utterances:
            return True, 0, "No patient input: ToM needed for initial assessment"
        
        last_patient_utterance = patient_utterances[-1].strip()
        
        simple_acknowledgment_patterns = [
            r'^(ok|okay|yes|no|sure|alright|好的|是的|没有|行|明白).?$',
            r'^(thank you|thanks|谢谢|感谢).?$',
            r'^.{1,3}$'
        ]
        
        is_simple_acknowledgment = any(
            re.match(pattern, last_patient_utterance, re.IGNORECASE)
            for pattern in simple_acknowledgment_patterns
        )
        
        if is_simple_acknowledgment and len(patient_utterances) > 3:
            return False, 0, "Simple acknowledgment: No ToM needed - patient is responding to information delivery"
        
        dom_level = self._determine_adaptive_dom(context, dialogue_history, task_type)
        
        needs_tom_signals = [
            '?', 'worried', 'concerned', 'afraid', 'confused', 'don\'t know',
            'but', 'however', 'maybe', 'think', 'feel', 'pain', 'hurt',
            '？', '担心', '害怕', '困惑', '不知道', '但是', '觉得', '感觉', '痛', '疼'
        ]
        
        needs_tom = any(signal in last_patient_utterance.lower() for signal in needs_tom_signals)
        
        if needs_tom:
            return True, dom_level, f"ToM required: Patient utterance contains mental state signals (DoM={dom_level})"
        
        if dom_level == DoMLevel.FIRST_ORDER.value:
            return True, 1, "First-order ToM: Complex patient response requires perspective-taking"
        
        return True, 0, "Zero-order ToM: Standard information exchange with mental state tracking"
    
    def _build_temporal_chain_prompt(
        self,
        dialogue_history: List[DialogueTurn],
        previous_trajectory: Optional[TemporalMentalTrajectory],
        context: Dict[str, Any],
        dom_level: int
    ) -> str:
        """
        构建时序链式推理提示
        禁用普通CoT，强制时序链式推理
        """
        anchored_history = ""
        if previous_trajectory and previous_trajectory.anchored_history:
            anchored_history = f"""
=== ANCHORED HISTORY (Preventing Middle-Loss) ===
{json.dumps(previous_trajectory.anchored_history[-3:], indent=2, ensure_ascii=False)}
"""
        
        previous_state = ""
        if previous_trajectory and previous_trajectory.mental_state:
            previous_state = f"""
=== PREVIOUS MENTAL STATE (Turn {previous_trajectory.turn_number}) ===
- Beliefs: {previous_trajectory.mental_state.beliefs}
- Emotions: {previous_trajectory.mental_state.emotions}
- Intentions: {previous_trajectory.mental_state.intentions}
- Knowledge Gaps: {previous_trajectory.mental_state.knowledge_gaps}
- Chain Summary: {previous_trajectory.get_chain_summary()}
"""
        
        temporal_chain_example = """
TEMPORAL CHAIN REASONING FORMAT (NOT ordinary CoT):
Turn 1: Doctor asks about symptoms
  → Observation: Patient mentions chest pain
  → Inference: Patient believes something is wrong with heart
  → Mental Delta: +belief("heart problem"), +emotion("anxiety")
  → Evidence: "chest pain" → "heart concern"

Turn 2: Doctor explains possible causes
  → Observation: Patient asks "Is it serious?"
  → Inference: Patient's anxiety increased, seeking reassurance
  → Mental Delta: +emotion("fear"), +knowledge_gap("severity")
  → Evidence: "Is it serious?" → "fear of serious condition"
"""
        
        prompt = f"""You are performing TEMPORAL CHAIN Theory of Mind reasoning for medical consultation.

CRITICAL: This is NOT ordinary Chain-of-Thought. You must use TEMPORAL CHAIN REASONING:
1. Each inference MUST link to previous mental state
2. Each step MUST show temporal progression
3. Each conclusion MUST have evidence anchor
4. Track mental state changes across time

DoM Level: {dom_level} (0=direct observation, 1=patient's perspective)
{anchored_history}
{previous_state}

=== CURRENT DIALOGUE ===
{self._format_dialogue_history(dialogue_history)}

=== PATIENT BACKGROUND ===
{json.dumps(context.get('patient_info', {}), indent=2, ensure_ascii=False)}

{temporal_chain_example}

Perform TEMPORAL CHAIN ToM REASONING:

1. MENTAL BOUNDARY SEPARATION (Strict isolation):
   DOCTOR's Known Info: [What doctor has confirmed through dialogue/records]
   DOCTOR's Unknown Info: [What doctor still needs to find out]
   PATIENT's Known Info: [What patient understands about their condition]
   PATIENT's Knowledge Gaps: [What patient doesn't understand or is confused about]

2. TEMPORAL CHAIN REASONING (Link by link, time-ordered):
   For each turn, provide:
   - Trigger Input: What triggered this reasoning step
   - Observation: What was observed in patient's response
   - Inference: What mental state this implies
   - Mental State Delta: What changed from previous state
   - Evidence Link: Concrete evidence for the inference

3. PATIENT's CURRENT MENTAL STATE:
   - Beliefs: [What patient believes about condition]
   - Emotions: [Current emotional state]
   - Intentions: [What patient wants to achieve]
   - Knowledge Gaps: [What patient doesn't understand]

4. CAUSAL EVENT (What caused mental state change):
   - Trigger Event: [Specific event that caused change]
   - Before State: [Previous mental state]
   - After State: [Current mental state]
   - Change Description: [What changed and why]

5. PATIENT's POTENTIAL INTENTIONS:
   - Primary intentions patient is pursuing
   - Hidden concerns or fears
   - Information seeking goals

6. NEXT ACTION STRATEGY:
   - Based on temporal chain analysis
   - Address knowledge gaps
   - Respond to emotions
   - Gather missing information

OUTPUT FORMAT (JSON):
{{
    "mental_boundary": {{
        "doctor_known": ["confirmed fact 1", "confirmed fact 2"],
        "doctor_unknown": ["needed info 1", "needed info 2"],
        "patient_known": ["patient knows 1", "patient knows 2"],
        "patient_knowledge_gaps": ["gap 1", "gap 2"]
    }},
    "temporal_chain": [
        {{
            "turn_number": 1,
            "trigger_input": "Doctor's question or statement",
            "observation": "Patient's response content",
            "inference": "Mental state inference",
            "mental_state_delta": {{
                "beliefs_added": ["new belief"],
                "emotions_added": ["new emotion"],
                "intentions_added": ["new intention"],
                "gaps_added": ["new knowledge gap"]
            }},
            "evidence_links": ["evidence 1", "evidence 2"]
        }}
    ],
    "patient_mental_state": {{
        "beliefs": ["current belief 1", "current belief 2"],
        "emotions": ["current emotion 1", "current emotion 2"],
        "intentions": ["current intention 1", "current intention 2"],
        "knowledge_gaps": ["current gap 1", "current gap 2"]
    }},
    "causal_event": {{
        "trigger_event": "specific event",
        "trigger_type": "question|explanation|test result|medication discussion",
        "belief_changes": ["belief that changed"],
        "emotion_changes": ["emotion that changed"],
        "intention_changes": ["intention that changed"],
        "knowledge_gap_changes": ["gap that was filled or created"],
        "change_description": "comprehensive description"
    }},
    "patient_potential_intentions": ["intention 1", "intention 2"],
    "next_action_strategy": "detailed strategy based on temporal analysis"
}}
"""
        return prompt
    
    def step2_mental_state_inference(
        self,
        context: Dict[str, Any],
        dialogue_history: List[DialogueTurn],
        dom_level: int,
        task_type: str,
        previous_trajectory: Optional[TemporalMentalTrajectory]
    ) -> ToMReasoning:
        """
        论文1+2核心：Step2心理状态推理
        - 严格心智边界隔离
        - 时序链式推理（非普通CoT）
        - 因果触发链
        - 解决中间丢失问题
        """
        
        prompt = self._build_temporal_chain_prompt(
            dialogue_history, previous_trajectory, context, dom_level
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2500,
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            
            mental_boundary = MentalBoundary(
                doctor_known=result.get("mental_boundary", {}).get("doctor_known", []),
                doctor_unknown=result.get("mental_boundary", {}).get("doctor_unknown", []),
                patient_known=result.get("mental_boundary", {}).get("patient_known", []),
                patient_knowledge_gaps=result.get("mental_boundary", {}).get("patient_knowledge_gaps", [])
            )
            
            boundary_errors = mental_boundary.validate_separation()
            if boundary_errors:
                print(f"[WARNING] Mental boundary validation: {boundary_errors}")
            
            mental_state = MentalState(
                beliefs=result.get("patient_mental_state", {}).get("beliefs", []),
                emotions=result.get("patient_mental_state", {}).get("emotions", []),
                intentions=result.get("patient_mental_state", {}).get("intentions", []),
                knowledge_gaps=result.get("patient_mental_state", {}).get("knowledge_gaps", [])
            )
            
            patient_utterance = ""
            for turn in reversed(dialogue_history):
                if turn.role == "user":
                    patient_utterance = turn.content
                    break
            
            errors, corrected_state, corrected_intentions = self.error_detector.detect_and_correct_errors(
                patient_utterance=patient_utterance,
                mental_state=mental_state,
                intentions=result.get("patient_potential_intentions", []),
                dialogue_history=dialogue_history,
                patient_info=context.get('patient_info', {}),
                turn_number=len(dialogue_history),
                mental_boundary=mental_boundary
            )
            
            temporal_chain_links = []
            for chain_item in result.get("temporal_chain", []):
                link = TemporalChainLink(
                    turn_number=chain_item.get("turn_number", 0),
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    trigger_input=chain_item.get("trigger_input", ""),
                    observation=chain_item.get("observation", ""),
                    inference=chain_item.get("inference", ""),
                    mental_state_delta=chain_item.get("mental_state_delta", {}),
                    evidence_links=chain_item.get("evidence_links", [])
                )
                temporal_chain_links.append(link)
            
            causal_data = result.get("causal_event", {})
            causal_event = None
            if causal_data.get("trigger_event"):
                causal_event = CausalEvent(
                    trigger_event=causal_data.get("trigger_event", ""),
                    trigger_type=causal_data.get("trigger_type", ""),
                    mental_state_before=previous_trajectory.mental_state if previous_trajectory else MentalState(),
                    mental_state_after=corrected_state,
                    change_description=causal_data.get("change_description", ""),
                    belief_changes=causal_data.get("belief_changes", []),
                    emotion_changes=causal_data.get("emotion_changes", []),
                    intention_changes=causal_data.get("intention_changes", []),
                    knowledge_gap_changes=causal_data.get("knowledge_gap_changes", [])
                )
            
            anchored_history = []
            if previous_trajectory:
                anchored_history = previous_trajectory.anchored_history.copy()
            anchored_history.append({
                "turn_number": len(dialogue_history),
                "mental_state_summary": corrected_state.to_dict(),
                "key_inference": temporal_chain_links[-1].inference if temporal_chain_links else ""
            })
            if len(anchored_history) > 10:
                anchored_history = anchored_history[-10:]
            
            trajectory = TemporalMentalTrajectory(
                turn_number=len(dialogue_history),
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                mental_state=corrected_state,
                causal_event=causal_event,
                changes_from_previous={
                    "beliefs": causal_data.get("belief_changes", []),
                    "emotions": causal_data.get("emotion_changes", []),
                    "intentions": causal_data.get("intention_changes", []),
                    "knowledge_gaps": causal_data.get("knowledge_gap_changes", [])
                },
                temporal_chain=temporal_chain_links,
                anchored_history=anchored_history
            )
            
            self.trajectory_history.append(trajectory)
            
            return ToMReasoning(
                should_invoke_tom=True,
                dom_level=dom_level,
                step1_decision_reason="ToM invoked based on Step1 decision",
                mental_boundary=mental_boundary,
                patient_potential_intentions=corrected_intentions,
                patient_mental_state=corrected_state,
                next_action_strategy=result.get("next_action_strategy", ""),
                temporal_trajectory=trajectory,
                tom_errors_detected=errors,
                temporal_chain_reasoning=temporal_chain_links
            )
            
        except Exception as e:
            print(f"[ERROR] Mental state inference failed: {e}")
            import traceback
            traceback.print_exc()
            
            return ToMReasoning(
                should_invoke_tom=True,
                dom_level=dom_level,
                step1_decision_reason=f"Inference error occurred: {str(e)}",
                mental_boundary=MentalBoundary(),
                patient_mental_state=MentalState()
            )
    
    def _format_dialogue_history(self, dialogue_history: List[DialogueTurn]) -> str:
        formatted = []
        for turn in dialogue_history:
            role_label = "DOCTOR" if turn.role == "assistant" else "PATIENT"
            formatted.append(f"[Turn {turn.turn_number}] {role_label}: {turn.content}")
        return "\n".join(formatted)
    


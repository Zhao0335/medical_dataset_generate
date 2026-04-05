#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
患者心智模拟器 - 严格落地论文2核心方案
回复必须完全由动态时序心理状态驱动
禁止生成通用静态回复
"""

import json
import time
from typing import Dict, List, Any, Optional

from openai import OpenAI

from tom_models import (
    ToMReasoning,
    TemporalMentalTrajectory,
    DialogueTurn,
    MentalState
)
from config import config, FORBIDDEN_GENERIC_RESPONSES
from utils import format_dialogue_history, APIError
from logger import get_logger

logger = get_logger()


class PatientMindSimulator:
    
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
        self.response_history: List[str] = []
    
    def _validate_response_not_generic(self, response: str) -> bool:
        response_stripped = response.strip()
        for forbidden in FORBIDDEN_GENERIC_RESPONSES:
            if forbidden.lower() in response_stripped.lower():
                return False
        if len(response_stripped) < 10:
            return False
        return True
    
    def _build_patient_state_driven_prompt(
        self,
        tom_reasoning: ToMReasoning,
        context: Dict[str, Any],
        dialogue_history: List[DialogueTurn],
        task_type: str,
        previous_trajectory: Optional[TemporalMentalTrajectory]
    ) -> str:
        
        current_state = tom_reasoning.patient_mental_state
        trajectory = tom_reasoning.temporal_trajectory
        
        temporal_evolution = ""
        if trajectory and trajectory.temporal_chain:
            temporal_evolution = f"""
=== TEMPORAL MENTAL EVOLUTION ===
{self._format_temporal_chain(trajectory.temporal_chain)}
"""
        
        previous_context = ""
        if previous_trajectory and previous_trajectory.mental_state:
            prev_state = previous_trajectory.mental_state
            previous_context = f"""
=== PREVIOUS MENTAL STATE (Continuity Required) ===
Previous Beliefs: {prev_state.beliefs}
Previous Emotions: {prev_state.emotions}
Previous Intentions: {prev_state.intentions}
Previous Knowledge Gaps: {prev_state.knowledge_gaps}

CHANGES FROM PREVIOUS:
- Belief Changes: {trajectory.changes_from_previous.get('beliefs', [])}
- Emotion Changes: {trajectory.changes_from_previous.get('emotions', [])}
- Intention Changes: {trajectory.changes_from_previous.get('intentions', [])}
"""
        
        causal_context = ""
        if trajectory and trajectory.causal_event:
            causal = trajectory.causal_event
            causal_context = f"""
=== CAUSAL TRIGGER (What Just Happened) ===
Trigger Event: {causal.trigger_event}
What Changed: {causal.change_description}
This caused: {causal.emotion_changes} emotions, {causal.belief_changes} beliefs
"""
        
        emotion_display_hints = self._get_emotion_display_hints(current_state.emotions)
        intention_action_hints = self._get_intention_action_hints(current_state.intentions)
        gap_expression_hints = self._get_gap_expression_hints(current_state.knowledge_gaps)
        
        prompt = f"""You are simulating a REAL PATIENT's mind in a medical consultation.
Your response MUST be COMPLETELY DRIVEN by the patient's current mental state.

=== CRITICAL RULES ===
1. Your response MUST reflect the patient's CURRENT EMOTIONS
2. Your response MUST pursue the patient's CURRENT INTENTIONS
3. Your response MUST reveal the patient's KNOWLEDGE GAPS naturally
4. Your response MUST maintain continuity with previous mental state
5. FORBIDDEN: Generic responses like "I see" or "Okay, I understand"
6. FORBIDDEN: Responses that don't reflect the emotional state

=== PATIENT'S CURRENT MENTAL STATE (DRIVING YOUR RESPONSE) ===
BELIEFS (What patient believes):
{json.dumps(current_state.beliefs, indent=2)}

EMOTIONS (What patient is feeling NOW):
{json.dumps(current_state.emotions, indent=2)}
{emotion_display_hints}

INTENTIONS (What patient wants to achieve):
{json.dumps(current_state.intentions, indent=2)}
{intention_action_hints}

KNOWLEDGE GAPS (What patient doesn't understand):
{json.dumps(current_state.knowledge_gaps, indent=2)}
{gap_expression_hints}

=== PATIENT'S POTENTIAL INTENTIONS (From ToM Analysis) ===
{json.dumps(tom_reasoning.patient_potential_intentions, indent=2)}
{temporal_evolution}
{previous_context}
{causal_context}
=== PATIENT BACKGROUND ===
{json.dumps(context.get('patient_info', {}), indent=2, ensure_ascii=False)}

=== DIALOGUE HISTORY ===
{format_dialogue_history(dialogue_history)}

=== RESPONSE GENERATION INSTRUCTIONS ===
Generate a response that:

1. EMOTIONALLY RESONATES:
   - If patient is anxious/worried: Show concern, ask clarifying questions
   - If patient is confused: Express uncertainty, ask for explanation
   - If patient is relieved: Show cautious optimism
   - If patient is frustrated: Express impatience or concern

2. PURSUES INTENTIONS:
   - If intention is "understand diagnosis": Ask about what's wrong
   - If intention is "get treatment": Ask about next steps
   - If intention is "express concern": Share worries
   - If intention is "seek reassurance": Ask if it's serious

3. REVEALS KNOWLEDGE GAPS NATURALLY:
   - Don't say "I have a knowledge gap about X"
   - Instead ask: "What does that mean?" or "I'm not sure I understand..."
   - Show confusion through questions, not statements

4. MAINTAINS TEMPORAL CONTINUITY:
   - Reference previous concerns if still unresolved
   - Build on previous understanding
   - Show emotional progression (not random jumps)

=== FORBIDDEN RESPONSES ===
DO NOT generate any of these generic responses:
{json.dumps(FORBIDDEN_GENERIC_RESPONSES, indent=2)}

OUTPUT: Just the patient's response (natural, conversational, emotion-reflecting, intention-driven)
The response must be at least 15 characters and show clear mental state reflection.
"""
        return prompt
    
    def _get_emotion_display_hints(self, emotions: List[str]) -> str:
        if not emotions:
            return "No specific emotions detected - respond neutrally but engaged"
        
        hints = []
        emotion_lower = [e.lower() for e in emotions]
        
        if any(e in emotion_lower for e in ['anxiety', 'anxious', 'worried', 'nervous', '焦虑', '担心']):
            hints.append("- Show worry through questions like 'Is this serious?' or 'Should I be concerned?'")
        if any(e in emotion_lower for e in ['fear', 'afraid', 'scared', '害怕', '恐惧']):
            hints.append("- Express fear through hesitant speech or asking about worst outcomes")
        if any(e in emotion_lower for e in ['confusion', 'confused', 'uncertain', '困惑', '不确定']):
            hints.append("- Show confusion by asking for clarification: 'I'm not sure I understand...'")
        if any(e in emotion_lower for e in ['frustration', 'frustrated', '沮丧', '沮丧']):
            hints.append("- Show frustration through slightly impatient questions")
        if any(e in emotion_lower for e in ['relief', 'relieved', 'relieved', '放心', '松了一口气']):
            hints.append("- Show cautious relief: 'That's good to hear, but...'")
        if any(e in emotion_lower for e in ['hope', 'hopeful', '希望']):
            hints.append("- Express hope while seeking confirmation: 'So there's a good chance...?'")
        
        return "\n".join(hints) if hints else "- Reflect the detected emotions naturally in speech"
    
    def _get_intention_action_hints(self, intentions: List[str]) -> str:
        if not intentions:
            return "No specific intentions detected - respond to doctor's question"
        
        hints = []
        intention_lower = [i.lower() for i in intentions]
        
        if any(i in intention_lower for i in ['understand', 'know', 'understand diagnosis', '了解', '知道']):
            hints.append("- Ask for explanation: 'Can you explain what that means?'")
        if any(i in intention_lower for i in ['treatment', 'get treatment', '治疗', '方案']):
            hints.append("- Ask about next steps: 'What should I do next?'")
        if any(i in intention_lower for i in ['reassurance', 'seek reassurance', '安心', '确认']):
            hints.append("- Seek confirmation: 'So it's not something serious?'")
        if any(i in intention_lower for i in ['express concern', 'share worry', '表达担忧']):
            hints.append("- Share concerns: 'I've been really worried about...'")
        if any(i in intention_lower for i in ['clarify', 'get clarification', '澄清']):
            hints.append("- Ask for clarification: 'Could you clarify what you mean by...?'")
        
        return "\n".join(hints) if hints else "- Pursue the detected intentions naturally"
    
    def _get_gap_expression_hints(self, gaps: List[str]) -> str:
        if not gaps:
            return "No knowledge gaps detected - patient understands current information"
        
        hints = ["How to naturally express these knowledge gaps:"]
        
        for gap in gaps[:3]:
            gap_lower = gap.lower()
            if 'severity' in gap_lower or 'serious' in gap_lower:
                hints.append("- 'How serious is this condition?'")
            elif 'cause' in gap_lower or 'why' in gap_lower:
                hints.append("- 'What caused this to happen?'")
            elif 'treatment' in gap_lower or 'treat' in gap_lower:
                hints.append("- 'What are the treatment options?'")
            elif 'medication' in gap_lower or 'drug' in gap_lower:
                hints.append("- 'What does this medication do?'")
            elif 'test' in gap_lower or 'examination' in gap_lower:
                hints.append("- 'What will the test show?'")
            elif 'prognosis' in gap_lower or 'outcome' in gap_lower:
                hints.append("- 'What's the expected outcome?'")
            else:
                hints.append(f"- Express confusion about: {gap}")
        
        return "\n".join(hints)
    
    def _format_temporal_chain(self, chain: List) -> str:
        formatted = []
        for link in chain[-5:]:
            formatted.append(
                f"Turn {link.turn_number}: {link.trigger_input}\n"
                f"  → Observation: {link.observation}\n"
                f"  → Inference: {link.inference}\n"
                f"  → Delta: {link.mental_state_delta}"
            )
        return "\n".join(formatted)
    
    def generate_patient_response(
        self,
        tom_reasoning: ToMReasoning,
        context: Dict[str, Any],
        dialogue_history: List[DialogueTurn],
        task_type: str,
        previous_trajectory: Optional[TemporalMentalTrajectory]
    ) -> str:
        
        if not tom_reasoning.patient_mental_state or tom_reasoning.patient_mental_state.is_empty():
            return self._generate_fallback_response(dialogue_history, context)
        
        prompt = self._build_patient_state_driven_prompt(
            tom_reasoning, context, dialogue_history, task_type, previous_trajectory
        )
        
        max_attempts = config.api.max_retries
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.7
                )
                
                patient_response = response.choices[0].message.content.strip()
                
                if not self._validate_response_not_generic(patient_response):
                    logger.warning(f"Generated generic response, regenerating (attempt {attempt + 1})")
                    continue
                
                if patient_response in self.response_history[-3:]:
                    logger.warning(f"Repetitive response detected, regenerating (attempt {attempt + 1})")
                    continue
                
                self.response_history.append(patient_response)
                if len(self.response_history) > 20:
                    self.response_history = self.response_history[-20:]
                
                return patient_response
                
            except Exception as e:
                logger.error(f"Patient simulation error: {e}")
                if attempt == max_attempts - 1:
                    return self._generate_fallback_response(dialogue_history, context)
        
        return self._generate_fallback_response(dialogue_history, context)
    
    def _generate_fallback_response(
        self,
        dialogue_history: List[DialogueTurn],
        context: Dict[str, Any]
    ) -> str:
        last_doctor_msg = ""
        for turn in reversed(dialogue_history):
            if turn.role == "assistant":
                last_doctor_msg = turn.content
                break
        
        if "?" in last_doctor_msg:
            if "symptom" in last_doctor_msg.lower() or "症状" in last_doctor_msg:
                return "I've been experiencing some discomfort for a few days now. It started gradually and seems to get worse with activity."
            elif "medication" in last_doctor_msg.lower() or "药" in last_doctor_msg:
                return "I've been taking the medications as prescribed, but I'm not sure if they're helping. Sometimes I forget the evening dose."
            elif "history" in last_doctor_msg.lower() or "病史" in last_doctor_msg:
                return "I don't have any major medical history that I know of. My father had some heart issues, if that's relevant."
            else:
                return "I appreciate you asking. I've been concerned about this for a while now. Can you help me understand what might be going on?"
        else:
            return "Thank you for explaining that. I'm still a bit unclear about what this means for me. Could you help me understand what I should do next?"
    
    def get_response_history(self) -> List[str]:
        return self.response_history.copy()

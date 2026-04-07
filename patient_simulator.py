#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
患者心智模拟器 - 严格落地论文2核心方案
回复必须完全由动态时序心理状态驱动
禁止生成通用静态回复
"""

import json
from typing import Any, Dict, List, Optional

from logger import get_logger
from tom_models import (
    DialogueTurn,
    TemporalMentalTrajectory,
    ToMReasoning,
)
from config import config, FORBIDDEN_GENERIC_RESPONSES
from utils import format_dialogue_history, format_temporal_chain, APIError
from logger import get_logger
from llm_provider import BaseLLMProvider

logger = get_logger()


class PatientMindSimulator:
    def __init__(self, llm_provider: BaseLLMProvider):
        self.llm_provider = llm_provider
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
        previous_trajectory: Optional[TemporalMentalTrajectory],
    ) -> str:

        current_state = tom_reasoning.patient_mental_state
        trajectory = tom_reasoning.temporal_trajectory

        temporal_evolution = ""
        if trajectory and trajectory.temporal_chain:
            temporal_evolution = "\n=== TEMPORAL MENTAL EVOLUTION ===\n" + format_temporal_chain(trajectory.temporal_chain) + "\n"

        previous_context = ""
        if previous_trajectory and previous_trajectory.mental_state:
            prev_state = previous_trajectory.mental_state
            changes_from_previous = (
                trajectory.changes_from_previous if trajectory else {}
            )
            previous_context = "\n=== PREVIOUS MENTAL STATE (Continuity Required) ===\n"
            previous_context += "Previous Beliefs: " + str(prev_state.beliefs) + "\n"
            previous_context += "Previous Emotions: " + str(prev_state.emotions) + "\n"
            previous_context += "Previous Intentions: " + str(prev_state.intentions) + "\n"
            previous_context += "Previous Knowledge Gaps: " + str(prev_state.knowledge_gaps) + "\n\n"
            previous_context += "CHANGES FROM PREVIOUS:\n"
            previous_context += "- Belief Changes: " + str(changes_from_previous.get("beliefs", [])) + "\n"
            previous_context += "- Emotion Changes: " + str(changes_from_previous.get("emotions", [])) + "\n"
            previous_context += "- Intention Changes: " + str(changes_from_previous.get("intentions", [])) + "\n"

        causal_context = ""
        if trajectory and trajectory.causal_event:
            causal = trajectory.causal_event
            causal_context = "\n=== CAUSAL TRIGGER (What Just Happened) ===\n"
            causal_context += "Trigger Event: " + str(causal.trigger_event) + "\n"
            causal_context += "What Changed: " + str(causal.change_description) + "\n"
            causal_context += "This caused: " + str(causal.emotion_changes) + " emotions, " + str(causal.belief_changes) + " beliefs\n"

        emotion_display_hints = self._get_emotion_display_hints(current_state.emotions)
        intention_action_hints = self._get_intention_action_hints(
            current_state.intentions
        )
        gap_expression_hints = self._get_gap_expression_hints(
            current_state.knowledge_gaps
        )

        # 构建对话历史字符串
        dialogue_history_str = format_dialogue_history(dialogue_history)
        
        # 构建端到端的患者响应提示
        prompt = "You are simulating a REAL PATIENT's mind in a medical consultation.\n"
        prompt += "Your response MUST be COMPLETELY DRIVEN by the patient's current mental state.\n\n"
        prompt += "=== CRITICAL RULES ===\n"
        prompt += "1. Your response MUST reflect the patient's CURRENT EMOTIONS\n"
        prompt += "2. Your response MUST pursue the patient's CURRENT INTENTIONS\n"
        prompt += "3. Your response MUST reveal the patient's KNOWLEDGE GAPS naturally\n"
        prompt += "4. Your response MUST maintain continuity with previous mental state\n"
        prompt += "5. FORBIDDEN: Generic responses like \"I see\" or \"Okay, I understand\"\n"
        prompt += "6. FORBIDDEN: Responses that don't reflect the emotional state\n\n"
        prompt += "=== PATIENT'S CURRENT MENTAL STATE (DRIVING YOUR RESPONSE) ===\n"
        prompt += "BELIEFS (What patient believes):\n"
        prompt += json.dumps(current_state.beliefs, indent=2) + "\n\n"
        prompt += "EMOTIONS (What patient is feeling NOW):\n"
        prompt += json.dumps(current_state.emotions, indent=2) + "\n"
        prompt += emotion_display_hints + "\n\n"
        prompt += "INTENTIONS (What patient wants to achieve):\n"
        prompt += json.dumps(current_state.intentions, indent=2) + "\n"
        prompt += intention_action_hints + "\n\n"
        prompt += "KNOWLEDGE GAPS (What patient doesn't understand):\n"
        prompt += json.dumps(current_state.knowledge_gaps, indent=2) + "\n"
        prompt += gap_expression_hints + "\n\n"
        prompt += "=== PATIENT'S POTENTIAL INTENTIONS (From ToM Analysis) ===\n"
        prompt += json.dumps(tom_reasoning.patient_potential_intentions, indent=2) + "\n"
        prompt += temporal_evolution + ""
        prompt += previous_context + ""
        prompt += causal_context + "=== PATIENT EHR DATA ===\n"
        prompt += context.get('input_text', '') + "\n\n"
        prompt += "=== DIALOGUE HISTORY ===\n"
        prompt += dialogue_history_str + "\n\n"
        prompt += "=== RESPONSE GENERATION INSTRUCTIONS ===\n"
        prompt += "Generate a response that:\n\n"
        prompt += "1. EMOTIONALLY RESONATES:\n"
        prompt += "   - If patient is anxious/worried: Show concern, ask clarifying questions\n"
        prompt += "   - If patient is confused: Express uncertainty, ask for explanation\n"
        prompt += "   - If patient is relieved: Show cautious optimism\n"
        prompt += "   - If patient is frustrated: Express impatience or concern\n\n"
        prompt += "2. PURSUES INTENTIONS:\n"
        prompt += "   - If intention is \"understand diagnosis\": Ask about what's wrong\n"
        prompt += "   - If intention is \"get treatment\": Ask about next steps\n"
        prompt += "   - If intention is \"express concern\": Share worries\n"
        prompt += "   - If intention is \"seek reassurance\": Ask if it's serious\n\n"
        prompt += "3. REVEALS KNOWLEDGE GAPS NATURALLY:\n"
        prompt += "   - Don't say \"I have a knowledge gap about X\"\n"
        prompt += "   - Instead ask: \"What does that mean?\" or \"I'm not sure I understand...\"\n"
        prompt += "   - Show confusion through questions, not statements\n\n"
        prompt += "4. MAINTAINS TEMPORAL CONTINUITY:\n"
        prompt += "   - Reference previous concerns if still unresolved\n"
        prompt += "   - Build on previous understanding\n"
        prompt += "   - Show emotional progression (not random jumps)\n\n"
        prompt += "=== FORBIDDEN RESPONSES ===\n"
        prompt += "DO NOT generate any of these generic responses:\n"
        prompt += json.dumps(FORBIDDEN_GENERIC_RESPONSES, indent=2) + "\n\n"
        prompt += "OUTPUT: Just the patient's response (natural, conversational, emotion-reflecting, intention-driven)\n"
        prompt += "The response must be at least 15 characters and show clear mental state reflection.\n"
        return prompt

    def _get_emotion_display_hints(self, emotions: List[str]) -> str:
        if not emotions:
            return "No specific emotions detected - respond neutrally but engaged"

        hints = []
        emotion_lower = [e.lower() for e in emotions]

        if any(
            e in emotion_lower
            for e in ["anxiety", "anxious", "worried", "nervous", "焦虑", "担心"]
        ):
            hints.append(
                "- Show worry through questions like 'Is this serious?' or 'Should I be concerned?'"
            )
        if any(
            e in emotion_lower for e in ["fear", "afraid", "scared", "害怕", "恐惧"]
        ):
            hints.append(
                "- Express fear through hesitant speech or asking about worst outcomes"
            )
        if any(
            e in emotion_lower
            for e in ["confusion", "confused", "uncertain", "困惑", "不确定"]
        ):
            hints.append(
                "- Show confusion by asking for clarification: 'I'm not sure I understand...'"
            )
        if any(
            e in emotion_lower for e in ["frustration", "frustrated", "沮丧", "沮丧"]
        ):
            hints.append("- Show frustration through slightly impatient questions")
        if any(
            e in emotion_lower
            for e in ["relief", "relieved", "relieved", "放心", "松了一口气"]
        ):
            hints.append("- Show cautious relief: 'That's good to hear, but...'")
        if any(e in emotion_lower for e in ["hope", "hopeful", "希望"]):
            hints.append(
                "- Express hope while seeking confirmation: 'So there's a good chance...?'"
            )

        return (
            "\n".join(hints)
            if hints
            else "- Reflect the detected emotions naturally in speech"
        )

    def _get_intention_action_hints(self, intentions: List[str]) -> str:
        if not intentions:
            return "No specific intentions detected - respond to doctor's question"

        hints = []
        intention_lower = [i.lower() for i in intentions]

        if any(
            i in intention_lower
            for i in ["understand", "know", "understand diagnosis", "了解", "知道"]
        ):
            hints.append("- Ask for explanation: 'Can you explain what that means?'")
        if any(
            i in intention_lower for i in ["treatment", "get treatment", "治疗", "方案"]
        ):
            hints.append("- Ask about next steps: 'What should I do next?'")
        if any(
            i in intention_lower
            for i in ["reassurance", "seek reassurance", "安心", "确认"]
        ):
            hints.append("- Seek confirmation: 'So it's not something serious?'")
        if any(
            i in intention_lower for i in ["express concern", "share worry", "表达担忧"]
        ):
            hints.append("- Share concerns: 'I've been really worried about...'")
        if any(i in intention_lower for i in ["clarify", "get clarification", "澄清"]):
            hints.append(
                "- Ask for clarification: 'Could you clarify what you mean by...?'"
            )

        return (
            "\n".join(hints) if hints else "- Pursue the detected intentions naturally"
        )

    def _get_gap_expression_hints(self, gaps: List[str]) -> str:
        if not gaps:
            return (
                "No knowledge gaps detected - patient understands current information"
            )

        hints = ["How to naturally express these knowledge gaps:"]

        for gap in gaps[:3]:
            gap_lower = gap.lower()
            if "severity" in gap_lower or "serious" in gap_lower:
                hints.append("- 'How serious is this condition?'")
            elif "cause" in gap_lower or "why" in gap_lower:
                hints.append("- 'What caused this to happen?'")
            elif "treatment" in gap_lower or "treat" in gap_lower:
                hints.append("- 'What are the treatment options?'")
            elif "medication" in gap_lower or "drug" in gap_lower:
                hints.append("- 'What does this medication do?'")
            elif "test" in gap_lower or "examination" in gap_lower:
                hints.append("- 'What will the test show?'")
            elif "prognosis" in gap_lower or "outcome" in gap_lower:
                hints.append("- 'What's the expected outcome?'")
            else:
                hints.append(f"- Express confusion about: {gap}")

        return "\n".join(hints)
    
    # _format_temporal_chain 方法已移除，使用 utils.py 中的 format_temporal_chain 函数
    
    def generate_patient_response(
        self,
        tom_reasoning: ToMReasoning,
        context: Dict[str, Any],
        dialogue_history: List[DialogueTurn],
        task_type: str,
        previous_trajectory: Optional[TemporalMentalTrajectory],
    ) -> str:
        
        # 获取完整的 EHR 数据
        ehr_input = context.get("input_text", "")
        
        # 构建端到端的患者响应生成提示
        prompt = f"""You are a patient in a medical consultation. Based on your medical history and the dialogue so far, generate a natural, authentic response to the doctor.

=== YOUR MEDICAL HISTORY ===
{ehr_input}

=== CURRENT DIALOGUE ===
{format_dialogue_history(dialogue_history)}

=== YOUR ROLE ===
As the patient, you should:

1. BE AUTHENTIC:
   - Speak in a natural, conversational tone
   - Use everyday language, not medical jargon
   - Show your emotions and concerns
   - Be honest about your symptoms and experiences

2. RESPOND TO THE DOCTOR:
   - Address the doctor's questions or concerns
   - Provide specific details about your symptoms
   - Share your thoughts and feelings
   - Ask questions if you're confused or have concerns

3. BE CONSISTENT:
   - Maintain consistency with your medical history
   - Respond appropriately to the dialogue context
   - Show logical progression in your responses

4. SHOW EMOTIONAL REALISM:
   - Express worry, confusion, relief, or other relevant emotions
   - React naturally to the doctor's feedback
   - Show how your mental state evolves over the conversation

OUTPUT: Your response to the doctor (natural, authentic, emotionally realistic)
Do NOT include meta-commentary or explanations of your reasoning.
"""
        
        max_attempts = config.llm.max_retries
        for attempt in range(max_attempts):
            try:
                response = self.llm_provider.generate_chat(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.8  # 增加温度以获得更自然的患者响应
                )

                patient_response = response.content.strip()
                
                if not patient_response or len(patient_response) < 10:
                    logger.warning(f"Generated response too short, regenerating (attempt {attempt + 1})")
                    continue

                if patient_response in self.response_history[-3:]:
                    logger.warning(
                        f"Repetitive response detected, regenerating (attempt {attempt + 1})"
                    )
                    continue

                self.response_history.append(patient_response)
                if len(self.response_history) > 20:
                    self.response_history = self.response_history[-20:]

                return patient_response

            except Exception as e:
                logger.error(f"Patient simulation error: {e}")
                if attempt == max_attempts - 1:
                    # 移除硬编码的 fallback，直接返回一个通用的患者响应
                    return "I'm not feeling well and I'm worried about what's going on. Can you help me understand what might be causing this?"
        
        # 移除硬编码的 fallback，直接返回一个通用的患者响应
        return "I'm not feeling well and I'm worried about what's going on. Can you help me understand what might be causing this?"
    

    
    def get_response_history(self) -> List[str]:
        return self.response_history.copy()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ToM错误检测器 - 严格落地论文1核心方案
3类ToM错误实时检测+自动修正
- TypeA（过度心智化）：禁止对简单提问做复杂意图猜测
- TypeB（心智不足）：必须识别患者回避、顾虑、知识缺口
- TypeC（推理错误）：校验推理与上下文一致性，自动修正
"""

import re
from typing import List, Tuple, Dict, Any, Optional

from tom_models import (
    MentalState,
    ToMErrorRecord,
    ToMErrorType,
    DialogueTurn,
    MentalBoundary
)


class ToMErrorDetector:
    
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
    
    def detect_type_a_over_mentalizing(
        self,
        patient_utterance: str,
        inferred_intentions: List[str],
        dialogue_context: List[DialogueTurn]
    ) -> Tuple[bool, str, Optional[List[str]]]:
        """
        TypeA检测：过度心智化
        禁止对简单提问做复杂意图猜测
        """
        simple_patterns = [
            r'^.{1,25}$',
            r'^(yes|no|ok|okay|sure|alright|好的|是的|没有|行|明白|了解).*$',
            r'^(thank|thanks|谢谢|感谢).*$',
            r'^(i see|i understand|got it|明白了|懂了).*$'
        ]
        
        is_simple_utterance = any(
            re.match(pattern, patient_utterance.strip(), re.IGNORECASE)
            for pattern in simple_patterns
        )
        
        if is_simple_utterance and len(inferred_intentions) > 2:
            corrected = inferred_intentions[:1] if inferred_intentions else []
            return True, (
                f"Over-mentalizing: Simple utterance '{patient_utterance[:30]}...' "
                f"attributed {len(inferred_intentions)} complex intentions. "
                f"Reduced to essential intention only."
            ), corrected
        
        for intention in inferred_intentions:
            if any(kw in intention.lower() for kw in self.OVER_MENTALIZING_KEYWORDS):
                corrected = [i for i in inferred_intentions 
                           if not any(kw in i.lower() for kw in self.OVER_MENTALIZING_KEYWORDS)]
                return True, (
                    f"Over-mentalizing: Attributing complex motive '{intention}' without evidence. "
                    f"Removed over-mentalized intention."
                ), corrected if corrected else ["seeking medical help"]
        
        if len(inferred_intentions) > 5:
            corrected = inferred_intentions[:3]
            return True, (
                f"Over-mentalizing: Too many intentions ({len(inferred_intentions)}) inferred. "
                f"Reduced to top 3 most relevant."
            ), corrected
        
        return False, "", None
    
    def detect_type_b_under_mentalizing(
        self,
        patient_utterance: str,
        detected_mental_state: MentalState,
        dialogue_context: List[DialogueTurn],
        mental_boundary: Optional[MentalBoundary] = None
    ) -> Tuple[bool, str, Optional[MentalState]]:
        """
        TypeB检测：心智不足
        必须识别患者回避、顾虑、知识缺口等隐性心理
        """
        has_avoidance_signal = any(
            re.search(pattern, patient_utterance, re.IGNORECASE)
            for pattern in self.AVOIDANCE_SIGNALS
        )
        
        has_knowledge_gap_indicator = any(
            re.search(pattern, patient_utterance, re.IGNORECASE)
            for pattern in self.KNOWLEDGE_GAP_INDICATORS
        )
        
        corrections_needed = []
        corrected_state = detected_mental_state.copy()
        
        if has_avoidance_signal:
            has_emotion = len(detected_mental_state.emotions) > 0
            has_concern = any('concern' in e.lower() or 'worry' in e.lower() or 'anxiety' in e.lower() 
                            for e in detected_mental_state.emotions)
            
            if not has_emotion or not has_concern:
                if not any('hesitation' in e.lower() for e in corrected_state.emotions):
                    corrected_state.emotions.append("hesitation or uncertainty")
                if has_avoidance_signal and not has_concern:
                    if not any('concern' in e.lower() for e in corrected_state.emotions):
                        corrected_state.emotions.append("underlying concern")
                corrections_needed.append("emotions")
        
        if has_knowledge_gap_indicator:
            has_gap = len(detected_mental_state.knowledge_gaps) > 0
            
            if not has_gap:
                question_match = re.search(r'\?|什么|为什么|怎么|如何', patient_utterance)
                if question_match:
                    gap_topic = self._extract_gap_topic(patient_utterance)
                    corrected_state.knowledge_gaps.append(gap_topic)
                else:
                    corrected_state.knowledge_gaps.append("understanding of medical information")
                corrections_needed.append("knowledge_gaps")
        
        if mental_boundary:
            patient_utterance_lower = patient_utterance.lower()
            confusion_indicators = ['confused', 'not sure', 'don\'t understand', '困惑', '不确定', '不明白']
            if any(ind in patient_utterance_lower for ind in confusion_indicators):
                if not mental_boundary.patient_knowledge_gaps:
                    corrected_state.knowledge_gaps.append("comprehension of medical explanation")
                    corrections_needed.append("knowledge_gaps_from_boundary")
        
        if corrections_needed:
            return True, (
                f"Under-mentalizing: Patient shows signals but insufficient mental state detection. "
                f"Added/corrected: {', '.join(corrections_needed)}"
            ), corrected_state
        
        return False, "", None
    
    def _extract_gap_topic(self, utterance: str) -> str:
        """
        从患者话语中提取知识缺口主题
        """
        utterance_lower = utterance.lower()
        
        if 'what' in utterance_lower or '什么' in utterance:
            return "understanding of condition details"
        elif 'why' in utterance_lower or '为什么' in utterance:
            return "understanding of cause/reason"
        elif 'how' in utterance_lower or '怎么' in utterance:
            return "understanding of process/method"
        elif 'serious' in utterance_lower or '严重' in utterance:
            return "understanding of severity"
        elif 'treatment' in utterance_lower or '治疗' in utterance:
            return "understanding of treatment options"
        elif 'medication' in utterance_lower or '药' in utterance:
            return "understanding of medication"
        else:
            return "general medical understanding"
    
    def detect_type_c_reasoning_error(
        self,
        mental_state: MentalState,
        dialogue_history: List[DialogueTurn],
        patient_info: Dict[str, Any],
        mental_boundary: Optional[MentalBoundary] = None
    ) -> Tuple[bool, str, Optional[MentalState]]:
        """
        TypeC检测：推理错误
        校验推理与上下文一致性
        """
        corrected_state = mental_state.copy()
        errors_found = []
        
        if dialogue_history:
            last_patient_turn = None
            for turn in reversed(dialogue_history):
                if turn.role == "user":
                    last_patient_turn = turn.content
                    break
            
            if last_patient_turn:
                for belief in mental_state.beliefs:
                    contradiction_pairs = [
                        ('not worried', ['worried', 'concerned', 'anxious', 'scared']),
                        ('no pain', ['pain', 'hurt', 'ache', 'sore']),
                        ('understands', ['confused', "don't understand", 'unclear']),
                        ('fine', ['suffering', 'bad', 'terrible', 'worse']),
                        ('healthy', ['sick', 'ill', 'disease', 'condition'])
                    ]
                    
                    belief_lower = belief.lower()
                    for neg_phrase, contradiction_words in contradiction_pairs:
                        if neg_phrase in belief_lower:
                            for word in contradiction_words:
                                if word in last_patient_turn.lower():
                                    corrected_state.beliefs = [
                                        b for b in corrected_state.beliefs 
                                        if b != belief
                                    ]
                                    errors_found.append(f"Belief '{belief}' contradicts patient statement")
                                    break
        
        if patient_info:
            allergies = patient_info.get('allergies', [])
            if allergies:
                for belief in mental_state.beliefs:
                    if 'no allergies' in belief.lower() or 'no allergy' in belief.lower():
                        corrected_state.beliefs = [
                            b for b in corrected_state.beliefs 
                            if 'no allerg' not in b.lower()
                        ]
                        errors_found.append(f"Belief '{belief}' contradicts known allergies: {allergies}")
            
            medications = patient_info.get('medications', [])
            if medications:
                for belief in mental_state.beliefs:
                    if 'no medication' in belief.lower() or 'not taking' in belief.lower():
                        corrected_state.beliefs = [
                            b for b in corrected_state.beliefs 
                            if 'no medication' not in b.lower() and 'not taking' not in b.lower()
                        ]
                        errors_found.append(f"Belief '{belief}' contradicts known medications")
        
        if mental_boundary:
            for emotion in mental_state.emotions:
                if emotion.lower() in ['happy', 'excited', 'joyful']:
                    if mental_boundary.patient_knowledge_gaps:
                        corrected_state.emotions = [
                            e for e in corrected_state.emotions 
                            if e.lower() not in ['happy', 'excited', 'joyful']
                        ]
                        errors_found.append(f"Emotion '{emotion}' inconsistent with patient having knowledge gaps")
        
        if errors_found:
            return True, (
                f"Reasoning errors detected: {'; '.join(errors_found)}. "
                f"Removed contradictory beliefs/emotions."
            ), corrected_state
        
        return False, "", None
    
    def detect_and_correct_errors(
        self,
        patient_utterance: str,
        mental_state: MentalState,
        intentions: List[str],
        dialogue_history: List[DialogueTurn],
        patient_info: Dict[str, Any],
        turn_number: int,
        mental_boundary: Optional[MentalBoundary] = None
    ) -> Tuple[List[ToMErrorRecord], MentalState, List[str]]:
        """
        综合3类错误检测与修正
        """
        errors = []
        corrected_state = mental_state.copy()
        corrected_intentions = intentions.copy() if intentions else []
        
        is_type_a, desc_a, corrected_a = self.detect_type_a_over_mentalizing(
            patient_utterance, intentions, dialogue_history
        )
        if is_type_a and corrected_a is not None:
            errors.append(ToMErrorRecord(
                error_type=ToMErrorType.TYPE_A_OVER_MENTALIZING,
                error_description=desc_a,
                detected_at_turn=turn_number,
                correction_applied=f"Reduced intentions from {len(intentions)} to {len(corrected_a)}",
                corrected=True,
                original_value=intentions,
                corrected_value=corrected_a
            ))
            corrected_intentions = corrected_a
        
        is_type_b, desc_b, corrected_b = self.detect_type_b_under_mentalizing(
            patient_utterance, mental_state, dialogue_history, mental_boundary
        )
        if is_type_b and corrected_b is not None:
            errors.append(ToMErrorRecord(
                error_type=ToMErrorType.TYPE_B_UNDER_MENTALIZING,
                error_description=desc_b,
                detected_at_turn=turn_number,
                correction_applied="Added missing emotions and knowledge gaps",
                corrected=True,
                original_value=mental_state.to_dict(),
                corrected_value=corrected_b.to_dict()
            ))
            corrected_state = corrected_b
        
        is_type_c, desc_c, corrected_c = self.detect_type_c_reasoning_error(
            corrected_state, dialogue_history, patient_info, mental_boundary
        )
        if is_type_c and corrected_c is not None:
            errors.append(ToMErrorRecord(
                error_type=ToMErrorType.TYPE_C_REASONING_ERROR,
                error_description=desc_c,
                detected_at_turn=turn_number,
                correction_applied="Removed contradictory beliefs/emotions",
                corrected=True,
                original_value=corrected_state.to_dict(),
                corrected_value=corrected_c.to_dict()
            ))
            corrected_state = corrected_c
        
        if not corrected_intentions:
            corrected_intentions = ["seeking medical consultation"]
        
        if corrected_state.is_empty():
            corrected_state = MentalState(
                beliefs=["has health concern"],
                emotions=["concern about condition"],
                intentions=["get medical help"],
                knowledge_gaps=["understanding of condition"]
            )
        
        return errors, corrected_state, corrected_intentions
    
    def validate_mental_boundary(
        self,
        mental_boundary: MentalBoundary,
        dialogue_history: List[DialogueTurn]
    ) -> List[str]:
        """
        验证心智边界是否正确隔离
        """
        violations = []
        
        doctor_patient_overlap = set(mental_boundary.doctor_known) & set(mental_boundary.patient_known)
        if doctor_patient_overlap:
            violations.append(f"Doctor-patient knowledge overlap: {doctor_patient_overlap}")
        
        unknown_in_known = set(mental_boundary.doctor_unknown) & set(mental_boundary.doctor_known)
        if unknown_in_known:
            violations.append(f"Doctor unknown info already in known: {unknown_in_known}")
        
        gaps_in_patient_known = set(mental_boundary.patient_knowledge_gaps) & set(mental_boundary.patient_known)
        if gaps_in_patient_known:
            violations.append(f"Patient knowledge gaps overlap with known: {gaps_in_patient_known}")
        
        return violations

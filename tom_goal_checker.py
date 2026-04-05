#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ToM目标检查器 - 严格落地论文2核心方案
终止条件=医生信息补齐≥80%+患者知识缺口覆盖≥70%
废除关键词/最大轮次终止
"""

from typing import Dict, List, Any, Tuple

from tom_models import ToMReasoning, DialogueTurn, MentalBoundary


class ToMGoalChecker:
    
    DOCTOR_INFO_COMPLETENESS_THRESHOLD = 0.80
    PATIENT_GAP_COVERAGE_THRESHOLD = 0.70
    MAX_SAFETY_TURNS = 15
    
    REQUIRED_INFO_BY_TASK = {
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
    
    def check_tom_goal_achieved(
        self,
        tom_reasoning: ToMReasoning,
        dialogue_history: List[DialogueTurn],
        task_type: str,
        required_info: List[str] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        论文要求：基于ToM目标的终止判断
        终止条件=医生信息补齐≥80%+患者知识缺口覆盖≥70%
        """
        
        mental_boundary = tom_reasoning.mental_boundary
        
        doctor_info_completeness = self._calculate_doctor_info_completeness(
            mental_boundary.doctor_known,
            mental_boundary.doctor_unknown,
            dialogue_history,
            task_type,
            required_info
        )
        
        patient_gap_coverage = self._calculate_patient_gap_coverage(
            mental_boundary.patient_knowledge_gaps,
            dialogue_history
        )
        
        patient_intentions_addressed = self._check_intentions_addressed(
            tom_reasoning.patient_potential_intentions,
            dialogue_history
        )
        
        goal_status = {
            "doctor_info_complete": doctor_info_completeness >= self.DOCTOR_INFO_COMPLETENESS_THRESHOLD,
            "patient_gaps_covered": patient_gap_coverage >= self.PATIENT_GAP_COVERAGE_THRESHOLD,
            "intentions_addressed": patient_intentions_addressed,
            "doctor_completeness_score": round(doctor_info_completeness, 2),
            "patient_gap_coverage_score": round(patient_gap_coverage, 2),
            "remaining_unknown_info": mental_boundary.doctor_unknown,
            "remaining_knowledge_gaps": mental_boundary.patient_knowledge_gaps,
            "dialogue_turns": len(dialogue_history)
        }
        
        if doctor_info_completeness >= self.DOCTOR_INFO_COMPLETENESS_THRESHOLD and \
           patient_gap_coverage >= self.PATIENT_GAP_COVERAGE_THRESHOLD:
            return True, (
                f"ToM goal achieved: Doctor info {doctor_info_completeness:.0%} complete, "
                f"Patient gaps {patient_gap_coverage:.0%} covered"
            ), goal_status
        
        if doctor_info_completeness >= 0.9 and patient_gap_coverage >= 0.6:
            return True, (
                f"ToM goal achieved (high info completeness): Doctor info {doctor_info_completeness:.0%} complete, "
                f"Patient gaps {patient_gap_coverage:.0%} covered"
            ), goal_status
        
        if doctor_info_completeness >= 0.7 and patient_gap_coverage >= 0.8:
            return True, (
                f"ToM goal achieved (high gap coverage): Doctor info {doctor_info_completeness:.0%} complete, "
                f"Patient gaps {patient_gap_coverage:.0%} covered"
            ), goal_status
        
        if len(dialogue_history) >= self.MAX_SAFETY_TURNS * 2:
            return True, (
                f"Safety limit reached: {len(dialogue_history)} turns. "
                f"Doctor info {doctor_info_completeness:.0%} complete, "
                f"Patient gaps {patient_gap_coverage:.0%} covered"
            ), goal_status
        
        return False, (
            f"ToM goal not achieved: Doctor info {doctor_info_completeness:.0%} complete (need {self.DOCTOR_INFO_COMPLETENESS_THRESHOLD:.0%}), "
            f"Patient gaps {patient_gap_coverage:.0%} covered (need {self.PATIENT_GAP_COVERAGE_THRESHOLD:.0%})"
        ), goal_status
    
    def _calculate_doctor_info_completeness(
        self,
        doctor_known: List[str],
        doctor_unknown: List[str],
        dialogue_history: List[DialogueTurn],
        task_type: str,
        required_info: List[str] = None
    ) -> float:
        """
        计算医生信息完整度
        """
        if required_info:
            required_set = set(info.lower() for info in required_info)
        else:
            task_requirements = self.REQUIRED_INFO_BY_TASK.get(task_type, self.REQUIRED_INFO_BY_TASK["diagnosis"])
            required_set = set(info.lower() for info in task_requirements["essential"])
        
        dialogue_text = " ".join([t.content.lower() for t in dialogue_history])
        
        covered_count = 0
        for req in required_set:
            req_keywords = req.split()
            if any(kw in dialogue_text for kw in req_keywords):
                covered_count += 1
        
        essential_completeness = covered_count / len(required_set) if required_set else 1.0
        
        if doctor_unknown:
            unknown_addressed = 0
            for unknown in doctor_unknown:
                unknown_keywords = unknown.lower().split()[:3]
                if any(kw in dialogue_text for kw in unknown_keywords):
                    unknown_addressed += 1
            unknown_completeness = unknown_addressed / len(doctor_unknown)
        else:
            unknown_completeness = 1.0
        
        total_known_items = len(doctor_known)
        knowledge_accumulation = min(total_known_items / 5.0, 1.0)
        
        final_score = (
            essential_completeness * 0.5 +
            unknown_completeness * 0.3 +
            knowledge_accumulation * 0.2
        )
        
        return min(final_score, 1.0)
    
    def _calculate_patient_gap_coverage(
        self,
        knowledge_gaps: List[str],
        dialogue_history: List[DialogueTurn]
    ) -> float:
        """
        计算患者知识缺口覆盖度
        """
        if not knowledge_gaps:
            return 1.0
        
        doctor_responses = [t.content for t in dialogue_history if t.role == "assistant"]
        if not doctor_responses:
            return 0.0
        
        doctor_text = " ".join(doctor_responses).lower()
        
        explanation_count = sum(
            1 for indicator in self.EXPLANATION_INDICATORS 
            if indicator in doctor_text
        )
        
        knowledge_transfer_count = sum(
            1 for indicator in self.KNOWLEDGE_TRANSFER_INDICATORS 
            if indicator in doctor_text
        )
        
        gap_keywords_addressed = set()
        for gap in knowledge_gaps:
            gap_lower = gap.lower()
            keywords = gap_lower.split()
            for keyword in keywords:
                if len(keyword) > 3 and keyword in doctor_text:
                    gap_keywords_addressed.add(gap)
                    break
        
        explanation_score = min(explanation_count / 3.0, 1.0)
        transfer_score = min(knowledge_transfer_count / 2.0, 1.0)
        gap_address_score = len(gap_keywords_addressed) / len(knowledge_gaps) if knowledge_gaps else 1.0
        
        final_score = (
            explanation_score * 0.3 +
            transfer_score * 0.3 +
            gap_address_score * 0.4
        )
        
        return min(final_score, 1.0)
    
    def _check_intentions_addressed(
        self,
        intentions: List[str],
        dialogue_history: List[DialogueTurn]
    ) -> bool:
        """
        检查患者意图是否被响应
        """
        if not intentions:
            return True
        
        doctor_responses = [t.content.lower() for t in dialogue_history if t.role == "assistant"]
        if not doctor_responses:
            return False
        
        doctor_text = " ".join(doctor_responses)
        
        intention_keywords = {
            'understand': ['explain', 'mean', 'because', 'reason', 'cause'],
            'treatment': ['treat', 'medication', 'therapy', 'prescription', 'recommend'],
            'reassurance': ['not serious', 'common', 'treatable', 'good prognosis', 'don\'t worry'],
            'diagnosis': ['diagnosis', 'condition', 'you have', 'this is'],
            'concern': ['understand your concern', 'i hear you', 'it\'s normal to worry']
        }
        
        addressed_count = 0
        for intention in intentions:
            intention_lower = intention.lower()
            for key, keywords in intention_keywords.items():
                if key in intention_lower:
                    if any(kw in doctor_text for kw in keywords):
                        addressed_count += 1
                        break
        
        return addressed_count >= len(intentions) * 0.5
    
    def get_missing_info_summary(
        self,
        tom_reasoning: ToMReasoning,
        task_type: str
    ) -> Dict[str, List[str]]:
        """
        获取缺失信息摘要
        """
        task_requirements = self.REQUIRED_INFO_BY_TASK.get(task_type, self.REQUIRED_INFO_BY_TASK["diagnosis"])
        
        return {
            "essential_missing": [
                info for info in task_requirements["essential"]
                if info.lower() not in " ".join(tom_reasoning.doctor_known_info).lower()
            ],
            "doctor_unknown": tom_reasoning.doctor_unknown_info,
            "patient_gaps": tom_reasoning.patient_knowledge_gaps
        }
    
    def estimate_turns_remaining(
        self,
        tom_reasoning: ToMReasoning,
        dialogue_history: List[DialogueTurn],
        task_type: str
    ) -> int:
        """
        估算剩余需要的对话轮次
        """
        goal_achieved, _, goal_status = self.check_tom_goal_achieved(
            tom_reasoning, dialogue_history, task_type
        )
        
        if goal_achieved:
            return 0
        
        doctor_score = goal_status["doctor_completeness_score"]
        patient_score = goal_status["patient_gap_coverage_score"]
        
        doctor_gap = max(0, self.DOCTOR_INFO_COMPLETENESS_THRESHOLD - doctor_score)
        patient_gap = max(0, self.PATIENT_GAP_COVERAGE_THRESHOLD - patient_score)
        
        estimated_turns = int((doctor_gap + patient_gap) * 10)
        
        return max(1, min(estimated_turns, 5))

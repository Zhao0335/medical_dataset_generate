#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据模型定义 - 严格落地论文1+2核心数据结构
论文1：双步骤ToM - 心智边界隔离、自适应DoM
论文2：动态时序ToM - 时序轨迹、因果触发链
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class DoMLevel(Enum):
    ZERO_ORDER = 0
    FIRST_ORDER = 1


class ToMErrorType(Enum):
    TYPE_A_OVER_MENTALIZING = "over_mentalizing"
    TYPE_B_UNDER_MENTALIZING = "under_mentalizing"
    TYPE_C_REASONING_ERROR = "reasoning_error"


class TaskType(Enum):
    DIAGNOSIS = "diagnosis"
    MEDRECON = "medrecon"
    PRESCRIPTIONS = "prescriptions"


@dataclass
class MentalState:
    beliefs: List[str] = field(default_factory=list)
    emotions: List[str] = field(default_factory=list)
    intentions: List[str] = field(default_factory=list)
    knowledge_gaps: List[str] = field(default_factory=list)
    
    def is_empty(self) -> bool:
        return not (self.beliefs or self.emotions or self.intentions or self.knowledge_gaps)
    
    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "beliefs": self.beliefs,
            "emotions": self.emotions,
            "intentions": self.intentions,
            "knowledge_gaps": self.knowledge_gaps
        }
    
    def copy(self) -> 'MentalState':
        return MentalState(
            beliefs=self.beliefs.copy(),
            emotions=self.emotions.copy(),
            intentions=self.intentions.copy(),
            knowledge_gaps=self.knowledge_gaps.copy()
        )


@dataclass
class CausalEvent:
    trigger_event: str = ""
    trigger_type: str = ""
    mental_state_before: MentalState = field(default_factory=MentalState)
    mental_state_after: MentalState = field(default_factory=MentalState)
    change_description: str = ""
    belief_changes: List[str] = field(default_factory=list)
    emotion_changes: List[str] = field(default_factory=list)
    intention_changes: List[str] = field(default_factory=list)
    knowledge_gap_changes: List[str] = field(default_factory=list)
    
    def is_valid(self) -> bool:
        return bool(self.trigger_event and self.change_description)


@dataclass
class TemporalChainLink:
    turn_number: int = 0
    timestamp: str = ""
    trigger_input: str = ""
    observation: str = ""
    inference: str = ""
    mental_state_delta: Dict[str, List[str]] = field(default_factory=dict)
    evidence_links: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_number": self.turn_number,
            "timestamp": self.timestamp,
            "trigger_input": self.trigger_input,
            "observation": self.observation,
            "inference": self.inference,
            "mental_state_delta": self.mental_state_delta,
            "evidence_links": self.evidence_links
        }


@dataclass
class TemporalMentalTrajectory:
    turn_number: int = 0
    timestamp: str = ""
    mental_state: MentalState = field(default_factory=MentalState)
    causal_event: Optional[CausalEvent] = None
    changes_from_previous: Dict[str, List[str]] = field(default_factory=dict)
    temporal_chain: List[TemporalChainLink] = field(default_factory=list)
    anchored_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_chain_summary(self) -> str:
        if not self.temporal_chain:
            return "No temporal chain recorded"
        return " -> ".join([link.inference for link in self.temporal_chain])


@dataclass
class ToMErrorRecord:
    error_type: ToMErrorType
    error_description: str
    detected_at_turn: int
    correction_applied: str
    corrected: bool = False
    original_value: Any = None
    corrected_value: Any = None


@dataclass
class MentalBoundary:
    doctor_known: List[str] = field(default_factory=list)
    doctor_unknown: List[str] = field(default_factory=list)
    patient_known: List[str] = field(default_factory=list)
    patient_knowledge_gaps: List[str] = field(default_factory=list)
    
    def validate_separation(self) -> List[str]:
        errors = []
        doctor_set = set(self.doctor_known)
        patient_set = set(self.patient_known)
        overlap = doctor_set & patient_set
        if overlap:
            errors.append(f"Boundary violation: Overlap detected: {overlap}")
        return errors
    
    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "doctor_known": self.doctor_known,
            "doctor_unknown": self.doctor_unknown,
            "patient_known": self.patient_known,
            "patient_knowledge_gaps": self.patient_knowledge_gaps
        }


@dataclass
class ToMReasoning:
    should_invoke_tom: bool = False
    dom_level: int = 0
    step1_decision_reason: str = ""
    mental_boundary: MentalBoundary = field(default_factory=MentalBoundary)
    patient_potential_intentions: List[str] = field(default_factory=list)
    patient_mental_state: MentalState = field(default_factory=MentalState)
    next_action_strategy: str = ""
    temporal_trajectory: TemporalMentalTrajectory = field(default_factory=TemporalMentalTrajectory)
    tom_errors_detected: List[ToMErrorRecord] = field(default_factory=list)
    temporal_chain_reasoning: List[TemporalChainLink] = field(default_factory=list)
    
    @property
    def doctor_known_info(self) -> List[str]:
        return self.mental_boundary.doctor_known
    
    @property
    def doctor_unknown_info(self) -> List[str]:
        return self.mental_boundary.doctor_unknown
    
    @property
    def patient_known_info(self) -> List[str]:
        return self.mental_boundary.patient_known
    
    @property
    def patient_knowledge_gaps(self) -> List[str]:
        return self.mental_boundary.patient_knowledge_gaps
    
    def has_valid_data(self) -> bool:
        return bool(
            self.mental_boundary.doctor_known or
            self.mental_boundary.doctor_unknown or
            self.patient_mental_state.beliefs or
            self.patient_mental_state.emotions or
            self.patient_mental_state.intentions or
            self.patient_mental_state.knowledge_gaps
        )


@dataclass
class DialogueTurn:
    content: str
    role: str
    turn_number: int = 0
    tom_reasoning: Optional[ToMReasoning] = None
    mental_state_at_turn: Optional[MentalState] = None


@dataclass
class TargetFormat:
    data_source: str
    topic: str
    department: str
    subdepartment: str
    disease: str
    prompt: List[Dict[str, Any]]
    ability: str
    reward_model: Dict[str, str]
    tom_annotations: List[Dict[str, Any]] = field(default_factory=list)

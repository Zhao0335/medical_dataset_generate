#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据模型定义
双步骤ToM - 心智边界隔离、自适应DoM
动态时序ToM - 时序轨迹、因果触发链
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class DoMLevel(Enum):
    """
    DoM等级枚举
    """
    ZERO_ORDER = 0
    FIRST_ORDER = 1


class ToMErrorType(Enum):
    """
    ToM错误类型枚举
    """
    TYPE_A_OVER_MENTALIZING = "over_mentalizing"
    TYPE_B_UNDER_MENTALIZING = "under_mentalizing"
    TYPE_C_REASONING_ERROR = "reasoning_error"


class TaskType(Enum):
    """
    任务类型枚举
    """
    DIAGNOSIS = "diagnosis"
    MEDRECON = "medrecon"
    PRESCRIPTIONS = "prescriptions"


@dataclass
class MentalState:
    """
    心智状态类
    属性：
    - beliefs: 信念列表
    - emotions: 情感列表
    - intentions: 意图列表
    - knowledge_gaps: 知识缺口列表
    """
    beliefs: List[str] = field(default_factory=list)
    emotions: List[str] = field(default_factory=list)
    intentions: List[str] = field(default_factory=list)
    knowledge_gaps: List[str] = field(default_factory=list)
    
    def is_empty(self) -> bool:
        return not (self.beliefs or self.emotions or self.intentions or self.knowledge_gaps)
    
    def to_dict(self) -> Dict[str, List[str]]:
        """
        将智状态转换为字典表示
        返回：
        - 包含信念、情感、意图和知识缺口的字典
        """
        return {
            "beliefs": self.beliefs,
            "emotions": self.emotions,
            "intentions": self.intentions,
            "knowledge_gaps": self.knowledge_gaps
        }
    
    def copy(self) -> 'MentalState':
        """
        深拷贝智状态
        返回：
        - 新的智状态实例
        """
        return MentalState(
            beliefs=self.beliefs.copy(),
            emotions=self.emotions.copy(),
            intentions=self.intentions.copy(),
            knowledge_gaps=self.knowledge_gaps.copy()
        )


@dataclass
class CausalEvent:
    """
    因果事件类
    
    属性：
    - trigger_event: 触发事件
    - trigger_type: 触发类型
    - mental_state_before: 事件前的心智状态
    - mental_state_after: 事件后的心智状态
    - change_description: 变化描述
    - belief_changes: 信念变化
    - emotion_changes: 情感变化
    - intention_changes: 意图变化
    - knowledge_gap_changes: 知识缺口变化
    """
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
        """
        检查因果事件是否有效
        
        返回：
        - bool: 是否有效
        """
        return bool(self.trigger_event and self.change_description)


@dataclass
class TemporalChainLink:
    """
    时序链链接类
    
    属性：
    - turn_number: 轮次编号
    - timestamp: 时间戳
    - trigger_input: 触发输入
    - observation: 观察内容
    - inference: 推理结果
    - mental_state_delta: 心智状态变化
    - evidence_links: 证据链接
    """
    turn_number: int = 0
    timestamp: str = ""
    trigger_input: str = ""
    observation: str = ""
    inference: str = ""
    mental_state_delta: Dict[str, List[str]] = field(default_factory=dict)
    evidence_links: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典表示
        
        返回：
        - Dict[str, Any]: 字典表示
        """
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
    """
    时序心智轨迹类
    
    属性：
    - turn_number: 轮次编号
    - timestamp: 时间戳
    - mental_state: 心智状态
    - causal_event: 因果事件
    - changes_from_previous: 与之前的变化
    - temporal_chain: 时序链
    - anchored_history: 锚定历史
    """
    turn_number: int = 0
    timestamp: str = ""
    mental_state: MentalState = field(default_factory=MentalState)
    causal_event: Optional[CausalEvent] = None
    changes_from_previous: Dict[str, List[str]] = field(default_factory=dict)
    temporal_chain: List[TemporalChainLink] = field(default_factory=list)
    anchored_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_chain_summary(self) -> str:
        """
        获取链摘要
        
        返回：
        - str: 链摘要
        """
        if not self.temporal_chain:
            return "No temporal chain recorded"
        return " -> ".join([link.inference for link in self.temporal_chain])


@dataclass
class ToMErrorRecord:
    """
    ToM 错误记录类
    
    属性：
    - error_type: 错误类型
    - error_description: 错误描述
    - detected_at_turn: 检测到错误的轮次
    - correction_applied: 应用的修正
    - corrected: 是否已修正
    - original_value: 原始值
    - corrected_value: 修正后的值
    """
    error_type: ToMErrorType
    error_description: str
    detected_at_turn: int
    correction_applied: str
    corrected: bool = False
    original_value: Any = None
    corrected_value: Any = None


@dataclass
class MentalBoundary:
    """
    心智边界类
    
    属性：
    - doctor_known: 医生已知信息
    - doctor_unknown: 医生未知信息
    - patient_known: 患者已知信息
    - patient_knowledge_gaps: 患者知识缺口
    """
    doctor_known: List[str] = field(default_factory=list)
    doctor_unknown: List[str] = field(default_factory=list)
    patient_known: List[str] = field(default_factory=list)
    patient_knowledge_gaps: List[str] = field(default_factory=list)
    
    def validate_separation(self) -> List[str]:
        """
        验证边界分离
        
        返回：
        - List[str]: 错误列表
        """
        errors = []
        doctor_set = set(self.doctor_known)
        patient_set = set(self.patient_known)
        overlap = doctor_set & patient_set
        if overlap:
            errors.append(f"Boundary violation: Overlap detected: {overlap}")
        return errors
    
    def to_dict(self) -> Dict[str, List[str]]:
        """
        转换为字典表示
        
        返回：
        - Dict[str, List[str]]: 字典表示
        """
        return {
            "doctor_known": self.doctor_known,
            "doctor_unknown": self.doctor_unknown,
            "patient_known": self.patient_known,
            "patient_knowledge_gaps": self.patient_knowledge_gaps
        }


@dataclass
class ToMReasoning:
    """
    ToM 推理结果类
    
    属性：
    - should_invoke_tom: 是否调用 ToM
    - dom_level: 心智化水平
    - step1_decision_reason: Step1 决策理由
    - mental_boundary: 心智边界
    - patient_potential_intentions: 患者潜在意图
    - patient_mental_state: 患者心智状态
    - next_action_strategy: 下一步行动策略
    - temporal_trajectory: 时序心智轨迹
    - tom_errors_detected: 检测到的 ToM 错误
    - temporal_chain_reasoning: 时序链推理
    """
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
        """
        获取医生已知信息
        
        返回：
        - List[str]: 医生已知信息
        """
        return self.mental_boundary.doctor_known
    
    @property
    def doctor_unknown_info(self) -> List[str]:
        """
        获取医生未知信息
        
        返回：
        - List[str]: 医生未知信息
        """
        return self.mental_boundary.doctor_unknown
    
    @property
    def patient_known_info(self) -> List[str]:
        """
        获取患者已知信息
        
        返回：
        - List[str]: 患者已知信息
        """
        return self.mental_boundary.patient_known
    
    @property
    def patient_knowledge_gaps(self) -> List[str]:
        """
        获取患者知识缺口
        
        返回：
        - List[str]: 患者知识缺口
        """
        return self.mental_boundary.patient_knowledge_gaps
    
    def has_valid_data(self) -> bool:
        """
        检查是否有有效数据
        
        返回：
        - bool: 是否有有效数据
        """
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
    """
    对话轮次类
    
    属性：
    - content: 内容
    - role: 角色
    - turn_number: 轮次编号
    - tom_reasoning: ToM 推理结果
    - mental_state_at_turn: 该轮次的心智状态
    """
    content: str
    role: str
    turn_number: int = 0
    tom_reasoning: Optional[ToMReasoning] = None
    mental_state_at_turn: Optional[MentalState] = None


@dataclass
class TargetFormat:
    """
    目标格式类
    
    属性：
    - data_source: 数据来源
    - topic: 主题
    - department: 科室
    - subdepartment: 子科室
    - disease: 疾病
    - prompt: 提示列表
    - ability: 能力
    - reward_model: 奖励模型
    - tom_annotations: ToM 标注列表
    """
    data_source: str
    topic: str
    department: str
    subdepartment: str
    disease: str
    prompt: List[Dict[str, Any]]
    ability: str
    reward_model: Dict[str, str]
    tom_annotations: List[Dict[str, Any]] = field(default_factory=list)

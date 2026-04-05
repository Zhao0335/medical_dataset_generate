#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具模块 - 公共方法
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict

from tom_models import DialogueTurn, ToMReasoning, MentalState, TemporalMentalTrajectory
from logger import get_logger

logger = get_logger()


def format_dialogue_history(dialogue_history: List[DialogueTurn]) -> str:
    formatted = []
    for turn in dialogue_history:
        role_label = "DOCTOR" if turn.role == "assistant" else "PATIENT"
        formatted.append(f"[Turn {turn.turn_number}] {role_label}: {turn.content}")
    return "\n".join(formatted)


def format_temporal_chain(chain: List, max_links: int = 5) -> str:
    formatted = []
    for link in chain[-max_links:]:
        formatted.append(
            f"Turn {link.turn_number}: {link.trigger_input}\n"
            f"  → Observation: {link.observation}\n"
            f"  → Inference: {link.inference}"
        )
    return "\n".join(formatted)


def validate_api_key(api_key: str) -> Tuple[bool, str]:
    if not api_key:
        return False, "API key is required but not provided"
    if len(api_key) < 10:
        return False, "API key appears to be invalid (too short)"
    return True, ""


def safe_json_loads(text: str, default: Any = None) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode error: {e}")
        return default


def truncate_text(text: str, max_length: int = 100) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def extract_json_from_response(response_text: str) -> Optional[Dict[str, Any]]:
    json_match = re.search(r'\{[\s\S]*\}', response_text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            return None
    return None


def build_tom_annotation(turn_index: int, turn: DialogueTurn) -> Dict[str, Any]:
    if not turn.tom_reasoning:
        return None
    
    tom = turn.tom_reasoning
    trajectory = tom.temporal_trajectory
    causal = trajectory.causal_event if trajectory and trajectory.causal_event else None
    
    return {
        "turn_index": turn_index,
        "turn_number": turn.turn_number,
        "step1_decision": {
            "should_invoke_tom": tom.should_invoke_tom,
            "dom_level": tom.dom_level,
            "decision_reason": tom.step1_decision_reason
        },
        "mental_boundary_separation": tom.mental_boundary.to_dict(),
        "patient_mental_state": tom.patient_mental_state.to_dict(),
        "patient_potential_intentions": tom.patient_potential_intentions,
        "temporal_trajectory": {
            "turn_number": trajectory.turn_number if trajectory else 0,
            "changes_from_previous": trajectory.changes_from_previous if trajectory else {},
            "causal_event": {
                "trigger": causal.trigger_event if causal else None,
                "trigger_type": causal.trigger_type if causal else None,
                "change_description": causal.change_description if causal else None,
                "belief_changes": causal.belief_changes if causal else [],
                "emotion_changes": causal.emotion_changes if causal else [],
                "intention_changes": causal.intention_changes if causal else []
            } if causal else None,
            "temporal_chain": [link.to_dict() for link in trajectory.temporal_chain] if trajectory else [],
            "anchored_history": trajectory.anchored_history if trajectory else []
        },
        "temporal_chain_reasoning": [link.to_dict() for link in tom.temporal_chain_reasoning],
        "tom_errors_detected": [
            {
                "error_type": e.error_type.value,
                "description": e.error_description,
                "correction": e.correction_applied,
                "corrected": e.corrected,
                "original_value": truncate_text(str(e.original_value)) if e.original_value else None,
                "corrected_value": truncate_text(str(e.corrected_value)) if e.corrected_value else None
            } for e in tom.tom_errors_detected
        ],
        "next_action_strategy": tom.next_action_strategy
    }


def safe_write_jsonl(file_path: str, data: List[Dict[str, Any]]) -> bool:
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        return True
    except Exception as e:
        logger.error(f"Failed to write JSONL file: {e}")
        return False


class APIError(Exception):
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


class ValidationError(Exception):
    pass


class ConfigurationError(Exception):
    pass

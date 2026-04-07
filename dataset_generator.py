#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医疗数据集生成器主类
整合所有ToM模块，生成完整的对话数据集
"""

import json
import os
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict
from contextlib import contextmanager

from tom_models import (
    ToMReasoning,
    MentalState,
    DialogueTurn,
    TargetFormat,
    TemporalMentalTrajectory,
    MentalBoundary,
    TaskType
)
from tom_reasoning import ToMReasoningModule
from patient_simulator import PatientMindSimulator
from tom_goal_checker import ToMGoalChecker
from config import config
from utils import (
    format_dialogue_history,
    format_temporal_chain,
    validate_api_key,
    build_tom_annotation,
    ConfigurationError,
    ValidationError
)
from logger import get_logger

logger = get_logger()


@contextmanager
def open_jsonl_files(output_dir: str, task_types: List[str]):
    files = {}
    try:
        os.makedirs(output_dir, exist_ok=True)
        for task in task_types:
            file_path = os.path.join(output_dir, f"{task}_tom_dataset.jsonl")
            files[task] = open(file_path, 'w', encoding='utf-8')
        yield files
    finally:
        for f in files.values():
            f.close()


class MedicalDatasetGenerator:
    
    def __init__(
        self,
        provider: str = "openai",
        api_key: str = None,
        base_url: str = None,
        model: str = "gpt-4",
        local_model_path: str = None,
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        self.config = Config.from_args(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            local_model_path=local_model_path,
            device=device
        )
        
        self.llm_provider = self.config.create_llm_provider()
        
        self.tom_module = ToMReasoningModule(self.llm_provider)
        self.patient_simulator = PatientMindSimulator(self.llm_provider)
        self.goal_checker = ToMGoalChecker()
    
    
    def generate_doctor_response_with_tom(
        self,
        context: Dict[str, Any],
        dialogue_history: List[DialogueTurn],
        tom_reasoning: ToMReasoning,
        task_type: str
    ) -> str:
        
        task_config = self.config.task_configs.get(task_type)
        if not task_config:
            raise ValidationError(f"Unknown task type: {task_type}")
        
        # 获取完整的 EHR 数据
        ehr_input = context.get("input_text", "")
        
        # 构建端到端的 ToM 驱动对话生成提示
        prompt = f"""{task_config.system_prompt}

You are a medical professional using Theory of Mind (ToM) to provide empathetic, patient-centered care.

=== PATIENT EHR DATA ===
{ehr_input}

=== CURRENT DIALOGUE ===
{format_dialogue_history(dialogue_history)}

=== YOUR ROLE ===
Based on the patient's EHR data and the dialogue history, you must:

1. ANALYZE PATIENT's MENTAL STATE:
   - Beliefs: What does the patient believe about their condition?
   - Emotions: What is the patient feeling right now?
   - Intentions: What does the patient want to achieve?
   - Knowledge Gaps: What does the patient not understand?

2. PERFORM TEMPORAL CHAIN REASONING:
   - Track how the patient's mental state has evolved
   - Identify causal events that changed their mental state
   - Understand the temporal progression of their concerns

3. GENERATE TO M-DRIVEN RESPONSE:
   - Address the patient's knowledge gaps with clear explanations
   - Respond to their emotions with empathy
   - Help them achieve their intentions
   - Gather any missing information needed for diagnosis/treatment
   - Make medically appropriate recommendations based on EHR data

4. MAINTAIN NATURAL DIALOGUE:
   - Speak in a natural, conversational tone
   - Avoid jargon and technical language
   - Show genuine concern and understanding
   - Adapt your response to the patient's emotional state

OUTPUT: Your response to the patient (natural, empathetic, ToM-driven)
Do NOT include meta-commentary or explanations of your reasoning.
"""
        
        try:
            response = self.llm_provider.generate_chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7  # 增加温度以获得更自然的响应
            )
            return response.content.strip()
        except Exception as e:
            logger.error(f"Doctor response generation error: {e}")
            # 移除硬编码的 fallback，直接返回一个通用的询问
            return "I'm sorry, I need to understand more about your situation. Could you tell me more about what you're experiencing?"
    

    

    
    def generate_dialogue_with_tom(
        self,
        ehr_data: Dict[str, Any],
        task_type: str,
        max_turns: int = None
    ) -> Tuple[List[DialogueTurn], List[ToMReasoning]]:
        
        if max_turns is None:
            max_turns = self.config.tom_thresholds.max_dialogue_turns
        
        dialogue = []
        tom_reasonings = []
        
        # 直接使用完整的 EHR 数据作为上下文
        context = {
            "ehr_data": ehr_data,
            "input_text": ehr_data.get("input", "")
        }
        
        previous_trajectory = None
        
        # 使用 LLM 生成初始医生开场
        ehr_input = ehr_data.get("input", "")
        initial_prompt = f"""You are a doctor starting a consultation with a patient. Based on the patient's EHR data, generate a natural, empathetic opening question.

=== PATIENT EHR DATA ===
{ehr_input}

OUTPUT: Your opening question to the patient (natural, empathetic, focused on the chief complaint)
"""
        
        try:
            response = self.llm_provider.generate_chat(
                messages=[{"role": "user", "content": initial_prompt}],
                max_tokens=200,
                temperature=0.7
            )
            initial_doctor_msg = response.content.strip()
        except Exception as e:
            logger.error(f"Initial doctor message generation error: {e}")
            # 简单的 fallback
            initial_doctor_msg = "Hello, I'm here to help you. Can you tell me what brings you in today?"
        
        dialogue.append(DialogueTurn(
            content=initial_doctor_msg,
            role="assistant",
            turn_number=0
        ))
        
        should_invoke, dom_level, decision_reason = self.tom_module.step1_tom_invocation_decision(
            context, dialogue, task_type
        )
        
        if should_invoke:
            tom_reasoning = self.tom_module.step2_mental_state_inference(
                context, dialogue, dom_level, task_type, previous_trajectory
            )
        else:
            tom_reasoning = ToMReasoning(
                should_invoke_tom=False,
                dom_level=0,
                step1_decision_reason=decision_reason,
                mental_boundary=MentalBoundary(
                    doctor_known=["patient is here for consultation"],
                    doctor_unknown=["symptoms", "history"],
                    patient_known=[],
                    patient_knowledge_gaps=["understanding of condition"]
                ),
                patient_mental_state=MentalState(
                    beliefs=["has health concern"],
                    emotions=["concern"],
                    intentions=["seek medical help"],
                    knowledge_gaps=["understanding of condition"]
                )
            )
        
        tom_reasonings.append(tom_reasoning)
        dialogue[-1].tom_reasoning = tom_reasoning
        dialogue[-1].mental_state_at_turn = tom_reasoning.patient_mental_state
        
        if tom_reasoning.temporal_trajectory:
            previous_trajectory = tom_reasoning.temporal_trajectory
        
        task_config = self.config.task_configs.get(task_type)
        required_info = task_config.required_info if task_config else []
        
        for turn_num in range(1, max_turns):
            # 生成患者响应
            patient_response = self.patient_simulator.generate_patient_response(
                tom_reasoning, context, dialogue, task_type, previous_trajectory
            )
            
            dialogue.append(DialogueTurn(
                content=patient_response,
                role="user",
                turn_number=turn_num * 2 - 1
            ))
            
            # Step1: ToM 调用决策
            should_invoke, dom_level, decision_reason = self.tom_module.step1_tom_invocation_decision(
                context, dialogue, task_type
            )
            
            # Step2: 心理状态推理
            if should_invoke:
                tom_reasoning = self.tom_module.step2_mental_state_inference(
                    context, dialogue, dom_level, task_type, previous_trajectory
                )
            else:
                tom_reasoning = ToMReasoning(
                    should_invoke_tom=False,
                    dom_level=0,
                    step1_decision_reason=decision_reason,
                    mental_boundary=MentalBoundary(),
                    patient_mental_state=previous_trajectory.mental_state.copy() if previous_trajectory and previous_trajectory.mental_state else MentalState()
                )
            
            tom_reasonings.append(tom_reasoning)
            
            if tom_reasoning.temporal_trajectory:
                previous_trajectory = tom_reasoning.temporal_trajectory
            
            # LLM 评估目标达成情况
            ehr_input = context.get("input_text", "")
            goal_prompt = f"""Evaluate if the medical consultation has achieved its goals:

=== PATIENT EHR DATA ===
{ehr_input}

=== DIALOGUE HISTORY ===
{format_dialogue_history(dialogue)}

=== PATIENT'S MENTAL STATE ===
Beliefs: {tom_reasoning.patient_mental_state.beliefs}
Emotions: {tom_reasoning.patient_mental_state.emotions}
Intentions: {tom_reasoning.patient_mental_state.intentions}
Knowledge Gaps: {tom_reasoning.patient_mental_state.knowledge_gaps}

=== TASK TYPE ===
{task_type}

Evaluate:
1. Has the doctor gathered sufficient information?
2. Have the patient's knowledge gaps been addressed?
3. Have the patient's intentions been fulfilled?
4. Is the patient's emotional state positive?
5. Should the consultation conclude?

OUTPUT:
{
    "goal_achieved": true/false,
    "reason": "detailed explanation",
    "goal_status": {
        "doctor_info_complete": true/false,
        "patient_gaps_covered": true/false,
        "intentions_fulfilled": true/false,
        "emotional_state_positive": true/false
    }
}
"""
            
            goal_achieved = False
            goal_status = {}
            
            try:
                goal_response = self.llm_provider.generate_chat(
                    messages=[{"role": "user", "content": goal_prompt}],
                    max_tokens=300,
                    temperature=0.3
                )
                goal_result = safe_json_loads(goal_response.content)
                if goal_result:
                    goal_achieved = goal_result.get("goal_achieved", False)
                    goal_status = goal_result.get("goal_status", {})
            except Exception as e:
                logger.error(f"Goal achievement evaluation error: {e}")
            
            if goal_achieved:
                # LLM 生成最终响应
                final_prompt = f"""Generate a final summary response for the medical consultation:

=== PATIENT EHR DATA ===
{ehr_input}

=== DIALOGUE HISTORY ===
{format_dialogue_history(dialogue)}

=== PATIENT'S MENTAL STATE ===
Beliefs: {tom_reasoning.patient_mental_state.beliefs}
Emotions: {tom_reasoning.patient_mental_state.emotions}
Intentions: {tom_reasoning.patient_mental_state.intentions}

=== GOAL STATUS ===
{json.dumps(goal_status, indent=2)}

OUTPUT: Your final summary and recommendations to the patient (natural, empathetic, comprehensive)
"""
                
                try:
                    final_response = self.llm_provider.generate_chat(
                        messages=[{"role": "user", "content": final_prompt}],
                        max_tokens=500,
                        temperature=0.5
                    )
                    doctor_response = final_response.content.strip()
                except Exception as e:
                    logger.error(f"Final response generation error: {e}")
                    doctor_response = "Thank you for sharing your concerns. Based on our discussion, I have a clear understanding of your situation and will provide you with the appropriate recommendations."
                
                dialogue.append(DialogueTurn(
                    content=doctor_response,
                    role="assistant",
                    turn_number=turn_num * 2,
                    tom_reasoning=tom_reasoning,
                    mental_state_at_turn=tom_reasoning.patient_mental_state
                ))
                break
            
            # 生成医生响应
            doctor_response = self.generate_doctor_response_with_tom(
                context, dialogue, tom_reasoning, task_type
            )
            
            dialogue.append(DialogueTurn(
                content=doctor_response,
                role="assistant",
                turn_number=turn_num * 2,
                tom_reasoning=tom_reasoning,
                mental_state_at_turn=tom_reasoning.patient_mental_state
            ))
        
        return dialogue, tom_reasonings
    

    
    def extract_disease_from_ehr(self, ehr_data: Dict) -> str:
        """
        使用 LLM 从 EHR 数据中提取疾病信息
        """
        input_text = ehr_data.get("input", "")
        
        prompt = f"""Extract the primary disease or chief complaint from the patient's EHR data:

=== EHR DATA ===
{input_text}

Output only the primary disease or chief complaint, without any additional text.
"""
        
        try:
            response = self.llm_provider.generate_chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.3
            )
            disease = response.content.strip()
            return disease if disease else "Unknown Condition"
        except Exception as e:
            logger.error(f"Disease extraction error: {e}")
            return "Unknown Condition"
    
    def determine_department(self, ehr_data: Dict, disease: str) -> Tuple[str, str]:
        """
        使用 LLM 基于疾病和 EHR 数据确定科室
        """
        input_text = ehr_data.get("input", "")
        
        prompt = f"""Based on the patient's disease and EHR data, determine the most appropriate medical department and subdepartment:

=== PATIENT DISEASE ===
{disease}

=== EHR DATA ===
{input_text}

Output format:
Department: [Main Department]
Subdepartment: [Subdepartment]

Example:
Department: Cardiology
Subdepartment: Cardiovascular Medicine
"""
        
        try:
            response = self.llm_provider.generate_chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.3
            )
            
            response_text = response.content.strip()
            department_match = re.search(r'Department:\s*(.+?)\n', response_text)
            subdepartment_match = re.search(r'Subdepartment:\s*(.+?)\n', response_text)
            
            if department_match and subdepartment_match:
                return (department_match.group(1).strip(), subdepartment_match.group(1).strip())
            else:
                return ("Internal Medicine", "General Internal Medicine")
        except Exception as e:
            logger.error(f"Department determination error: {e}")
            return ("Internal Medicine", "General Internal Medicine")
    
    def generate_single_sample(
        self,
        ehr_data: Dict,
        task_type: str
    ) -> Optional[TargetFormat]:
        
        # 直接使用完整的 EHR 数据，不再手动提取患者信息
        self.tom_module.trajectory_history = []
        self.patient_simulator.response_history = []
        
        dialogue, tom_reasonings = self.generate_dialogue_with_tom(ehr_data, task_type)
        
        if not dialogue:
            return None
        
        disease = self.extract_disease_from_ehr(ehr_data)
        department, subdepartment = self.determine_department(ehr_data, disease)
        
        prompt = []
        tom_annotations = []
        
        for i, turn in enumerate(dialogue):
            prompt.append({
                "content": turn.content,
                "role": turn.role
            })
            
            annotation = build_tom_annotation(i, turn)
            if annotation:
                tom_annotations.append(annotation)
        
        ability_map = {
            TaskType.DIAGNOSIS.value: "medical_diagnosis_with_tom",
            TaskType.MEDRECON.value: "medication_reconciliation_with_tom",
            TaskType.PRESCRIPTIONS.value: "prescription_writing_with_tom"
        }
        
        return TargetFormat(
            data_source="ehr_bench_tom_v2",
            topic=disease,
            department=department,
            subdepartment=subdepartment,
            disease=disease,
            prompt=prompt,
            ability=ability_map[task_type],
            reward_model={
                "ground_truth": disease,
                "style": "tom_dialogue_temporal"
            },
            tom_annotations=tom_annotations
        )
    
    def process_ehr_file(
        self,
        input_file: str,
        output_dir: str,
        task_types: List[str] = None,
        max_samples: int = None,
        delay: float = None
    ):
        
        if task_types is None:
            task_types = [TaskType.DIAGNOSIS.value, TaskType.MEDRECON.value, TaskType.PRESCRIPTIONS.value]
        
        if delay is None:
            delay = self.config.llm.delay
        
        with open_jsonl_files(output_dir, task_types) as output_files:
            with open(input_file, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    if max_samples and idx >= max_samples:
                        break
                    
                    try:
                        ehr_data = json.loads(line.strip())
                        logger.info(f"Processing sample {idx + 1}...")
                        logger.info(f"Chief Complaint: {self.extract_patient_info(ehr_data).get('chief_complaint', 'Unknown')}")
                        
                        for task_type in task_types:
                            logger.info(f"[{task_type.upper()}] Generating ToM-based dialogue...")
                            sample = self.generate_single_sample(ehr_data, task_type)
                            
                            if sample:
                                output_files[task_type].write(
                                    json.dumps(asdict(sample), ensure_ascii=False) + '\n'
                                )
                                logger.info(f"[{task_type.upper()}] Generated {len(sample.prompt)} dialogue turns")
                                logger.info(f"[{task_type.upper()}] ToM annotations: {len(sample.tom_annotations)}")
                                
                                if sample.tom_annotations:
                                    errors_count = sum(
                                        len(ann.get('tom_errors_detected', [])) 
                                        for ann in sample.tom_annotations
                                    )
                                    logger.info(f"[{task_type.upper()}] ToM errors detected & corrected: {errors_count}")
                                    
                                    dom_levels = [ann.get('step1_decision', {}).get('dom_level', 0) 
                                                for ann in sample.tom_annotations]
                                    logger.info(f"[{task_type.upper()}] DoM levels used: {set(dom_levels)}")
                            
                            time.sleep(delay)
                            
                    except Exception as e:
                        logger.error(f"Processing sample {idx}: {e}")
                        continue
        
        logger.info("ToM-based dataset generation completed!")
        logger.info(f"Output files saved to: {output_dir}")
        for task in task_types:
            logger.info(f"  - {task}_tom_dataset.jsonl")
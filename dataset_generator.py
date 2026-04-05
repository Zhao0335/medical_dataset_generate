#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医疗数据集生成器主类 - 严格落地论文1+2核心方案
整合所有ToM模块，生成完整的对话数据集
"""

import json
import os
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict

from openai import OpenAI

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


class MedicalDatasetGenerator:
    
    def __init__(self, api_key: str = None, base_url: str = None, model: str = "gpt-4"):
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL")
        )
        self.model = model
        
        self.tom_module = ToMReasoningModule(self.client, model)
        self.patient_simulator = PatientMindSimulator(self.client, model)
        self.goal_checker = ToMGoalChecker()
        
        self.task_configs = {
            TaskType.DIAGNOSIS.value: {
                "system": "You are an experienced doctor using Theory of Mind for diagnosis.",
                "required_info": ["symptoms", "duration", "severity", "medical history", "current medications"]
            },
            TaskType.MEDRECON.value: {
                "system": "You are a clinical pharmacist using Theory of Mind for medication reconciliation.",
                "required_info": ["current medications", "dosages", "frequency", "adherence", "side effects"]
            },
            TaskType.PRESCRIPTIONS.value: {
                "system": "You are a physician using Theory of Mind for prescription writing.",
                "required_info": ["diagnosis", "allergies", "current medications", "patient preferences"]
            }
        }
    
    def extract_patient_info(self, ehr_data: Dict) -> Dict[str, Any]:
        patient_info = {
            "demographics": {},
            "chief_complaint": "",
            "vital_signs": {},
            "lab_results": [],
            "medications": [],
            "medical_history": "",
            "allergies": []
        }
        
        input_text = ehr_data.get("input", "")
        
        demographics_match = re.search(r'## Patient Demographics \[None\](.*?)(?=##|\Z)', input_text, re.DOTALL)
        if demographics_match:
            demo_text = demographics_match.group(1)
            age_match = re.search(r'Anchor_Age:\s*(\d+)', demo_text)
            gender_match = re.search(r'Gender:\s*(\w+)', demo_text)
            if age_match:
                patient_info["demographics"]["age"] = int(age_match.group(1))
            if gender_match:
                patient_info["demographics"]["gender"] = gender_match.group(1)
        
        triage_match = re.search(r'## Triage \[.*?\](.*?)(?=##|\Z)', input_text, re.DOTALL)
        if triage_match:
            triage_text = triage_match.group(1)
            chief_match = re.search(r'Chiefcomplaint:\s*(.+?)(?:\n|$)', triage_text)
            if chief_match:
                patient_info["chief_complaint"] = chief_match.group(1).strip()
            
            vs_fields = ['Temperature', 'Heartrate', 'Resprate', 'O2Sat', 'Sbp', 'Dbp', 'Pain']
            for field in vs_fields:
                match = re.search(rf'{field}:\s*([\d.]+|nan)', triage_text)
                if match and match.group(1) != 'nan':
                    patient_info["vital_signs"][field.lower()] = float(match.group(1))
        
        discharge_match = re.search(r'## Discharge \[.*?\](.*?)(?=##|\Z)', input_text, re.DOTALL)
        if discharge_match:
            discharge_text = discharge_match.group(1)
            
            allergies_match = re.search(r'Allergies:\s*\n(.+?)(?=\n\n|\nAttending:)', discharge_text, re.DOTALL)
            if allergies_match:
                allergies_text = allergies_match.group(1).strip()
                patient_info["allergies"] = [a.strip() for a in allergies_text.split('/') if a.strip()]
            
            pmh_match = re.search(r'Past Medical History:\s*(.+?)(?=\n\n|\nSocial History:)', discharge_text, re.DOTALL)
            if pmh_match:
                patient_info["medical_history"] = pmh_match.group(1).strip()[:500]
        
        medrecon_match = re.search(r'## Medrecon \[.*?\](.*?)(?=##|\Z)', input_text, re.DOTALL)
        if medrecon_match:
            medrecon_text = medrecon_match.group(1)
            med_matches = re.findall(r'\|\s*([A-Za-z0-9\s\-\[\]]+?)\s*\|', medrecon_text)
            patient_info["medications"] = [m.strip() for m in med_matches 
                                           if m.strip() and m.strip() not in ['Name', 'Atc Type']][:10]
        
        return patient_info
    
    def generate_doctor_response_with_tom(
        self,
        context: Dict[str, Any],
        dialogue_history: List[DialogueTurn],
        tom_reasoning: ToMReasoning,
        task_type: str
    ) -> str:
        """
        论文要求：医生回复必须完全基于ToM推理结果
        禁止ToM推理与回复脱节
        """
        
        config = self.task_configs.get(task_type, self.task_configs[TaskType.DIAGNOSIS.value])
        
        if not tom_reasoning.has_valid_data():
            print("[WARNING] ToM reasoning has no valid data, generating fallback response")
            return self._generate_fallback_doctor_response(dialogue_history, task_type)
        
        error_corrections_summary = ""
        if tom_reasoning.tom_errors_detected:
            error_corrections_summary = f"""
ToM ERROR CORRECTIONS APPLIED:
{chr(10).join([f"- {e.error_type.value}: {e.correction_applied}" for e in tom_reasoning.tom_errors_detected])}
"""
        
        temporal_chain_summary = ""
        if tom_reasoning.temporal_chain_reasoning:
            temporal_chain_summary = f"""
TEMPORAL CHAIN REASONING:
{self._format_temporal_chain(tom_reasoning.temporal_chain_reasoning)}
"""
        
        prompt = f"""{config['system']}

=== ToM REASONING RESULTS (MUST DRIVE YOUR RESPONSE) ===
Step1 Decision: {"ToM invoked" if tom_reasoning.should_invoke_tom else "ToM not needed"}
DoM Level: {tom_reasoning.dom_level}
Reason: {tom_reasoning.step1_decision_reason}

=== MENTAL BOUNDARY SEPARATION (Strict Isolation) ===
DOCTOR's Known Info: {tom_reasoning.mental_boundary.doctor_known}
DOCTOR's Unknown Info: {tom_reasoning.mental_boundary.doctor_unknown}
PATIENT's Known Info: {tom_reasoning.mental_boundary.patient_known}
PATIENT's Knowledge Gaps: {tom_reasoning.mental_boundary.patient_knowledge_gaps}

=== PATIENT's MENTAL STATE ===
Beliefs: {tom_reasoning.patient_mental_state.beliefs}
Emotions: {tom_reasoning.patient_mental_state.emotions}
Intentions: {tom_reasoning.patient_mental_state.intentions}
Knowledge Gaps: {tom_reasoning.patient_mental_state.knowledge_gaps}

=== TEMPORAL TRAJECTORY ===
Current Turn: {tom_reasoning.temporal_trajectory.turn_number if tom_reasoning.temporal_trajectory else 'N/A'}
Changes: {tom_reasoning.temporal_trajectory.changes_from_previous if tom_reasoning.temporal_trajectory else {}}
Causal Trigger: {tom_reasoning.temporal_trajectory.causal_event.trigger_event if tom_reasoning.temporal_trajectory and tom_reasoning.temporal_trajectory.causal_event else 'N/A'}
{temporal_chain_summary}
{error_corrections_summary}
=== NEXT ACTION STRATEGY ===
{tom_reasoning.next_action_strategy}

=== DIALOGUE HISTORY ===
{self._format_dialogue_history(dialogue_history)}

=== CRITICAL INSTRUCTIONS ===
Your response MUST be DIRECTLY DRIVEN by the ToM analysis above:

1. ADDRESS PATIENT's KNOWLEDGE GAPS:
   - If patient has knowledge gaps, explain clearly
   - Use simple language, avoid jargon
   - Check for understanding

2. RESPOND TO PATIENT's EMOTIONS:
   - Acknowledge emotions: {tom_reasoning.patient_mental_state.emotions}
   - Show empathy appropriately
   - Provide reassurance if worried/anxious

3. PURSUE PATIENT's INTENTIONS:
   - Help patient achieve: {tom_reasoning.patient_mental_state.intentions}
   - Address their goals

4. GATHER MISSING INFO:
   - Still need to know: {tom_reasoning.mental_boundary.doctor_unknown}
   - Ask targeted questions

5. FOLLOW NEXT ACTION STRATEGY:
   - {tom_reasoning.next_action_strategy}

OUTPUT: Your response to the patient (natural, empathetic, ToM-driven)
Do NOT include meta-commentary or explanations of your reasoning.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.5
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ERROR] Doctor response generation error: {e}")
            return self._generate_fallback_doctor_response(dialogue_history, task_type)
    
    def _generate_fallback_doctor_response(
        self,
        dialogue_history: List[DialogueTurn],
        task_type: str
    ) -> str:
        """
        生成后备医生回复
        """
        last_patient_msg = ""
        for turn in reversed(dialogue_history):
            if turn.role == "user":
                last_patient_msg = turn.content
                break
        
        if task_type == TaskType.DIAGNOSIS.value:
            if not last_patient_msg:
                return "Hello, I understand you're here for a consultation. Can you tell me what brings you in today?"
            return "Thank you for sharing that. Can you tell me more about when these symptoms started and how severe they are?"
        elif task_type == TaskType.MEDRECON.value:
            return "Let's review your current medications. Can you tell me what medications you're currently taking and how you're taking them?"
        else:
            return "Based on what we've discussed, let me explain the treatment options available to you."
    
    def _format_temporal_chain(self, chain: List) -> str:
        formatted = []
        for link in chain[-5:]:
            formatted.append(
                f"Turn {link.turn_number}: {link.trigger_input}\n"
                f"  → Observation: {link.observation}\n"
                f"  → Inference: {link.inference}"
            )
        return "\n".join(formatted)
    
    def generate_dialogue_with_tom(
        self,
        patient_info: Dict[str, Any],
        task_type: str,
        max_turns: int = 12
    ) -> Tuple[List[DialogueTurn], List[ToMReasoning]]:
        """
        论文核心：完整的双步骤ToM对话生成流程
        """
        dialogue = []
        tom_reasonings = []
        
        context = {
            "patient_info": patient_info,
            "chief_complaint": patient_info.get("chief_complaint", "Unknown")
        }
        
        previous_trajectory = None
        
        initial_doctor_msg = f"Hello, I understand you're here because of {patient_info.get('chief_complaint', 'some health concerns')}. Can you tell me more about what's been going on?"
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
        
        config = self.task_configs.get(task_type, self.task_configs[TaskType.DIAGNOSIS.value])
        required_info = config.get("required_info", [])
        
        for turn_num in range(1, max_turns):
            patient_response = self.patient_simulator.generate_patient_response(
                tom_reasoning, context, dialogue, task_type, previous_trajectory
            )
            dialogue.append(DialogueTurn(
                content=patient_response,
                role="user",
                turn_number=turn_num * 2 - 1
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
                    mental_boundary=MentalBoundary(),
                    patient_mental_state=previous_trajectory.mental_state.copy() if previous_trajectory and previous_trajectory.mental_state else MentalState()
                )
            
            tom_reasonings.append(tom_reasoning)
            
            if tom_reasoning.temporal_trajectory:
                previous_trajectory = tom_reasoning.temporal_trajectory
            
            goal_achieved, reason, goal_status = self.goal_checker.check_tom_goal_achieved(
                tom_reasoning, dialogue, task_type, required_info
            )
            
            if goal_achieved:
                doctor_response = self._generate_final_response(
                    context, dialogue, tom_reasoning, task_type, goal_status
                )
                dialogue.append(DialogueTurn(
                    content=doctor_response,
                    role="assistant",
                    turn_number=turn_num * 2,
                    tom_reasoning=tom_reasoning,
                    mental_state_at_turn=tom_reasoning.patient_mental_state
                ))
                break
            
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
    
    def _generate_final_response(
        self,
        context: Dict[str, Any],
        dialogue_history: List[DialogueTurn],
        tom_reasoning: ToMReasoning,
        task_type: str,
        goal_status: Dict[str, Any]
    ) -> str:
        """
        生成最终回复，确保患者知识缺口被覆盖
        """
        
        final_prompts = {
            TaskType.DIAGNOSIS.value: "Based on our discussion and the information you've provided, here is my diagnosis and recommendation...",
            TaskType.MEDRECON.value: "After reviewing your medications and discussing your concerns, here are my recommendations...",
            TaskType.PRESCRIPTIONS.value: "Based on your diagnosis and our discussion, here are the prescriptions I recommend..."
        }
        
        prompt = f"""{self.task_configs[task_type]['system']}

=== FINAL ToM SUMMARY ===
Doctor's Known Info: {tom_reasoning.mental_boundary.doctor_known}
Patient's Knowledge Gaps Addressed: {tom_reasoning.mental_boundary.patient_knowledge_gaps}
Patient's Intentions Fulfilled: {tom_reasoning.patient_potential_intentions}
Patient's Final Mental State:
- Beliefs: {tom_reasoning.patient_mental_state.beliefs}
- Emotions: {tom_reasoning.patient_mental_state.emotions}
- Intentions: {tom_reasoning.patient_mental_state.intentions}

=== GOAL ACHIEVEMENT STATUS ===
Doctor Info Complete: {goal_status.get('doctor_info_complete', False)} ({goal_status.get('doctor_completeness_score', 0):.0%})
Patient Gaps Covered: {goal_status.get('patient_gaps_covered', False)} ({goal_status.get('patient_gap_coverage_score', 0):.0%})

=== DIALOGUE HISTORY ===
{self._format_dialogue_history(dialogue_history)}

Generate a final response that:
1. Provides clear diagnosis/medication reconciliation/prescription
2. ADDRESSES all patient knowledge gaps identified
3. RESPONDS to patient's emotions and intentions
4. Shows empathy and ensures patient understanding
5. Provides clear next steps

Start with: {final_prompts.get(task_type, "Based on our discussion...")}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.5
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return final_prompts.get(task_type, "Thank you for the consultation.")
    
    def _format_dialogue_history(self, dialogue_history: List[DialogueTurn]) -> str:
        formatted = []
        for turn in dialogue_history:
            formatted.append(f"[Turn {turn.turn_number}] {turn.role.upper()}: {turn.content}")
        return "\n".join(formatted)
    
    def extract_disease_from_ehr(self, ehr_data: Dict) -> str:
        input_text = ehr_data.get("input", "")
        
        discharge_dx_match = re.search(
            r'Discharge Diagnosis:\s*Primary:\s*(.+?)(?:\n\n|\nSecondary:|\Z)', 
            input_text, re.DOTALL
        )
        if discharge_dx_match:
            diseases = discharge_dx_match.group(1).strip()
            return diseases.split('\n')[0].strip()
        
        chief_match = re.search(r'Chiefcomplaint:\s*(.+?)(?:\n|$)', input_text)
        if chief_match:
            return chief_match.group(1).strip()
        
        return "Unknown Condition"
    
    def determine_department(self, ehr_data: Dict, disease: str) -> Tuple[str, str]:
        input_text = ehr_data.get("input", "")
        
        service_match = re.search(r'Service:\s*(\w+)', input_text)
        if service_match:
            service = service_match.group(1)
            department_map = {
                "MEDICINE": ("Internal Medicine", "General Internal Medicine"),
                "SURGERY": ("Surgery", "General Surgery"),
                "ORTHOPAEDICS": ("Orthopedics", "Orthopedic Surgery"),
                "NEUROSURGERY": ("Neurosurgery", "Neurological Surgery"),
                "UROLOGY": ("Urology", "Urological Surgery"),
                "PLASTIC": ("Plastic Surgery", "Reconstructive Surgery")
            }
            return department_map.get(service, ("General Medicine", "General Practice"))
        
        disease_lower = disease.lower()
        if any(kw in disease_lower for kw in ['heart', 'cardiac', 'chest pain']):
            return ("Cardiology", "Cardiovascular Medicine")
        elif any(kw in disease_lower for kw in ['stomach', 'gastro', 'abdom']):
            return ("Gastroenterology", "Digestive Diseases")
        elif any(kw in disease_lower for kw in ['brain', 'neuro', 'headache']):
            return ("Neurology", "Neurological Sciences")
        elif any(kw in disease_lower for kw in ['respiratory', 'lung', 'breath']):
            return ("Pulmonology", "Respiratory Medicine")
        
        return ("Internal Medicine", "General Internal Medicine")
    
    def generate_single_sample(
        self,
        ehr_data: Dict,
        task_type: str
    ) -> Optional[TargetFormat]:
        
        patient_info = self.extract_patient_info(ehr_data)
        
        if task_type == TaskType.PRESCRIPTIONS.value:
            patient_info["diagnosis"] = self.extract_disease_from_ehr(ehr_data)
        
        self.tom_module.trajectory_history = []
        self.patient_simulator.response_history = []
        
        dialogue, tom_reasonings = self.generate_dialogue_with_tom(patient_info, task_type)
        
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
            
            if turn.tom_reasoning:
                tom_annotations.append({
                    "turn_index": i,
                    "turn_number": turn.turn_number,
                    "step1_decision": {
                        "should_invoke_tom": turn.tom_reasoning.should_invoke_tom,
                        "dom_level": turn.tom_reasoning.dom_level,
                        "decision_reason": turn.tom_reasoning.step1_decision_reason
                    },
                    "mental_boundary_separation": turn.tom_reasoning.mental_boundary.to_dict(),
                    "patient_mental_state": turn.tom_reasoning.patient_mental_state.to_dict(),
                    "patient_potential_intentions": turn.tom_reasoning.patient_potential_intentions,
                    "temporal_trajectory": {
                        "turn_number": turn.tom_reasoning.temporal_trajectory.turn_number if turn.tom_reasoning.temporal_trajectory else 0,
                        "changes_from_previous": turn.tom_reasoning.temporal_trajectory.changes_from_previous if turn.tom_reasoning.temporal_trajectory else {},
                        "causal_event": {
                            "trigger": turn.tom_reasoning.temporal_trajectory.causal_event.trigger_event if turn.tom_reasoning.temporal_trajectory and turn.tom_reasoning.temporal_trajectory.causal_event else None,
                            "trigger_type": turn.tom_reasoning.temporal_trajectory.causal_event.trigger_type if turn.tom_reasoning.temporal_trajectory and turn.tom_reasoning.temporal_trajectory.causal_event else None,
                            "change_description": turn.tom_reasoning.temporal_trajectory.causal_event.change_description if turn.tom_reasoning.temporal_trajectory and turn.tom_reasoning.temporal_trajectory.causal_event else None,
                            "belief_changes": turn.tom_reasoning.temporal_trajectory.causal_event.belief_changes if turn.tom_reasoning.temporal_trajectory and turn.tom_reasoning.temporal_trajectory.causal_event else [],
                            "emotion_changes": turn.tom_reasoning.temporal_trajectory.causal_event.emotion_changes if turn.tom_reasoning.temporal_trajectory and turn.tom_reasoning.temporal_trajectory.causal_event else [],
                            "intention_changes": turn.tom_reasoning.temporal_trajectory.causal_event.intention_changes if turn.tom_reasoning.temporal_trajectory and turn.tom_reasoning.temporal_trajectory.causal_event else []
                        } if turn.tom_reasoning.temporal_trajectory and turn.tom_reasoning.temporal_trajectory.causal_event else None,
                        "temporal_chain": [link.to_dict() for link in turn.tom_reasoning.temporal_trajectory.temporal_chain] if turn.tom_reasoning.temporal_trajectory else [],
                        "anchored_history": turn.tom_reasoning.temporal_trajectory.anchored_history if turn.tom_reasoning.temporal_trajectory else []
                    },
                    "temporal_chain_reasoning": [link.to_dict() for link in turn.tom_reasoning.temporal_chain_reasoning],
                    "tom_errors_detected": [
                        {
                            "error_type": e.error_type.value,
                            "description": e.error_description,
                            "correction": e.correction_applied,
                            "corrected": e.corrected,
                            "original_value": str(e.original_value)[:100] if e.original_value else None,
                            "corrected_value": str(e.corrected_value)[:100] if e.corrected_value else None
                        } for e in turn.tom_reasoning.tom_errors_detected
                    ],
                    "next_action_strategy": turn.tom_reasoning.next_action_strategy
                })
        
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
        delay: float = 2.0
    ):
        
        if task_types is None:
            task_types = [TaskType.DIAGNOSIS.value, TaskType.MEDRECON.value, TaskType.PRESCRIPTIONS.value]
        
        os.makedirs(output_dir, exist_ok=True)
        
        output_files = {
            task: open(os.path.join(output_dir, f"{task}_tom_dataset.jsonl"), 'w', encoding='utf-8')
            for task in task_types
        }
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if max_samples and idx >= max_samples:
                    break
                
                try:
                    ehr_data = json.loads(line.strip())
                    print(f"\n{'='*60}")
                    print(f"Processing sample {idx + 1}...")
                    print(f"Chief Complaint: {self.extract_patient_info(ehr_data).get('chief_complaint', 'Unknown')}")
                    
                    for task_type in task_types:
                        print(f"\n  [{task_type.upper()}] Generating ToM-based dialogue...")
                        sample = self.generate_single_sample(ehr_data, task_type)
                        
                        if sample:
                            output_files[task_type].write(
                                json.dumps(asdict(sample), ensure_ascii=False) + '\n'
                            )
                            print(f"  [{task_type.upper()}] Generated {len(sample.prompt)} dialogue turns")
                            print(f"  [{task_type.upper()}] ToM annotations: {len(sample.tom_annotations)}")
                            
                            if sample.tom_annotations:
                                errors_count = sum(
                                    len(ann.get('tom_errors_detected', [])) 
                                    for ann in sample.tom_annotations
                                )
                                print(f"  [{task_type.upper()}] ToM errors detected & corrected: {errors_count}")
                                
                                dom_levels = [ann.get('step1_decision', {}).get('dom_level', 0) 
                                            for ann in sample.tom_annotations]
                                print(f"  [{task_type.upper()}] DoM levels used: {set(dom_levels)}")
                        
                        time.sleep(delay)
                        
                except Exception as e:
                    print(f"[ERROR] Processing sample {idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        for f in output_files.values():
            f.close()
        
        print(f"\n{'='*60}")
        print("ToM-based dataset generation completed!")
        print(f"Output files saved to: {output_dir}")
        for task in task_types:
            print(f"  - {task}_tom_dataset.jsonl")

# 医疗干预智能体数据集生成工具 - ToM深度实现

基于Theory of Mind（心智理论）深度实现的医疗对话数据集生成工具，**严格落地两篇ToM论文核心方案**。

## 核心特性

### 论文1：双步骤ToM（Two-Step ToM）

**Step 1: ToM调用自主决策**
- 模型自主判断是否需要调用ToM推理（可返回`should_invoke_tom=False`）
- **绝不强制永远调用**
- 自适应确定心理化深度（DoM Level: 0阶/1阶，医疗合作场景禁止高阶）

**Step 2: 心理状态推理**
- **严格心智边界隔离**：
  - 医生已知信息 vs 医生未知信息
  - 患者已知信息 vs 患者知识缺口
- 患者潜在意图推理
- 患者心理状态（信念、情绪、意图、知识缺口）
- 下一步行动策略

**3类ToM错误实时检测+自动修正**：
- **TypeA（过度心智化）**：禁止对患者简单提问做复杂意图猜测
- **TypeB（心智不足）**：必须识别患者回避、顾虑、知识缺口等隐性心理
- **TypeC（推理错误）**：校验心智推理与对话上下文一致性

### 论文2：动态时序ToM（Dynamic Temporal ToM）

- **时序心理轨迹**：按对话轮次记录信念→情绪→意图的连续变化，形成时间线
- **因果触发链**：强制记录"什么事件→导致患者心理改变"
- **时序链式推理**：废弃普通CoT，按对话时间线连续推导，联动历史所有心理状态
- **解决中间丢失问题**：每轮强制锚定历史心理状态，禁止遗忘中间轮次信息

### 患者心智模拟器

回复完全由动态时序心理状态驱动：
- 反映患者当前心理状态（信念、情绪、意图）
- 揭示与知识缺口一致的信息
- 展示适当的情感反应
- 自然追求患者意图
- 维持与前一心理状态的连续性
- **禁止生成通用静态回复**

### ToM目标达成判断

对话终止条件（废除关键词/最大轮次终止）：
- ✅ 医生信息补齐（doctor_unknown_info收集完整度≥80%）
- ✅ 患者知识缺口覆盖（patient_knowledge_gaps解决度≥70%）
- ✅ 安全限制：最大对话长度15轮

## 项目结构

```
d:\4github\medical_dataset_generate/
├── tom_models.py           # 数据模型定义（MentalState, ToMReasoning, TemporalMentalTrajectory等）
├── tom_error_detector.py   # ToM错误检测器（TypeA/B/C三类错误检测与修正）
├── tom_reasoning.py        # ToM推理模块（双步骤决策、自适应DoM、时序链式推理）
├── patient_simulator.py    # 患者心智模拟器（动态时序心理状态驱动）
├── tom_goal_checker.py     # ToM目标检查器（信息补齐+知识缺口覆盖终止条件）
├── dataset_generator.py    # 数据集生成器主类
└── main.py                 # 主入口
```

## 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                    MedicalDatasetGenerator                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │ ToMReasoningModule │  │PatientMindSimulator│  │ToMGoalChecker│ │
│  │  - Step1: 自主决策  │  │  - 时序心理驱动   │  │  - 目标判断  │ │
│  │  - Step2: 状态推理  │  │  - 自然回复生成   │  │  - 终止条件  │ │
│  │  - 自适应DoM       │  │  - 情绪/意图反映  │  │              │ │
│  │  - 时序链式推理     │  │  - 禁止通用回复   │  │              │ │
│  └─────────┬─────────┘  └──────────────────┘  └──────────────┘ │
│            │                                                    │
│            ▼                                                    │
│  ┌───────────────────┐                                         │
│  │ ToMErrorDetector  │                                         │
│  │  - TypeA: 过度心智化检测+修正                                │
│  │  - TypeB: 心智不足检测+修正                                  │
│  │  - TypeC: 推理错误检测+修正                                  │
│  └───────────────────┘                                         │
└─────────────────────────────────────────────────────────────────┘
```

## 数据结构

### 核心数据模型

```python
@dataclass
class MentalState:
    beliefs: List[str]          # 患者信念
    emotions: List[str]         # 患者情绪
    intentions: List[str]       # 患者意图
    knowledge_gaps: List[str]   # 患者知识缺口

@dataclass
class CausalEvent:
    trigger_event: str          # 触发事件
    trigger_type: str           # 触发类型
    mental_state_before: MentalState
    mental_state_after: MentalState
    belief_changes: List[str]   # 信念变化
    emotion_changes: List[str]  # 情绪变化
    intention_changes: List[str] # 意图变化
    change_description: str     # 变化描述

@dataclass
class TemporalChainLink:
    turn_number: int            # 轮次号
    trigger_input: str          # 触发输入
    observation: str            # 观察
    inference: str              # 推断
    mental_state_delta: Dict    # 心理状态变化
    evidence_links: List[str]   # 证据链接

@dataclass
class TemporalMentalTrajectory:
    turn_number: int            # 轮次号
    mental_state: MentalState   # 当前心理状态
    causal_event: CausalEvent   # 因果事件
    temporal_chain: List[TemporalChainLink]  # 时序链
    anchored_history: List[Dict] # 锚定历史（解决中间丢失）

@dataclass
class MentalBoundary:
    doctor_known: List[str]     # 医生已知
    doctor_unknown: List[str]   # 医生未知
    patient_known: List[str]    # 患者已知
    patient_knowledge_gaps: List[str]  # 患者知识缺口

@dataclass
class ToMReasoning:
    should_invoke_tom: bool     # Step1决策：是否调用ToM
    dom_level: int              # 心理化深度 (0/1)
    step1_decision_reason: str  # Step1决策原因
    mental_boundary: MentalBoundary  # 心智边界
    patient_mental_state: MentalState
    temporal_trajectory: TemporalMentalTrajectory
    temporal_chain_reasoning: List[TemporalChainLink]
    tom_errors_detected: List[ToMErrorRecord]
    next_action_strategy: str
```

## 使用方法

### 安装依赖

```bash
pip install openai
```

### 基本用法

```bash
python main.py `
    --input ehr_bench_decision_making.jsonl `
    --output ./output_tom `
    --api_key YOUR_API_KEY `
    --base_url YOUR_API_BASE_URL `
    --model gpt-4
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 输入EHR JSONL文件路径 | `ehr_bench_decision_making.jsonl` |
| `--output` | 输出目录 | `./output_tom` |
| `--tasks` | 任务类型列表 | `diagnosis medrecon prescriptions` |
| `--max_samples` | 最大处理样本数 | 全部处理 |
| `--api_key` | OpenAI API密钥 | 从环境变量读取 |
| `--base_url` | API基础URL | 从环境变量读取 |
| `--model` | 使用的LLM模型 | `gpt-4` |
| `--delay` | API调用间隔（秒） | `2.0` |

### 示例命令

```bash
# 测试运行（处理前3个样本）
python main.py --max_samples 3

# 只生成诊断任务数据
python main.py --tasks diagnosis

# 使用特定模型
python main.py --model gpt-4-turbo --delay 3.0
```

## 三大医疗任务

### 1. Diagnosis（诊断任务）

- **ToM焦点**：理解患者对症状的认知和担忧
- **推理重点**：
  - 患者对症状的理解程度
  - 患者的恐惧和期望
  - 信息收集策略
- **输出**：诊断结果和治疗建议

### 2. MedRecon（药物调和任务）

- **ToM焦点**：理解患者的用药信念和依从性
- **推理重点**：
  - 患者对药物的理解
  - 患者的用药顾虑
  - 潜在的依从性问题
- **输出**：药物调和建议

### 3. Prescriptions（处方任务）

- **ToM焦点**：理解患者的治疗偏好和能力
- **推理重点**：
  - 患者对治疗的期望
  - 患者的生活方式约束
  - 患者的经济考虑
- **输出**：个性化处方方案

## 论文要求落地对照

### 论文1：双步骤ToM

| 要求 | 实现位置 |
|------|----------|
| Step1自主决策是否调用ToM | `tom_reasoning.py` → `step1_tom_invocation_decision()` |
| 可返回should_invoke_tom=False | ✅ 支持返回False |
| 自适应DoM（0-1阶） | `tom_reasoning.py` → `_determine_adaptive_dom()` |
| 禁止高阶DoM | ✅ 医疗合作场景只使用0阶/1阶 |
| TypeA/B/C错误检测修正 | `tom_error_detector.py` → `detect_and_correct_errors()` |
| 心智边界隔离 | `tom_models.py` → `MentalBoundary` |

### 论文2：动态时序ToM

| 要求 | 实现位置 |
|------|----------|
| 时序心理轨迹 | `tom_models.py` → `TemporalMentalTrajectory` |
| 因果触发链 | `tom_models.py` → `CausalEvent` |
| 时序链式推理 | `tom_models.py` → `TemporalChainLink` |
| 禁用普通CoT | `tom_reasoning.py` → `_build_temporal_chain_prompt()` |
| 患者回复由心理状态驱动 | `patient_simulator.py` → `generate_patient_response()` |
| 禁止通用静态回复 | `patient_simulator.py` → `FORBIDDEN_GENERIC_RESPONSES` |
| 解决中间丢失 | `TemporalMentalTrajectory` → `anchored_history` |
| ToM目标终止条件 | `tom_goal_checker.py` → `check_tom_goal_achieved()` |

### 全局禁止行为检查

| 禁止行为 | 状态 |
|----------|------|
| 空函数、空列表、空字段仅定义不使用 | ✅ 所有字段都有默认值和使用逻辑 |
| ToM推理与医生/患者回复脱节 | ✅ 回复完全基于ToM推理结果生成 |
| 固定ToM调用 | ✅ Step1可返回False |
| 固定DoM层级 | ✅ 自适应选择0阶/1阶 |
| 用普通链式思维替代时序链式推理 | ✅ 使用TemporalChainLink |
| 患者模拟器生成通用静态回复 | ✅ 禁止列表+验证机制 |

## 输出文件

运行后会在输出目录生成：

- `diagnosis_tom_dataset.jsonl` - 诊断任务数据集（含ToM标注）
- `medrecon_tom_dataset.jsonl` - 药物调和任务数据集（含ToM标注）
- `prescriptions_tom_dataset.jsonl` - 处方任务数据集（含ToM标注）

## 运行示例输出

```
============================================================
Processing sample 1...
Chief Complaint: Pleuritic chest pain

  [DIAGNOSIS] Generating ToM-based dialogue...
  [DIAGNOSIS] Generated 6 dialogue turns
  [DIAGNOSIS] ToM annotations: 6
  [DIAGNOSIS] ToM errors detected & corrected: 2
  [DIAGNOSIS] DoM levels used: {0, 1}

  [MEDRECON] Generating ToM-based dialogue...
  [MEDRECON] Generated 5 dialogue turns
  [MEDRECON] ToM annotations: 5
  [MEDRECON] ToM errors detected & corrected: 1
  [MEDRECON] DoM levels used: {0, 1}
```

## 注意事项

1. **API费用**：每个样本需要多次LLM调用（ToM推理+对话生成），请注意成本控制
2. **速率限制**：建议delay设置为2.0秒以上避免API限流
3. **数据质量**：建议对生成的ToM标注进行人工审核
4. **隐私保护**：确保处理的数据符合医疗隐私法规

## 许可证

本工具仅供研究和教育用途，请遵守相关法律法规和医疗伦理规范。

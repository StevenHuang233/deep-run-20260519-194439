# 框架对比说明

本文对比三类框架：

1. 老师/题目给出的目标框架。
2. 本项目从 0 构建的 OOP Harness 各迭代版本。
3. 参考开源仓库中的典型框架模式。

## 总览对比

| 框架 | 定位 | 核心闭环 | 失败处理 | 记忆机制 | 工具接口 | 当前适配度 |
| --- | --- | --- | --- | --- | --- | --- |
| 老师原始框架 | 课题目标与评测要求 | 任务输入 -> Agent -> 工具 -> 输出/轨迹 | 题目要求记录失败轨迹，后续可反思 | 强调跨任务自进化，但实现细节留给参赛者 | 已给定模型 API、搜索、浏览器服务启动方式 | 是最高层约束 |
| GPT 详细方案 1 | 通用自进化 Agent 设计 | Planner -> Executor -> Reflector -> Memory -> Log | 失败后离线反思，写入 Memory | Skill / episodic / bad pattern memory | 抽象 Environment 层 | 架构完整，但偏概念 |
| GPT 详细方案 2 | OOP 交付版设计 | Harness -> ReAct -> Tool Gate -> 32B Compiler -> Skeptical Memory | 物理门禁阻断死循环，32B 反思编译规则 | CLIN 风格规则 + ExpeL 奖惩淘汰 | 明确类接口和 JSONL 轨迹 | 最接近本实现 |
| 本项目 V0-V2 | 从 0 构建可运行基线 | Harness + ReAct + Gate + Memory + Reflection | 基础门禁、基础反思、基础日志 | `memory.json`，Top-2 检索 | 直接复用旧 search/browser 模块 | 可跑通基本评测 |
| 本项目 V3-V4 | 稳定性和批跑增强 | ReAct + retry/resume/status | API 重试、批量续跑、单样本异常不拖垮全批次 | Memory 裁剪和权重维护 | 接口保持不变 | 更适合长批量跑榜 |
| 本项目 V5 | 单 case 反思自救 | 首轮失败 -> 临时 CLIN 规则 -> 同 case 重试 | 曾加入非模型 `FALLBACK_ANSWER` 兜底 | 失败规则同时写入长期 Memory | 接口保持不变 | 工程上稳，但不符合“答案必须模型生成” |
| 本项目 V6 当前版 | 模型答案优先的刷榜版 | 首轮 ReAct + 至少 4 次反思恢复，共 5 次模型尝试 | 5 次模型仍失败则空 `pred`，不造答案 | 规则可写入长期 Memory，也可在同 case 临时使用 | 接口保持不变 | 当前推荐版 |

## 老师框架 vs 本项目当前框架

| 维度 | 老师/题目框架 | 本项目当前框架 V6 |
| --- | --- | --- |
| 抽象层级 | 任务书级别，定义目标、工具、数据和提交规范 | 代码级别，已经实现为独立可运行目录 `evo_harness_oop` |
| 构建方式 | 未限定必须从旧代码改还是从 0 构建 | 从 0 构建；不修改旧代码，只复用旧工具服务 |
| 数据入口 | 原题材料给出数据和运行方式；最终目标是 `benchmark.csv` | 直接支持 `benchmark.csv`，也兼容 SimpleVQA / 2Wiki JSONL |
| 主模型 | 要求调用给定模型 API | OpenAI-compatible API，默认 Qwen3.5 主模型 |
| 辅助模型 | 允许 32B 以下模型用于反思和记忆 | `REFLECTION_MODEL_NAME` 默认 `qwen-3-32b`，显式限制不超过 32B |
| 工具接口 | 搜索和浏览器接口由现有代码/服务提供 | `ToolEnvironment` 懒加载旧 `tools/search_tool.py` 与 `tools/browser_tool.py`，不复制、不修改 |
| 反思机制 | 目标上要求失败后可反思、可进化 | 实现了 `CognitiveCompiler`，生成结构化 reflection 和 CLIN 单行规则 |
| 自进化 | 强调跨任务经验累积 | 同时支持跨任务 Memory 和同 case 临时反思恢复 |
| 轨迹 | 要求记录 Thought/Tool/Observation/Reflection/Memory | JSONL 记录 system/user/assistant/tool/event，包括 run status、memory write、reflection |
| 失败输出 | 老师材料通常要求按提交格式输出 | 当前遵循“答案必须模型生成”：5 次模型尝试失败后 `pred` 为空 |
| 蒸馏 | 可作为加分项或后续扩展 | 当前不做 LoRA/SFT 蒸馏 |

核心区别一句话：老师框架是“该做什么”的目标架构；本项目 V6 是“怎么跑起来”的工程实现，并且为了当前刷榜约束，明确采用模型生成答案、失败留空、旧浏览器服务复用、`benchmark.csv` 批量稳定运行。

## 本项目几个版本的演进

| 版本 | Commit | 主题 | 主要变化 | 保留/废弃 |
| --- | --- | --- | --- | --- |
| V0 | `e8f77d5` | Baseline OOP Harness | 从 0 搭建 Harness、ReAct、Gate、Memory、Reflection、JSONL 轨迹 | 保留 |
| V1 | `53015ee` | 32B Reflection And Memory | 引入 32B 以下辅助模型限制；CLIN 规则；ExpeL 风格 `ADD/EDIT/IGNORE` | 保留 |
| V2 | `b596ab4` | Interface Contracts | 环境变量实例化读取；图片 data URI；任务 ID 安全；接口契约测试 | 保留 |
| V3 | `f9d54f9` | Stability Patterns | 模型调用 retry；`--resume`；最近 step 上下文裁剪 | 保留 |
| V4 | `9091f6e` | Batch Fault Tolerance | 单样本异常继续跑；`*.status.json`；Memory 上限裁剪 | 保留 |
| V5 | `16a17c7` | Single-Case Reflection Retry | 同 case 失败后临时 CLIN 规则恢复；曾加入非模型 fallback | 反思恢复保留，非模型 fallback 废弃 |
| V6 | `6d6b7ee` | Model-Only Five Attempts | 答案只能来自模型；默认至少 5 次模型尝试；仍失败则空 `pred` | 当前推荐版 |
| V7 | `2d3e5af` | Dream Memory Consolidation | 离线读取失败轨迹，整理成 CLIN 规则，合并/裁剪 `memory.json` | 推荐批跑后使用 |
| V8 | 当前提交 | Typed Hybrid Memory | 统一 schema、三类 memory、hybrid retrieval、usage logs、confidence 更新 | 当前推荐版 |

## 参考仓库框架对比

| 参考框架 | 核心模式 | 可借鉴点 | 本项目吸收方式 | 未采用部分 |
| --- | --- | --- | --- | --- |
| mini-swe-agent / SWE-agent | 极简 Agent 循环：query -> execute actions -> save trajectory -> exit status | 简洁控制循环、模型调用统计、轨迹保存、批量状态 | `run_status` event、API calls 统计、`--resume`、批量异常续跑 | 不采用其代码编辑环境和 SWE-bench 专用 patch 输出 |
| CLIN | 交互后总结因果 memory，用 memory 快速适应新任务/新环境 | CLIN 单行因果规则、episode 内失败恢复、最近轨迹摘要 | `CognitiveCompiler` 生成 `Context -> Action is Necessary` 规则；同 case 临时规则恢复 | 不使用 ScienceWorld 环境，不做句向量动作映射 |
| ExpeL | 收集经验 -> 抽取 insight -> 评测时召回；规则支持 ADD/EDIT/UPVOTE/DOWNVOTE/REMOVE | 自然语言经验库、规则合并、奖惩、淘汰 | `SkepticalMemoryManager` 的 `ADD/EDIT/IGNORE`、usage logs、confidence 更新、`MEMORY_MAX_RULES` | 暂不做完整人工审计式 experience gathering |
| AutoDream / OpenClaw memory consolidation | 后台整理多轮 session memory | 定期整理、合并重复、删除陈旧规则、写整理报告 | `MemoryDreamer` 读取失败轨迹，生成规则并更新 `memory.json` / `dream_report.json` | 不做常驻后台任务，避免影响评测运行 |
| OpenClaw / Tool Gate 类设计 | 工具调用层做死循环和错误拦截 | 物理门禁独立于 LLM，减少重复调用和 token 浪费 | `ToolGateMiddleware` 用 Jaccard 检测近似重复，追踪连续错误 | 没有复刻其全部工程实现，只吸收门禁思想 |
| LLaMA-Factory | 统一 SFT / LoRA / 数据格式流水线 | 后续蒸馏训练可用 | 当前只保留轨迹数据可导出能力 | 暂不实现蒸馏 |

## 当前框架的优势

- 从 0 构建，结构清晰，便于说明原创性和模块边界。
- 不修改旧代码，降低破坏现有搜索/浏览器启动方式的风险。
- 接口兼容：单题、批量、模型 API、搜索、浏览器环境变量都尽量沿用旧设计。
- 同时具备跨任务进化、同 case 自救和离线 Dream Memory：Memory 用于后续任务，临时 CLIN 规则用于当前 case 恢复，Dreamer 用于批跑后清洗失败模式。
- Memory schema 不绑定具体 benchmark 格式，`task_type` 只是检索 rerank 因子，未知任务仍可按 general/simpleqa/visual_qa/2wiki 等宽泛模式召回。
- 符合当前约束：答案必须由模型生成；至少 5 次尝试；失败后空 `pred`。

## 当前框架的不足

- 反思和 Dream Memory 编译默认可启发式 fallback；如果没有真实 32B API，记忆质量会弱于老师设想。
- Skeptical Memory 目前是关键词检索，不是向量检索，速度快但召回能力有限。
- 多角色 MAR 审计法庭尚未完整实现，目前是单编译器接口。
- 未做蒸馏训练链路，轨迹只是先按 JSONL 保存。
- 5 次尝试会增加 API 调用成本，正式跑榜前需要控制 `CASE_REFLECTION_MAX_STEPS` 和工具门禁阈值。

## 推荐表述

如果用于报告，可以这样概括：

> 本项目采用从 0 构建、接口兼容的 OOP Harness。整体框架吸收老师方案中的 Harness、ReAct、Tool Gate、32B Reflection、Skeptical Memory 五层设计，同时参考 mini-swe-agent 的极简控制循环、CLIN 的因果记忆、ExpeL 的经验奖惩淘汰机制。与老师方案相比，本实现更偏工程落地：直接面向 `benchmark.csv`，复用旧搜索/浏览器服务，提供批量运行、断点续跑、轨迹记录、状态统计和接口测试。当前 V6 版本严格要求答案由模型生成，默认至少尝试 5 次，失败则提交空 `pred`，避免非模型兜底答案污染结果。

## 参考链接

- mini-swe-agent: https://github.com/SWE-agent/mini-swe-agent
- SWE-agent trajectory docs: https://github.com/SWE-agent/SWE-agent/blob/main/docs/usage/trajectories.md
- CLIN: https://github.com/allenai/clin
- ExpeL: https://github.com/LeapLabTHU/ExpeL
- LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory

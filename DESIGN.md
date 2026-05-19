# 自进化任务求解 Agent 设计说明

## 目标

本实现的目标是在不改动原始代码的前提下，从 0 构建一个接口兼容的自进化任务求解 Harness。系统面向 `benchmark.csv`，支持文本题与 base64 图片题，复用现有搜索/浏览器服务，并通过轨迹、反思、记忆和物理门禁形成轻量进化闭环。

## 总体流程

```mermaid
flowchart TD
    A["benchmark.csv / 单题输入"] --> B["HarnessOrchestrator"]
    B --> C["SkepticalMemoryManager 检索 Top-2 规则"]
    C --> D["Planner 注入 <historical_guidelines>"]
    D --> E["Qwen3.5 主模型 ReAct 循环"]
    E --> F{"是否调用工具"}
    F -->|是| G["ToolGateMiddleware 防重复/异常"]
    G --> H["ToolEnvironment 复用旧 search/browser 模块"]
    H --> E
    F -->|否| I["短答案 pred"]
    G -->|阻断/失败| J["CognitiveCompiler 32B 以下模型反思"]
    J --> K["CLIN 规则"]
    K --> L["Memory ADD/EDIT/IGNORE"]
    L --> M["memory.json"]
    I --> N["提交 JSONL + 轨迹 JSONL"]
```

## 模块职责

### HarnessOrchestrator

位置：`evo_agent/harness.py`

- 读取 `benchmark.csv`、SimpleVQA JSONL、2Wiki JSONL。
- 对 `benchmark.csv` 的 base64 图片进行本地落盘，同时保留原始 `image` 字段用于提交 JSONL。
- 驱动最多 `MAX_STEPS` 轮 OpenAI-compatible tool calling。
- 记录 system/user/assistant/tool/event 轨迹。
- 在失败、门禁阻断或答案不匹配时触发反思和记忆写入。

### ToolEnvironment

位置：`evo_agent/environment.py`

- 不复制旧浏览器代码，只懒加载根目录 `tools/browser_tool.py`。
- 不复制旧搜索代码，只懒加载根目录 `tools/search_tool.py`。
- 暴露与当前服务兼容的工具名：`search_text`、`search_image`、`browser_navigate`、`browser_get_text`、`browser_click`、`browser_type`、`browser_parallel`。
- `search_image` 同时兼容 `image` 与 `image_url` 参数。

### ToolGateMiddleware

位置：`evo_agent/gate.py`

- 参考 OpenClaw loop detection 的思路维护最近工具调用历史。
- 用 Jaccard 相似度检测参数微调式死循环。
- 对 401/403/auth/API key 等硬错误立即阻断。
- 对 timeout/500/session/proxy-error 等连续错误阻断，减少 token 浪费。

### CognitiveCompiler

位置：`evo_agent/compiler.py`

- 默认启发式反思；设置 `REFLECTION_MODEL_ENABLED=1` 后调用辅助模型。
- 辅助模型显式名称不得超过 32B，推荐 `qwen3-32b`。
- 输出结构化 reflection，并生成单行 CLIN 规则。
- 额外提供 `advise_memory_update()`，让 32B 以下模型辅助判断记忆操作：`ADD`、`EDIT`、`IGNORE`。

### SkepticalMemoryManager

位置：`evo_agent/memory.py`

- 快速关键词硬匹配召回 Top-2 规则，避免 prompt 膨胀。
- 记忆仅作为线索注入，不作为事实。
- 参考 ExpeL 的规则编辑思想，用 `ADD/EDIT/IGNORE` 管理新规则。
- 使用 `UPVOTE/DOWNVOTE` 更新权重：

```text
W_c = (Up - Down) / (Up + Down + 0.1)
```

- 权重低于 `0.20` 的规则主动淘汰。

## 参考仓库对照后的优化

- mini-swe-agent：吸收其“控制循环 + 轨迹状态 + exit_status/model_stats”思想，在轨迹末尾写入 `run_status` event。
- CLIN：反思规则使用因果抽象形式，不写具体答案，只保留 `Necessary` / `Does Not Contribute` 这类可迁移关系。
- ExpeL：长期记忆不盲目 append，改为 `ADD/EDIT/IGNORE`，并用成功/失败结果做权重奖惩。
- OpenClaw loop detection 文档：门禁层独立于 LLM，通过物理代码阻断重复工具调用和连续工具错误。

## 第二轮鲁棒性优化

- `HarnessConfig` 改为实例化时读取环境变量，避免测试或外部 wrapper 在 import 后设置环境变量却不生效。
- base64 图片识别支持 `data:image/...;base64,...` 形式，便于单题 CLI、CSV 和多模态接口共用一条路径。
- 轨迹文件名会清洗 `task_id`，避免用户传入包含路径分隔符或特殊字符的任务 ID。
- 新增 `tests/test_interface_contracts.py`，用标准库 `unittest` 检查工具名、环境变量读取、32B 上限、图片 data URI 和输出 schema。

## 第三轮参考仓库吸收

- mini-swe-agent 的模型层会对瞬时 API 失败做指数退避重试。本实现加入 `LLM_RETRY_ATTEMPTS`、`LLM_RETRY_MIN_SECONDS`、`LLM_RETRY_MAX_SECONDS`，但对鉴权、权限、404、上下文窗口等非重试错误立即失败。
- mini-swe-agent 的 benchmark runner 会跳过已有预测，避免长批量运行中断后重跑。本实现新增 CLI `--resume`，读取输出 JSONL 中已有 `index` 并跳过。
- CLIN 会在长轨迹中优先保留最近 action-observation 对。本实现新增 `CONTEXT_RECENT_STEPS`，默认只把 system/user 和最近 8 个执行 step 回放给主模型，完整轨迹仍然写入磁盘。

## 第四轮参考仓库吸收

- mini-swe-agent 的批量 runner 会捕获单个样本异常、记录 exit status，并继续后续样本。本实现默认 `BATCH_CONTINUE_ON_ERROR=1`，遇到未捕获异常时写出空 `pred` 和 `uncaught_*` 状态；使用 `--strict` 可恢复遇错即停。
- mini-swe-agent 会保存批量状态报告。本实现为提交 JSONL 旁路生成 `*.status.json`，统计本次运行的 exit status、API calls 和 token 消耗。
- ExpeL 在规则列表变满后会优先 REMOVE、按计数排序并裁剪。本实现新增 `MEMORY_MAX_RULES`，按权重、merge/edit 次数、更新时间保留最强规则，默认最多 64 条。

## 接口兼容性

单题接口：

```bash
python -m task_runner --instruction "..." --task-id my_task_001
```

根目录批量接口：

```bash
python -m evo_harness_oop.task_runner --task-file benchmark.csv --output evo_harness_oop/outputs/benchmark_predictions.jsonl
```

模型接口：

- `LLM_BASE_URL`
- `MODEL_NAME`
- `MAX_STEPS`
- `MAX_TOKENS`
- `DISABLE_TOOLS`
- `CONTEXT_RECENT_STEPS`
- `LLM_RETRY_ATTEMPTS`
- `LLM_RETRY_MIN_SECONDS`
- `LLM_RETRY_MAX_SECONDS`
- `BATCH_CONTINUE_ON_ERROR`
- `MEMORY_MAX_RULES`

搜索/浏览器接口：

- `SEARCH_PROXY_URL`
- `SERPER_API_KEY`
- `JINA_API_KEY`
- `SANDBOX_BASE_URL`

辅助模型接口：

- `REFLECTION_MODEL_ENABLED`
- `MEMORY_MODEL_ENABLED`
- `REFLECTION_LLM_BASE_URL`
- `REFLECTION_MODEL_NAME`
- `REFLECTION_MODEL_MAX_B`
- `REFLECTION_API_KEY`

## 轨迹与输出

每个任务生成独立 JSONL 轨迹，包含：

- `system`：主提示词和历史规则。
- `user`：任务与图片输入。
- `assistant`：模型输出、tool_calls、reasoning_content、token usage。
- `tool`：工具 observation。
- `event`：memory retrieval、reflection、memory write、run status。

提交输出 JSONL：

```json
{"index": 0, "instruction": "...", "image": "...", "answer": "", "pred": "..."}
```

## 当前不做的部分

- 不做 LoRA/SFT 蒸馏。
- 不改旧浏览器服务代码。
- 不把参考仓库源码复制进本项目，只吸收公开设计模式。

## 验证命令

```bash
python -m compileall -q .
python -m unittest discover -s tests
MOCK_LLM=1 python -m task_runner --task-file ../benchmark.csv --limit 1 --output outputs/mock_predictions.jsonl
MOCK_LLM=1 python -m task_runner --task-file ../benchmark.csv --limit 1 --resume --output outputs/mock_predictions.jsonl
MOCK_LLM=1 python -m task_runner --task-file ../benchmark.csv --limit 1 --strict --output outputs/mock_predictions.jsonl
```

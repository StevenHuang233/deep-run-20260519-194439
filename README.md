# OOP Self-Evolving Harness

这是一个从 0 构建的独立实现目录，不修改根目录现有 harness 代码。

## 设计取舍

- ReAct 控制、轨迹、物理门禁、32B 辅助反思、Skeptical Memory、benchmark runner 均在 `evo_agent/` 内重新实现。
- 浏览器服务直接复用根目录现有 `sandbox_client.py` 和 `tools/browser_tool.py`。
- 搜索服务直接复用根目录现有 `tools/search_tool.py`。
- 暂不实现蒸馏训练链路。

## 单题启动

```bash
cd evo_harness_oop
python -m task_runner --instruction "请向我介绍上海创智学院" --task-id my_task_001
```

环境变量沿用旧 harness：

```bash
export LLM_BASE_URL=http://127.0.0.1:8000/v1
export MODEL_NAME=Qwen3.5-9B
export SEARCH_PROXY_URL=http://127.0.0.1:1227
export SANDBOX_BASE_URL=http://127.0.0.1:8080
```

## 32B 辅助模型

反思和记忆编辑可启用外部辅助模型，默认推荐 `qwen-3-32b`，并强制限制显式模型名不超过 32B。

```bash
export REFLECTION_MODEL_ENABLED=1
export MEMORY_MODEL_ENABLED=1
export REFLECTION_LLM_BASE_URL=http://127.0.0.1:8000/v1
export REFLECTION_MODEL_NAME=qwen3-32b
export REFLECTION_MODEL_MAX_B=32
```

## 单 case 反思自救

答案必须来自主模型生成，不使用 `unknown` 等非模型兜底答案。默认每个样本至少给主模型 5 次生成机会：

1. 常规 ReAct 首轮执行。
2. 如果首轮被门禁阻断、达到步数上限、模型没有产出可解析答案，Harness 会对同一个 case 的失败轨迹做反思，生成临时 CLIN 规则。
3. 临时规则会作为 skeptical hint 追加到同一个轨迹中，再给主模型短预算恢复尝试。
4. 如果 5 次模型尝试后仍没有可解析答案，提交 JSONL 中的 `pred` 保持空字符串。

可调参数：

```bash
export MIN_MODEL_ATTEMPTS=5
export CASE_REFLECTION_ATTEMPTS=4
export CASE_REFLECTION_MAX_STEPS=6
```

## benchmark.csv 批量运行

```bash
cd evo_harness_oop
python -m task_runner --task-file ../benchmark.csv --output outputs/benchmark_predictions.jsonl
```

也可以从仓库根目录运行：

```bash
python -m evo_harness_oop.task_runner --task-file benchmark.csv --output evo_harness_oop/outputs/benchmark_predictions.jsonl
```

断点续跑时不会删除已有输出，并会跳过已出现的 `index`：

```bash
python -m evo_harness_oop.task_runner --task-file benchmark.csv --output evo_harness_oop/outputs/benchmark_predictions.jsonl --resume
```

批量运行默认会给每次 run 创建独立 memory，位置在输出文件同目录的
`run_memories/memory_<run_id>.json`。同一次 run 内的样本共享该 memory；
下一次 50 条、100 条运行会自动用新的 memory，互不覆盖。需要复用全局
`MEMORY_DB_PATH` 时加 `--shared-memory`，需要固定本轮 memory 名称时加
`--run-id my_run_001`。

并发运行：

```bash
python -m task_runner --task-file "$SIMPLEVQA_FILE" --image-dir "$SIMPLEVQA_IMAGE_DIR" --workers 2
```

并发时同一个 run 内仍共享同一个 memory，memory 读写在进程内加锁，避免同进程
`--workers` 并发写入冲突；不要同时启动两个独立进程写同一个 `MEMORY_DB_PATH`。

默认批量运行会在单个样本出现未捕获异常时继续处理后续样本，并生成
`benchmark_predictions.jsonl.status.json` 状态摘要。需要遇错立即停止时使用：

```bash
python -m evo_harness_oop.task_runner --task-file benchmark.csv --output evo_harness_oop/outputs/benchmark_predictions.jsonl --strict
```

## Dream Memory 离线整理

参考 AutoDream 的批量记忆整理思路，系统提供离线 `MemoryDreamer`：读取最近失败轨迹，生成可泛化 CLIN 规则，合并/裁剪 `memory.json`，并写出 `dream_report.json`。它不参与单题答题，也不会填充 `pred`。

手动整理：

```bash
python -m task_runner --dream-memory --traj-dir trajectories --dream-report outputs/dream_report.json
```

批量运行后自动整理：

```bash
python -m task_runner --task-file ../benchmark.csv --output outputs/benchmark_predictions.jsonl --dream-after-run
```

可调参数：

```bash
export DREAM_MAX_TRAJECTORIES=64
export DREAM_MIN_FAILURES=1
export DREAM_REPORT_PATH=outputs/dream_report.json
```

## Memory Schema 与 Hybrid 检索

长期记忆采用统一 schema，分为 `skill`、`bad_pattern`、`reflection` 三类。Memory 只保存可复用经验，不保存完整轨迹和具体答案。每条记忆包含触发条件、适用任务、执行步骤、避免事项、来源和统计置信度。

默认检索使用轻量 hybrid 策略：

- dense route：默认本地 hash embedding，便于无依赖运行。
- lexical route：`trigger_keywords` 与任务文本做词汇碰撞。
- rerank：综合向量相似度、任务类型、关键词命中、confidence、历史成功率。

如果部署了 embedding 服务，可直接打开接口：

```bash
export MEMORY_EMBEDDING_ENABLED=1
export EMBEDDING_BASE_URL=http://127.0.0.1:8000/v1
export EMBEDDING_MODEL=all-MiniLM-L6-v2
export EMBEDDING_API_KEY=EMPTY
```

使用后的记忆会写入 `usage_logs.jsonl`，任务结束后用：

```text
confidence = (success_count + 1) / (success_count + failure_count + 2)
```

更新可信度。`usage_count >= 3` 且 `confidence < 0.3` 的记忆会被标记为 `deprecated`，后续不会检索。

`benchmark.csv` 中的 base64 图片会落到 `outputs/benchmark_images/`，用于模型视觉输入和 `search_image` 本地文件上传；输出 JSONL 仍保留原始 `image` 字段。

输出 JSONL 遵循提交格式：

```json
{"index": 0, "instruction": "...", "image": "", "answer": "", "pred": "..."}
```

## 本地无 API 验证

```bash
cd evo_harness_oop
MOCK_LLM=1 python -m task_runner --task-file ../benchmark.csv --limit 2 --output outputs/mock_predictions.jsonl
```

`MOCK_LLM=1` 只用于验证 CSV 解析、图片落盘、轨迹和输出格式；真实评测不要设置。

上下文和模型重试可用环境变量控制：

```bash
export CONTEXT_RECENT_STEPS=8
export LLM_RETRY_ATTEMPTS=3
export LLM_RETRY_MIN_SECONDS=1
export LLM_RETRY_MAX_SECONDS=8
export TOOL_RETRY_ATTEMPTS=3
export TOOL_RETRY_MIN_SECONDS=1
export TOOL_RETRY_MAX_SECONDS=8
export MEMORY_MAX_RULES=64
export BATCH_CONTINUE_ON_ERROR=1
export MIN_MODEL_ATTEMPTS=5
export CASE_REFLECTION_ATTEMPTS=4
```

接口契约测试：

```bash
python -m unittest discover -s tests
```

## SimpleVQA 本地图片与 LLM Judge

SimpleVQA JSONL 中的本地 `image` 会通过 `--image-dir` 读取为视觉输入；`image_url` 会保留给 `search_image` 做反向图搜。

```bash
cd /inspire/qb-ilm2/project/26summer-camp-01/26210500/evo_harness_oop
export SIMPLEVQA_FILE=/inspire/qb-ilm2/project/26summer-camp-01/26210500/datasets/simpleVQA/SimpleVQA.jsonl
export SIMPLEVQA_IMAGE_DIR=/inspire/qb-ilm2/project/26summer-camp-01/26210500/datasets/simpleVQA
export JUDGE_LLM_BASE_URL=https://notebook-inspire.sii.edu.cn/ws-7c23bd1d-9bae-4238-803a-737a35480e18/project-39fbffc7-dcca-4fb4-b43a-2f69f72f7e52/user-b1acf6ce-25a4-4cb6-b428-f427f4a59686/vscode/b2aa27b1-e0f7-425d-b208-acbd7f40ef68/68f1224c-8cc9-4e87-8701-523c6e59db1f/proxy/8001
export JUDGE_MODEL_NAME=Qwen3-32B

MOCK_LLM=1 python -m task_runner \
  --task-file "$SIMPLEVQA_FILE" \
  --image-dir "$SIMPLEVQA_IMAGE_DIR" \
  --limit 2 \
  --output outputs/simplevqa_mock_predictions.jsonl \
  --judge-after-run \
  --judge-output outputs/simplevqa_mock_judge.jsonl
```

已有预测文件可单独评分：

```bash
python -m task_runner \
  --judge-file outputs/simplevqa_predictions.jsonl \
  --judge-output outputs/simplevqa_judge.jsonl
```

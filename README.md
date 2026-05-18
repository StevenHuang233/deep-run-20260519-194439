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

## benchmark.csv 批量运行

```bash
cd evo_harness_oop
python -m task_runner --task-file ../benchmark.csv --output outputs/benchmark_predictions.jsonl
```

也可以从仓库根目录运行：

```bash
python -m evo_harness_oop.task_runner --task-file benchmark.csv --output evo_harness_oop/outputs/benchmark_predictions.jsonl
```

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

接口契约测试：

```bash
python -m unittest discover -s tests
```

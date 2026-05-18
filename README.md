# OOP Self-Evolving Harness

这是一个从 0 构建的独立实现目录，不修改根目录现有 harness 代码。

## 设计取舍

- ReAct 控制、轨迹、物理门禁、反思、记忆、benchmark runner 均在 `evo_agent/` 内重新实现。
- 浏览器服务直接复用根目录现有 `sandbox_client.py` 和 `tools/browser_tool.py`。
- 搜索服务直接复用根目录现有 `tools/search_tool.py`。
- 暂不实现蒸馏流程。

## 单题启动

```bash
cd evo_harness_oop
python -m task_runner --instruction "请向我介绍上海创智学院" --task-id my_task_001
```

环境变量沿用原 harness：

```bash
export LLM_BASE_URL=http://127.0.0.1:8000/v1
export MODEL_NAME=Qwen3.5-9B
export SEARCH_PROXY_URL=http://127.0.0.1:1227
export SANDBOX_BASE_URL=http://127.0.0.1:8080
```

## benchmark.csv 批量运行

```bash
cd evo_harness_oop
python -m task_runner --task-file ../benchmark.csv --output outputs/benchmark_predictions.jsonl
```

或从仓库根目录运行：

```bash
python -m evo_harness_oop.task_runner --task-file benchmark.csv --output evo_harness_oop/outputs/benchmark_predictions.jsonl
```

本目录会把 `benchmark.csv` 中的 base64 图片落到 `outputs/benchmark_images/`，用于模型视觉输入和 `search_image` 本地文件上传。

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

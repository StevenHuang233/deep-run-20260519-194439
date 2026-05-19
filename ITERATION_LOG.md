# Iteration Log

## Iteration 0: Baseline

Commit: `e8f77d5`

- 从 0 新建 OOP Harness。
- 保持旧搜索和浏览器服务为唯一工具实现来源。
- 支持单题和 `benchmark.csv` 批量运行。
- 支持轨迹 JSONL、Memory、Reflector、Gate 基础闭环。

## Iteration 1: 32B Reflection And Memory

Commit: `53015ee`

- 引入 32B 以下辅助模型限制，默认推荐 `qwen-3-32b`。
- 反思输出改为结构化 JSON 和 CLIN 单行因果规则。
- 记忆写入改为 ExpeL 风格 `ADD/EDIT/IGNORE`。
- 增加 `run_status` event 记录 exit status、API calls 和 token usage。
- 新增 `DESIGN.md` 总结整体系统流程。

## Iteration 2: Interface And Robustness Contracts

Commit: `b596ab4`

- 配置改为实例化时读取环境变量。
- 图片识别兼容 raw base64 和 data URI。
- 轨迹任务 ID 做文件名安全清洗。
- 新增标准库 unittest 接口契约测试。
- 设计文档补充验证命令和第二轮鲁棒性说明。

## Iteration 3: Reference Repo Stability Patterns

Commit: `f9d54f9`

- 参考 mini-swe-agent 的模型调用重试，加入指数退避和非重试错误识别。
- 参考 mini-swe-agent 的 benchmark 跳过机制，加入 `--resume` 断点续跑。
- 参考 CLIN 的最近 action-observation 上下文裁剪，加入 `CONTEXT_RECENT_STEPS`。
- 扩展接口契约测试，覆盖上下文裁剪和 resume 行为。

## Iteration 4: Batch Fault Tolerance And Memory Pruning

Commit: `9091f6e`

- 参考 mini-swe-agent 的 per-instance exception handling，批量默认单样本失败后继续跑。
- 为预测 JSONL 旁路生成 `*.status.json`，记录 exit status、API calls 和 token 汇总。
- 参考 ExpeL 的 bounded rule list，新增 `MEMORY_MAX_RULES` 并自动裁剪低价值规则。
- 扩展接口契约测试，覆盖批量异常续跑和 Memory 裁剪。

## Iteration 5: Single-Case Reflection Retry And Non-Empty Pred

Commit: `16a17c7`

- 参考 CLIN 的 episode 内小预算恢复，把失败轨迹即时编译成临时 CLIN 规则，并在同一个 case 内追加短 ReAct 重试。
- 参考 mini-swe-agent 的可追踪 exit/submission 模式，在轨迹中记录 `self_reflection_retry_start`、`self_reflection_retry_status` 和最终 `run_status`。
- 保留 ExpeL 风格长期记忆写入；同 case 恢复使用的规则也会参与后续奖惩。
- 默认 `ALWAYS_ANSWER=1`，正常答案和反思恢复都失败时才写入 `FALLBACK_ANSWER`，保证提交 JSONL 的 `pred` 非空。

## Iteration 6: Model-Only Five Attempts

Commit: `6d6b7ee`

- 按刷榜约束撤销非模型 fallback 答案，`pred` 只能来自主模型生成内容。
- 默认 `MIN_MODEL_ATTEMPTS=5`，首轮 ReAct 失败后至少追加 4 次单 case 反思恢复尝试。
- 5 次模型尝试后仍没有可解析答案时，提交 JSONL 保持空 `pred`。
- 批量未捕获异常恢复为 `uncaught_*` 状态和空 `pred`，避免混入非模型答案。

## Iteration 7: Dream Memory Consolidation

Commit: this commit

- 新增 `MemoryDreamer`，离线读取失败轨迹并整理成可泛化 CLIN 规则。
- 新增 CLI：`--dream-memory` 手动整理，`--dream-after-run` 在批量运行后整理。
- 新增 `dream_report.json`，记录 reviewed trajectories、failures、memory updates 和 prune 结果。
- Dream Memory 只更新 `memory.json`，不会生成或填充 `pred`。

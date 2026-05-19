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

Commit: this commit

- 参考 mini-swe-agent 的模型调用重试，加入指数退避和非重试错误识别。
- 参考 mini-swe-agent 的 benchmark 跳过机制，加入 `--resume` 断点续跑。
- 参考 CLIN 的最近 action-observation 上下文裁剪，加入 `CONTEXT_RECENT_STEPS`。
- 扩展接口契约测试，覆盖上下文裁剪和 resume 行为。

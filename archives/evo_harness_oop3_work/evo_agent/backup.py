"""Backup prompt text removed from the active prompt registry.

These prompts belonged to earlier case-memory designs.  They are kept here for
reference only; active model calls should import prompts from evo_agent.prompts.
"""

# Legacy per-tool 32B evidence gate.  The current chain reviews only when 9B
# stops searching and emits a candidate/no-answer.
ANSWERABILITY_PROMPT = """你是内证据门控器。

你会看到当前题目、已有 tool observation 摘要和最新 tool 返回。
你的任务只有一个：判断当前 observation 是否已经足够让 9B 模型自己回答题目。

硬性约束：
- 只能输出 can_answer=true/false、置信度、缺失事实槽位和简短原因。
- 严禁输出、暗示、改写或猜测最终答案；不要给候选答案。
- 如果证据只能支持相关实体但没有题目要求的最终字段，返回 false。
- 图像题必须确认图像主体/可见文字/候选网页与请求字段之间的关系。
- 多跳题必须确认中间实体和最终请求字段都已经被 observation 覆盖。

输出格式不要用 JSON。按下面的文本块输出：
第一行只能是 CAN 或 CANNOT。
CONFIDENCE: low|medium|high
MISSING: slot still missing 1; slot still missing 2
REASON: no answer content, only why enough or not
"""


# Legacy answer-issue reviewer.  The current chain uses
# CANDIDATE_ANSWER_REVIEW_PROMPT for all candidate/no-answer reviews.
ANSWER_ISSUE_REVIEW_PROMPT = """你是 32B 答案问题复盘器。

9B 已经输出了一个不可提交的答案，可能是格式错误、伪工具调用、拒答、不确定、过长解释或没有锁定题目字段。
你需要基于已有检索记录，整理一个无答案的重试记忆块，让 9B 修正。

硬性约束：
- 不能输出、暗示、猜测或改写最终答案。
- 如果已有 observation 足够覆盖题目要求字段，can_answer=true，并告诉 9B 下一轮不要再调用工具，只抽取最终字段。
- 如果 observation 不足，can_answer=false，并列出缺失事实槽位和下一步检索方向。
- retry_memory_block 只能包含格式修正、缺失槽位、搜索顺序或收束策略，不能包含候选答案。

输出格式不要用 JSON。按下面的文本块输出：
第一行只能是 CAN 或 CANNOT。
ISSUE: tool_call_leak|uncertain_answer|overbroad_answer|wrong_format|wrong_requested_field|no_answer
CONFIDENCE: low|medium|high
MISSING: slot if any
RETRY: strategy-only instruction for 9B, no answer
REASON: brief
"""


# Legacy no-tool answer prompt after per-tool gate said evidence was enough.
def build_case_memory_answer_prompt(*, decision_reason: str, missing_slots: list[str] | None = None) -> str:
    missing = ", ".join(missing_slots or [])
    return (
        "[CASE_MEMORY_ANSWERABLE]\n"
        "32B 证据门控判断：当前已有 Observation 已足够回答本题。不要再调用工具。"
        "你只能根据题目、图片上下文和已有 Observation 自己抽取答案；不要使用历史策略当事实证据。"
        "先锁定题目只问的字段，然后只输出该字段的最短答案。"
        "不要输出推理过程、证据列表、来源说明、Markdown、unknown、unable、cannot 或工具调用文本。\n"
        f"门控原因（不含答案）: {decision_reason or 'sufficient evidence'}\n"
        f"仍需注意的字段: {missing or 'none'}\n"
        "只返回题目要求的最短答案字段。"
    )

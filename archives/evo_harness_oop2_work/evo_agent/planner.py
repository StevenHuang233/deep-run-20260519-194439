"""Lightweight planner that injects skeptical memory into the system prompt."""

from __future__ import annotations


class Planner:
    """Task planner facade.

    The base model still performs dynamic ReAct planning. This class keeps the
    planning interface explicit and handles memory-conditioned prompt shaping.
    """

    def inject_historical_guidelines(self, system_prompt: str, rules: list[str]) -> str:
        if not rules:
            return system_prompt
        lines = "\n".join(f"- {rule}" for rule in rules)
        return (
            f"{system_prompt}\n"
            "<historical_guidelines>\n"
            f"{lines}\n"
            "</historical_guidelines>"
        )

    def task_kind(self, instruction: str, has_image: bool = False) -> str:
        text = instruction.lower()
        if has_image:
            return "visual_qa"
        if any(word in text for word in ("both", "same country", "older", "later")):
            return "comparison"
        if any(word in text for word in ("father", "director", "wife", "capital")):
            return "multi_hop"
        return "open_search"

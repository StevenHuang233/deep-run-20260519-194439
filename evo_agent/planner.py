"""Lightweight planner that injects skeptical memory into the system prompt."""

from __future__ import annotations

from .prompts import build_historical_guidelines_prompt


class Planner:
    """Task planner facade.

    The base model still performs dynamic ReAct planning. This class keeps the
    planning interface explicit and handles memory-conditioned prompt shaping.
    """

    def inject_historical_guidelines(self, system_prompt: str, rules: list[str]) -> str:
        return build_historical_guidelines_prompt(system_prompt, rules)

    def task_kind(self, instruction: str, has_image: bool = False) -> str:
        text = instruction.lower()
        if has_image:
            return "visual_qa"
        if any(word in text for word in ("both", "same country", "older", "later")):
            return "comparison"
        if any(word in text for word in ("father", "director", "wife", "capital")):
            return "multi_hop"
        return "open_search"

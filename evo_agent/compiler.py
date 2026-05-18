"""Cognitive compiler for failed trajectories."""

from __future__ import annotations

import json
import re
from typing import Any

from .prompts import REFLECTION_PROMPT
from .types import Reflection


class CognitiveCompiler:
    """Compile failed trajectories into single-line CLIN causal rules."""

    def __init__(
        self,
        ref_model: str = "qwen-3-32b",
        base_url: str = "",
        enable_model: bool = False,
    ) -> None:
        self.ref_model = ref_model
        self.base_url = base_url
        self.enable_model = enable_model and bool(base_url)
        self.generation_config = {
            "enable_thinking": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0,
        }

    def compile_failed_trace(
        self, trajectory: list[dict[str, Any]], task_instruction: str
    ) -> Reflection:
        if self.enable_model:
            try:
                rule = self._compile_with_model(trajectory, task_instruction)
                return Reflection(
                    failure_type="model_diagnosed_failure",
                    root_cause="External compiler generated a CLIN rule.",
                    correct_strategy=[rule],
                    memory_worthy=True,
                    clin_rule=rule,
                )
            except Exception:
                pass
        return self._heuristic_compile(trajectory, task_instruction)

    def _compile_with_model(
        self, trajectory: list[dict[str, Any]], task_instruction: str
    ) -> str:
        from openai import OpenAI

        client = OpenAI(base_url=self.base_url, api_key="EMPTY")
        trace_text = json.dumps(trajectory[-20:], ensure_ascii=False)[:24000]
        response = client.chat.completions.create(
            model=self.ref_model,
            messages=[
                {"role": "system", "content": REFLECTION_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Task:\n{task_instruction}\n\n"
                        f"Failed trajectory:\n{trace_text}"
                    ),
                },
            ],
            max_tokens=512,
            temperature=0.6,
            extra_body=self.generation_config,
        )
        content = response.choices[0].message.content or ""
        return self._single_line(content)

    def _heuristic_compile(
        self, trajectory: list[dict[str, Any]], task_instruction: str
    ) -> Reflection:
        tool_errors = []
        tool_names = []
        for item in trajectory:
            if item.get("role") != "tool":
                continue
            tool_names.append(item.get("fn_name", "tool"))
            content = str(item.get("content", ""))
            if "error" in content.lower() or "proxy-error" in content.lower():
                tool_errors.append((item.get("fn_name", "tool"), content[:300]))

        if tool_errors:
            name, err = tool_errors[-1]
            if name.startswith("browser"):
                failure_type = "tool_runtime_failure"
                root = f"{name} failed repeatedly: {err}"
                strategy = [
                    "Stop retrying the same browser call after service errors.",
                    "Switch to search_text or use existing snippets.",
                    "Return a short final answer once enough evidence exists.",
                ]
                rule = (
                    "[When browser tools return service/auth/timeout errors] -> "
                    "Action: switch to search_text or answer from existing observations "
                    "is Necessary to achieve Goal G (Confidence: should)"
                )
            else:
                failure_type = "search_or_api_failure"
                root = f"{name} failed: {err}"
                strategy = [
                    "Change query terms instead of repeating equivalent calls.",
                    "Use fewer fetched characters if context is growing.",
                ]
                rule = (
                    "[When search/API calls repeatedly fail or return proxy errors] -> "
                    "Action: change query terms and reduce fetch size is Necessary "
                    "to achieve Goal G (Confidence: should)"
                )
        elif len(tool_names) >= 3 and len(set(tool_names[-3:])) == 1:
            failure_type = "repeated_tool_loop"
            root = "The agent repeated the same tool family without new evidence."
            strategy = [
                "Compare the last observation before calling another tool.",
                "Stop after two equivalent calls and synthesize an answer.",
            ]
            rule = (
                "[When the last two observations do not add new evidence] -> "
                "Action: stop repeating equivalent tool calls is Necessary "
                "to achieve Goal G (Confidence: should)"
            )
        else:
            failure_type = "incomplete_reasoning_chain"
            root = "The trajectory ended before a concise final answer was produced."
            strategy = [
                "For multi-hop questions, resolve the intermediate entity first.",
                "Then query the final requested attribute.",
                "Only return the final answer.",
            ]
            rule = (
                "[For multi-hop or comparison questions] -> Action: resolve "
                "intermediate entities before final attributes is Necessary "
                "to achieve Goal G (Confidence: should)"
            )

        return Reflection(
            failure_type=failure_type,
            root_cause=root,
            correct_strategy=strategy,
            memory_worthy=True,
            clin_rule=rule,
        )

    def _single_line(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text.strip())
        if "\n" in text:
            text = text.splitlines()[0].strip()
        return text

"""Cognitive compiler for failed trajectories and memory operations."""

from __future__ import annotations

import json
import re
from typing import Any

from .config import normalize_openai_base_url
from .model_policy import assert_model_within_limit
from .prompts import MEMORY_UPDATE_PROMPT, REFLECTION_PROMPT
from .types import Reflection


class CognitiveCompiler:
    """Use a bounded auxiliary model to compile traces into CLIN-style rules."""

    def __init__(
        self,
        ref_model: str = "qwen-3-32b",
        base_url: str = "",
        api_key: str = "EMPTY",
        enable_model: bool = False,
        max_model_billion: float = 32,
    ) -> None:
        self.ref_model = ref_model
        self.base_url = base_url
        self.api_key = api_key
        self.max_model_billion = max_model_billion
        if enable_model:
            assert_model_within_limit(ref_model, max_model_billion)
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
        """Compile a failed run into structured reflection plus one CLIN rule."""
        if self.enable_model:
            try:
                return self._compile_with_model(trajectory, task_instruction)
            except Exception as exc:
                fallback = self._heuristic_compile(trajectory, task_instruction)
                fallback.root_cause = f"{fallback.root_cause} Model compiler fallback: {type(exc).__name__}: {exc}"
                return fallback
        return self._heuristic_compile(trajectory, task_instruction)

    def advise_memory_update(
        self,
        new_rule: str,
        similar_entries: list[dict[str, Any]],
        task_instruction: str,
    ) -> dict[str, Any]:
        """Return an ExpeL-style ADD/EDIT/IGNORE operation for memory writes."""
        if self.enable_model:
            try:
                return self._advise_memory_with_model(new_rule, similar_entries, task_instruction)
            except Exception:
                pass
        return self._heuristic_memory_advice(new_rule, similar_entries)

    def _compile_with_model(
        self, trajectory: list[dict[str, Any]], task_instruction: str
    ) -> Reflection:
        from openai import OpenAI

        client = OpenAI(base_url=normalize_openai_base_url(self.base_url), api_key=self.api_key)
        trace_text = json.dumps(trajectory[-24:], ensure_ascii=False)[:28000]
        response = client.chat.completions.create(
            model=self.ref_model,
            messages=[
                {"role": "system", "content": REFLECTION_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Task:\n{task_instruction}\n\n"
                        f"Trajectory JSONL rows:\n{trace_text}\n\n"
                        "Return strict JSON only."
                    ),
                },
            ],
            max_tokens=900,
            temperature=0.6,
            extra_body=self.generation_config,
        )
        content = response.choices[0].message.content or ""
        data = self._extract_json(content)
        rule = self._single_line(str(data.get("clin_rule") or data.get("rule") or content))
        return Reflection(
            failure_type=str(data.get("failure_type") or "model_diagnosed_failure"),
            root_cause=str(data.get("root_cause") or "Auxiliary compiler generated a rule."),
            correct_strategy=self._as_string_list(data.get("correct_strategy") or data.get("strategy") or [rule]),
            memory_worthy=bool(data.get("memory_worthy", True)),
            clin_rule=rule,
            confidence=str(data.get("confidence") or "should"),
        )

    def _advise_memory_with_model(
        self,
        new_rule: str,
        similar_entries: list[dict[str, Any]],
        task_instruction: str,
    ) -> dict[str, Any]:
        from openai import OpenAI

        client = OpenAI(base_url=normalize_openai_base_url(self.base_url), api_key=self.api_key)
        compact_entries = [
            {
                "id": entry.get("id"),
                "rule": entry.get("rule"),
                "weight": entry.get("weight"),
                "similarity": entry.get("similarity"),
            }
            for entry in similar_entries[:5]
        ]
        response = client.chat.completions.create(
            model=self.ref_model,
            messages=[
                {"role": "system", "content": MEMORY_UPDATE_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": task_instruction,
                            "candidate_rule": new_rule,
                            "similar_existing_rules": compact_entries,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            max_tokens=600,
            temperature=0.2,
            extra_body=self.generation_config,
        )
        data = self._extract_json(response.choices[0].message.content or "")
        operation = str(data.get("operation", "ADD")).upper()
        if operation not in {"ADD", "EDIT", "IGNORE"}:
            operation = "ADD"
        return {
            "operation": operation,
            "target_id": data.get("target_id"),
            "rule": self._single_line(str(data.get("rule") or new_rule)),
            "reason": str(data.get("reason") or "model_advice"),
        }

    def _heuristic_memory_advice(
        self, new_rule: str, similar_entries: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if not similar_entries:
            return {"operation": "ADD", "rule": new_rule, "reason": "no_similar_rule"}
        best = similar_entries[0]
        sim = float(best.get("similarity", 0.0))
        if sim >= 0.90:
            return {
                "operation": "IGNORE",
                "target_id": best.get("id"),
                "rule": best.get("rule", new_rule),
                "reason": f"near_duplicate_similarity_{sim:.2f}",
            }
        if sim >= 0.72:
            return {
                "operation": "EDIT",
                "target_id": best.get("id"),
                "rule": self._prefer_general_rule(best.get("rule", ""), new_rule),
                "reason": f"merge_similar_rule_{sim:.2f}",
            }
        return {"operation": "ADD", "rule": new_rule, "reason": f"weak_similarity_{sim:.2f}"}

    def _heuristic_compile(
        self, trajectory: list[dict[str, Any]], task_instruction: str
    ) -> Reflection:
        tool_errors = []
        tool_names = []
        final_content = ""
        for item in trajectory:
            role = item.get("role")
            if role == "assistant":
                final_content = str(item.get("content", "") or final_content)
            if role != "tool":
                continue
            tool_names.append(item.get("fn_name", "tool"))
            content = str(item.get("content", ""))
            if "error" in content.lower() or "proxy-error" in content.lower():
                tool_errors.append((item.get("fn_name", "tool"), content[:300]))

        visual_identity_question = any(
            marker in task_instruction.lower()
            for marker in ("图中", "image", "picture", "photo", "人物", "物体", "bridge")
        )
        if visual_identity_question and (
            "search_image" in tool_names
            or any(name == "search_text" for name in tool_names)
        ):
            failure_type = "vqa_subject_unverified"
            root = "The agent accepted a candidate visual subject without verifying it against the image/search evidence and the queried attribute."
            strategy = [
                "Generate candidates from visual evidence or image search first.",
                "Verify that the candidate subject matches the image and the requested attribute.",
                "Reject text-search hits that only match the attribute but not the visual subject.",
            ]
            rule = (
                "[When VQA requires identifying a person/object/place before asking its attribute] -> "
                "Action: verify the candidate subject against image evidence and the requested attribute before answering "
                "is Necessary to achieve Goal G (Confidence: should)"
            )
        elif tool_errors:
            name, err = tool_errors[-1]
            if name.startswith("browser"):
                failure_type = "tool_runtime_failure"
                root = f"{name} failed repeatedly: {err}"
                strategy = [
                    "Retry transient browser service errors a few times with the same arguments.",
                    "After repeated failures, switch to search_text or use existing snippets.",
                    "Return a short final answer once enough evidence exists.",
                ]
                rule = (
                    "[When browser tools return service/auth/timeout errors] -> "
                    "Action: retry briefly, then switch to search_text or answer from existing observations "
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
        elif not re.search(r"<answer>.*?</answer>", final_content, flags=re.S | re.I):
            failure_type = "missing_strict_answer"
            root = "The run ended without the requested strict answer tag."
            strategy = [
                "After evidence is sufficient, emit exactly one short answer.",
                "Wrap the answer in <answer>...</answer>.",
            ]
            rule = (
                "[When enough evidence has been observed] -> Action: emit one short "
                "<answer>...</answer> final response is Necessary to achieve Goal G "
                "(Confidence: should)"
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

    def _extract_json(self, text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
            text = re.sub(r"```$", "", text).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if not match:
                return {}
            data = json.loads(match.group(0))
        return data if isinstance(data, dict) else {}

    def _single_line(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    def _as_string_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if value:
            return [str(value).strip()]
        return []

    def _prefer_general_rule(self, old_rule: str, new_rule: str) -> str:
        old_rule = self._single_line(old_rule)
        new_rule = self._single_line(new_rule)
        if not old_rule:
            return new_rule
        if len(new_rule) > len(old_rule) * 1.4 and "->" in new_rule:
            return new_rule
        return old_rule

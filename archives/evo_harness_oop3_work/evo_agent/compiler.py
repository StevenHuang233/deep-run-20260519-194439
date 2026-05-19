"""Cognitive compiler for failed trajectories and memory operations."""

from __future__ import annotations

import json
import re
from typing import Any

from .config import normalize_openai_base_url
from .model_policy import assert_model_within_limit
from .prompts import MEMORY_UPDATE_PROMPT, REFLECTION_PROMPT, SEARCH_CHAIN_REFLECTION_PROMPT
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
                fallback.compiler_source = "heuristic_fallback"
                fallback.compiler_model = self.ref_model
                fallback.compiler_error = f"{type(exc).__name__}: {exc}"
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
            except Exception as exc:
                advice = self._heuristic_memory_advice(new_rule, similar_entries)
                advice["advisor_source"] = "heuristic_fallback"
                advice["advisor_model"] = self.ref_model
                advice["advisor_error"] = f"{type(exc).__name__}: {exc}"
                return advice
        advice = self._heuristic_memory_advice(new_rule, similar_entries)
        advice.setdefault("advisor_source", "heuristic")
        return advice

    def compile_search_chain_lesson(
        self,
        trajectory: list[dict[str, Any]],
        task_instruction: str,
        prediction: str,
    ) -> Reflection | None:
        """Ask the auxiliary model to extract reusable search-query strategy.

        This is a positive test-time-learning pass: it runs after the base agent
        has produced a parseable answer and looks for late useful queries,
        query templates, and fact-slot ordering lessons. It intentionally has no
        heuristic fallback because the user-requested behavior depends on the
        <=32B reviewer understanding the search chain.
        """
        if not self.enable_model:
            return None
        try:
            return self._compile_search_chain_with_model(trajectory, task_instruction, prediction)
        except Exception as exc:
            return Reflection(
                failure_type="search_chain_review_unavailable",
                root_cause=f"Search-chain reviewer failed: {type(exc).__name__}: {exc}",
                correct_strategy=[],
                memory_worthy=False,
                clin_rule="",
                compiler_source="search_chain_model_error",
                compiler_model=self.ref_model,
                compiler_error=f"{type(exc).__name__}: {exc}",
            )

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
                        f"Trajectory JSONL rows:\n{trace_text}"
                    ),
                },
            ],
            max_tokens=900,
            temperature=0.6,
            extra_body=self.generation_config,
        )
        content = response.choices[0].message.content or ""
        data = self._parse_reflection_text(content)
        rule = self._single_line(str(data.get("clin_rule") or data.get("rule") or content))
        return Reflection(
            failure_type=str(data.get("failure_type") or "model_diagnosed_failure"),
            root_cause=str(data.get("root_cause") or "Auxiliary compiler generated a rule."),
            correct_strategy=self._as_string_list(data.get("correct_strategy") or data.get("strategy") or [rule]),
            memory_worthy=bool(data.get("memory_worthy", True)),
            clin_rule=rule,
            confidence=str(data.get("confidence") or "should"),
            compiler_source="model",
            compiler_model=self.ref_model,
        )

    def _compile_search_chain_with_model(
        self,
        trajectory: list[dict[str, Any]],
        task_instruction: str,
        prediction: str,
    ) -> Reflection | None:
        from openai import OpenAI

        client = OpenAI(base_url=normalize_openai_base_url(self.base_url), api_key=self.api_key)
        compact_chain = self._compact_search_chain(trajectory)
        if not compact_chain:
            return None
        response = client.chat.completions.create(
            model=self.ref_model,
            messages=[
                {"role": "system", "content": SEARCH_CHAIN_REFLECTION_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": task_instruction,
                            "prediction": prediction,
                            "search_chain": compact_chain,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            max_tokens=900,
            temperature=0.2,
            extra_body={**self.generation_config, "enable_thinking": False},
        )
        data = self._parse_search_chain_text(response.choices[0].message.content or "")
        useful_raw = data.get("useful", False)
        useful = (
            useful_raw.strip().lower() in {"1", "true", "yes"}
            if isinstance(useful_raw, str)
            else bool(useful_raw)
        )
        strategy_text = self._single_line(
            str(data.get("strategy") or data.get("content") or data.get("clin_rule") or data.get("rule") or "")
        )
        rule = self._search_strategy_rule(strategy_text)
        lesson_type = str(data.get("lesson_type") or "search_chain_lesson")
        query_templates = self._as_string_list(data.get("query_templates") or data.get("query_template") or [])
        keywords = self._as_string_list(data.get("prioritize_keywords") or [])
        strategy = self._as_string_list(data.get("correct_strategy") or data.get("strategy") or [])
        metadata = {
            "lesson_type": lesson_type,
            "title": str(data.get("title") or "").strip(),
            "trigger_pattern": str(data.get("trigger_pattern") or "").strip(),
            "trigger_keywords": self._as_string_list(data.get("trigger_keywords") or []),
            "query_templates": query_templates[:8],
            "prioritize_keywords": keywords[:12],
            "fact_slots": self._as_string_list(data.get("fact_slots") or [])[:8],
            "tool_sequence": self._as_string_list(data.get("tool_sequence") or [])[:6],
            "correct_strategy": strategy[:8],
            "avoid": self._as_string_list(data.get("avoid") or [])[:6],
            "content": strategy_text,
        }
        if not useful or not strategy_text:
            return Reflection(
                failure_type=lesson_type if lesson_type != "none" else "search_chain_no_lesson",
                root_cause=str(data.get("reason") or data.get("root_cause") or "No reusable search-chain lesson found."),
                correct_strategy=strategy,
                memory_worthy=False,
                clin_rule=rule,
                confidence=str(data.get("confidence") or "may"),
                compiler_source="search_chain_model",
                compiler_model=self.ref_model,
                metadata=metadata,
            )
        if query_templates:
            strategy.append("Prioritize query templates: " + "; ".join(query_templates[:4]))
        if keywords:
            strategy.append("Prioritize keywords: " + ", ".join(keywords[:8]))
        return Reflection(
            failure_type=lesson_type,
            root_cause=str(data.get("reason") or data.get("root_cause") or "Auxiliary reviewer found a reusable search-chain lesson."),
            correct_strategy=strategy or [strategy_text],
            memory_worthy=True,
            clin_rule=rule,
            confidence=str(data.get("confidence") or "should"),
            compiler_source="search_chain_model",
            compiler_model=self.ref_model,
            metadata=metadata,
        )

    def _compact_search_chain(self, trajectory: list[dict[str, Any]], limit: int = 48) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for row in trajectory:
            role = row.get("role")
            if role == "assistant":
                content = str(row.get("content") or "")
                reasoning = str(row.get("reasoning_content") or "")
                tool_calls = row.get("tool_calls") or []
                if not content and not reasoning and not tool_calls:
                    continue
                compact.append(
                    {
                        "step": row.get("step_id"),
                        "role": "assistant",
                        "content": self._truncate(content, 600),
                        "reasoning": self._truncate(reasoning, 900),
                        "tool_calls": tool_calls,
                    }
                )
            elif role == "tool":
                fn_name = str(row.get("fn_name") or "")
                if not (
                    fn_name.startswith("search_")
                    or fn_name.startswith("browser")
                    or "search" in fn_name
                    or "browser" in fn_name
                ):
                    continue
                compact.append(
                    {
                        "step": row.get("step_id"),
                        "role": "tool",
                        "tool": fn_name,
                        "args": row.get("fn_args") or {},
                        "content": self._truncate(str(row.get("content") or ""), 1400),
                        "ok": row.get("ok"),
                    }
                )
        return compact[-limit:]

    def _truncate(self, text: str, max_chars: int) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."

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
        data = self._parse_memory_update_text(response.choices[0].message.content or "")
        operation = str(data.get("operation", "ADD")).upper()
        if operation not in {"ADD", "EDIT", "IGNORE"}:
            operation = "ADD"
        return {
            "operation": operation,
            "target_id": data.get("target_id"),
            "rule": self._single_line(str(data.get("rule") or new_rule)),
            "reason": str(data.get("reason") or "model_advice"),
            "advisor_source": "model",
            "advisor_model": self.ref_model,
        }

    def _heuristic_memory_advice(
        self, new_rule: str, similar_entries: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if not similar_entries:
            return {"operation": "ADD", "rule": new_rule, "reason": "no_similar_rule", "advisor_source": "heuristic"}
        best = similar_entries[0]
        sim = float(best.get("similarity", 0.0))
        if sim >= 0.90:
            return {
                "operation": "IGNORE",
                "target_id": best.get("id"),
                "rule": best.get("rule", new_rule),
                "reason": f"near_duplicate_similarity_{sim:.2f}",
                "advisor_source": "heuristic",
            }
        if sim >= 0.72:
            return {
                "operation": "EDIT",
                "target_id": best.get("id"),
                "rule": self._prefer_general_rule(best.get("rule", ""), new_rule),
                "reason": f"merge_similar_rule_{sim:.2f}",
                "advisor_source": "heuristic",
            }
        return {"operation": "ADD", "rule": new_rule, "reason": f"weak_similarity_{sim:.2f}", "advisor_source": "heuristic"}

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
                "Use visible text, style, location, or subject features to build search_text follow-up queries.",
                "Verify that the candidate subject matches the image and the requested attribute.",
                "Reject text-search hits that only match the attribute but not the visual subject.",
            ]
            rule = (
                "[When VQA requires identifying a person/object/place before asking its attribute] -> "
                "Action: cross-check search_image candidates with visual features and the requested field before answering "
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
                "Compare the last observation against the unresolved fact slot before calling another tool.",
                "Stop equivalent calls and search only for the missing target field or a stronger candidate source.",
            ]
            rule = (
                "[When the last two observations do not add new evidence] -> "
                "Action: stop repeating equivalent calls and search only for unresolved fact slots is Necessary "
                "to achieve Goal G (Confidence: should)"
            )
        elif not re.search(r"<answer>.*?</answer>", final_content, flags=re.S | re.I):
            failure_type = "missing_strict_answer"
            root = "The run ended without the requested strict answer tag."
            strategy = [
                "After evidence is sufficient, emit exactly one short answer.",
                "If a structured final turn is requested, return only {\"final_answer\":\"...\"}.",
                "Ensure the answer is the field requested by the question.",
            ]
            rule = (
                "[When enough evidence has been observed] -> Action: emit only the requested short final field "
                "in the required answer format is Necessary to achieve Goal G "
                "(Confidence: should)"
            )
        else:
            failure_type = "missing_final_field_or_evidence_slot"
            root = "The trajectory did not close the requested fact slot before producing a concise final answer."
            strategy = [
                "Identify the requested final field before selecting a final span.",
                "For multi-hop questions, resolve the intermediate entity first, then query the final requested field.",
                "Return only the final field, not the evidence sentence or intermediate entity.",
            ]
            rule = (
                "[For multi-hop, visual, or comparison questions] -> Action: close the requested final field "
                "after resolving intermediate entities is Necessary "
                "to achieve Goal G (Confidence: should)"
            )

        return Reflection(
            failure_type=failure_type,
            root_cause=root,
            correct_strategy=strategy,
            memory_worthy=True,
            clin_rule=rule,
            compiler_source="heuristic",
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

    def _try_extract_json(self, text: str) -> dict[str, Any]:
        if "{" not in (text or ""):
            return {}
        try:
            return self._extract_json(text)
        except Exception:
            return {}

    def _parse_reflection_text(self, text: str) -> dict[str, Any]:
        data = self._try_extract_json(text)
        if data:
            return data
        fields = self._parse_field_block(text, multiline_keys={"STRATEGY"})
        strategy = self._parse_bullets(fields.get("STRATEGY", ""))
        memory_raw = fields.get("MEMORY", "YES").strip().lower()
        return {
            "failure_type": fields.get("TYPE", "model_diagnosed_failure"),
            "root_cause": fields.get("ROOT_CAUSE", "Auxiliary compiler generated a rule."),
            "correct_strategy": strategy,
            "memory_worthy": memory_raw in {"yes", "true", "1", "y"},
            "clin_rule": fields.get("RULE", ""),
            "confidence": fields.get("CONFIDENCE", "should"),
        }

    def _parse_search_chain_text(self, text: str) -> dict[str, Any]:
        data = self._try_extract_json(text)
        if data:
            return data
        fields = self._parse_field_block(text, multiline_keys={"STRATEGY"})
        return {
            "useful": fields.get("USEFUL", "NO").strip().lower() in {"yes", "true", "1", "y"},
            "lesson_type": fields.get("TYPE", "none"),
            "trigger_pattern": fields.get("TRIGGER", ""),
            "query_templates": self._split_items(fields.get("QUERY_TEMPLATE", "")),
            "prioritize_keywords": self._split_items(fields.get("KEYWORDS", "")),
            "fact_slots": self._split_items(fields.get("FACT_SLOTS", "")),
            "tool_sequence": self._split_items(fields.get("TOOL_SEQUENCE", "")),
            "root_cause": fields.get("ROOT_CAUSE", ""),
            "correct_strategy": self._parse_bullets(fields.get("STRATEGY", "")),
            "clin_rule": fields.get("RULE", ""),
            "confidence": fields.get("CONFIDENCE", "may"),
        }

    def _parse_memory_update_text(self, text: str) -> dict[str, Any]:
        data = self._try_extract_json(text)
        if data:
            return data
        fields = self._parse_field_block(text)
        return {
            "operation": fields.get("OPERATION", "ADD"),
            "target_id": fields.get("TARGET_ID", ""),
            "rule": fields.get("RULE", ""),
            "reason": fields.get("REASON", "model_advice"),
        }

    def _parse_field_block(
        self, text: str, multiline_keys: set[str] | None = None
    ) -> dict[str, str]:
        multiline_keys = multiline_keys or set()
        fields: dict[str, str] = {}
        current_key = ""
        for raw_line in (text or "").splitlines():
            line = raw_line.rstrip()
            match = re.match(r"^([A-Z_]+)\s*:\s*(.*)$", line)
            if match:
                key = match.group(1).upper()
                fields[key] = match.group(2).strip()
                current_key = key if key in multiline_keys else ""
                continue
            if current_key:
                fields[current_key] = (fields.get(current_key, "") + "\n" + line).strip()
        return fields

    def _parse_bullets(self, text: str) -> list[str]:
        items = []
        for line in (text or "").splitlines():
            cleaned = re.sub(r"^\s*[-*]\s*", "", line).strip()
            if cleaned:
                items.append(cleaned)
        return items

    def _split_items(self, text: str) -> list[str]:
        if not text:
            return []
        parts = re.split(r"[;,，；]|\s+->\s+", text)
        return [part.strip(" -\t\r\n") for part in parts if part.strip(" -\t\r\n")]

    def _single_line(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    def _search_strategy_rule(self, strategy: str) -> str:
        strategy = self._single_line(strategy)
        if not strategy:
            return ""
        if "->" in strategy and "Goal G" in strategy:
            return strategy
        return (
            "[When a similar search-chain pattern appears] -> "
            f"Action: {strategy} is Necessary to achieve Goal G (Confidence: should)"
        )

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

"""Case-level evidence memory and text-block strategy memory.

This module keeps the new Track-D style learning path separate from the
existing CLIN/SkepticalMemoryManager path:
- case memory records every retrieval observation in the current case;
- a bounded 32B reviewer only emits CAN/CANNOT answerability signals;
- cross-case strategy memory stores reusable text blocks, not factual answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from .config import normalize_openai_base_url
from .model_policy import assert_model_within_limit
from .prompts import (
    CANDIDATE_ANSWER_REVIEW_PROMPT,
    CASE_RETRY_CONTEXT_PROMPT,
    STRATEGY_SELECTION_PROMPT,
    STRATEGY_WRITE_PROMPT,
)


_FILE_LOCK = threading.RLock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _stable_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _compact_text(value: Any, max_chars: int = 1200) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", (text or "").lower()))


def _jaccard(a: str, b: str) -> float:
    ta = _tokens(a)
    tb = _tokens(b)
    return len(ta & tb) / max(len(ta | tb), 1)


@dataclass
class RetrievalRecord:
    step_id: int
    tool_name: str
    tool_args: dict[str, Any]
    ok: bool
    error: str = ""
    query: str = ""
    result_count: int = 0
    refs: list[str] = field(default_factory=list)
    summary: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "ok": self.ok,
            "error": self.error,
            "query": self.query,
            "result_count": self.result_count,
            "refs": self.refs,
            "summary": self.summary,
        }


@dataclass
class AnswerabilityDecision:
    can_answer: bool
    confidence: str = "low"
    reason: str = ""
    missing_slots: list[str] = field(default_factory=list)
    source: str = "heuristic"
    model: str = ""
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "can_answer": self.can_answer,
            "confidence": self.confidence,
            "reason": self.reason,
            "missing_slots": self.missing_slots,
            "source": self.source,
            "model": self.model,
            "error": self.error,
        }


class StrategyBlockStore:
    """Append-only JSONL store for cross-case reusable strategy text blocks."""

    def __init__(self, path: str, max_blocks: int = 96) -> None:
        self.path = Path(path)
        self.max_blocks = max_blocks

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("text_block"):
                rows.append(row)
        return rows

    def upsert(self, block: dict[str, Any]) -> dict[str, Any]:
        text_block = _compact_text(block.get("text_block") or "", 2200)
        if not self._valid_text_block(text_block):
            return {"operation": "IGNORE", "reason": "invalid_or_answer_like_strategy_block"}

        now = _now_iso()
        block = dict(block)
        block["text_block"] = text_block
        block.setdefault("strategy_type", "search_strategy")
        block.setdefault("task_type", "general")
        block.setdefault("title", self._title_from_block(text_block))
        block.setdefault("trigger", "")
        block.setdefault("tags", [])
        block.setdefault("created_at", now)
        block["updated_at"] = now
        block["strategy_id"] = block.get("strategy_id") or _stable_id(
            "strategy", f"{block.get('task_type')}:{block.get('trigger')}:{text_block}"
        )

        with _FILE_LOCK:
            rows = self.load()
            best_idx = -1
            best_sim = 0.0
            for idx, row in enumerate(rows):
                sim = max(
                    _jaccard(text_block, str(row.get("text_block") or "")),
                    _jaccard(str(block.get("trigger") or ""), str(row.get("trigger") or "")),
                )
                if sim > best_sim:
                    best_idx = idx
                    best_sim = sim
            if best_sim >= 0.88 and best_idx >= 0:
                existing = rows[best_idx]
                existing["updated_at"] = now
                existing["merge_count"] = int(existing.get("merge_count", 0)) + 1
                existing["source_case_ids"] = sorted(
                    set(existing.get("source_case_ids") or []) | set(block.get("source_case_ids") or [])
                )[:24]
                existing["last_reason"] = block.get("reason", "")
                rows[best_idx] = existing
                self._write_rows(rows)
                return {
                    "operation": "EDIT",
                    "strategy_id": existing.get("strategy_id"),
                    "reason": f"merged_similar_strategy_{best_sim:.2f}",
                    "strategy": existing,
                }

            rows.append(block)
            rows = self._pruned(rows)
            self._write_rows(rows)
            return {
                "operation": "ADD",
                "strategy_id": block["strategy_id"],
                "reason": block.get("reason", "new_strategy_block"),
                "strategy": block,
            }

    def lexical_select(self, instruction: str, task_type: str, top_k: int = 3) -> list[dict[str, Any]]:
        rows = [row for row in self.load() if row.get("status", "active") != "deprecated"]
        scored = []
        query = f"{task_type} {instruction}"
        for row in rows:
            haystack = " ".join(
                str(row.get(field) or "")
                for field in ("task_type", "title", "trigger", "text_block", "strategy_type")
            )
            score = _jaccard(query, haystack)
            if row.get("task_type") == task_type:
                score += 0.10
            if score <= 0:
                continue
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _score, row in scored[:top_k]]

    def _write_rows(self, rows: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)

    def _pruned(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.max_blocks <= 0 or len(rows) <= self.max_blocks:
            return rows
        rows.sort(
            key=lambda row: (
                int(row.get("priority", 0)),
                int(row.get("merge_count", 0)),
                str(row.get("updated_at", "")),
            ),
            reverse=True,
        )
        return rows[: self.max_blocks]

    def _title_from_block(self, text_block: str) -> str:
        first = re.split(r"[\n。.!?]", text_block.strip(), maxsplit=1)[0]
        return first[:80] or "Reusable retrieval strategy"

    def _valid_text_block(self, text_block: str) -> bool:
        if len(text_block) < 40 or len(text_block) > 2200:
            return False
        low = text_block.lower()
        answer_like = (
            "the answer is",
            "答案是",
            "final answer",
            "<answer>",
            "{\"final_answer\"",
            "gold answer",
            "正确答案",
        )
        if any(marker in low for marker in answer_like):
            return False
        reusable_markers = (
            "when ",
            "trigger",
            "strategy",
            "优先",
            "当",
            "检索",
            "query",
            "search",
            "fact slot",
            "事实槽",
        )
        return any(marker in low for marker in reusable_markers)


class CaseMemoryManager:
    """32B-assisted case memory and strategy memory facade."""

    def __init__(
        self,
        *,
        enabled: bool,
        enable_model: bool,
        base_url: str,
        model_name: str,
        api_key: str,
        max_model_billion: float,
        case_log_path: str,
        strategy_path: str,
        strategy_top_k: int = 3,
        max_records_prompt: int = 12,
        late_search_threshold: int = 3,
        max_strategy_blocks: int = 96,
    ) -> None:
        self.enabled = enabled
        self.enable_model = enable_model and bool(base_url)
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key
        self.case_log_path = Path(case_log_path)
        self.strategy_store = StrategyBlockStore(strategy_path, max_blocks=max_strategy_blocks)
        self.strategy_top_k = strategy_top_k
        self.max_records_prompt = max_records_prompt
        self.late_search_threshold = late_search_threshold
        self.client = None
        self.generation_config = {
            "enable_thinking": False,
            "temperature": 0.1,
            "top_p": 0.9,
        }
        if self.enable_model:
            assert_model_within_limit(model_name, max_model_billion)

    # ------------------------------------------------------------------
    # Retrieval records
    # ------------------------------------------------------------------
    def build_record(
        self,
        *,
        step_id: int,
        tool_name: str,
        tool_args: dict[str, Any],
        raw_result: Any,
        error: str,
    ) -> RetrievalRecord:
        entries = self._result_entries(raw_result)
        refs = []
        snippets = []
        for entry in entries[:5]:
            title = str(entry.get("title") or entry.get("name") or "").strip()
            url = str(entry.get("url") or entry.get("link") or "").strip()
            snippet = str(entry.get("snippet") or entry.get("content") or entry.get("text") or "").strip()
            if url:
                refs.append(url)
            elif title:
                refs.append(title)
            if title or snippet:
                snippets.append(_compact_text(f"{title} {snippet}", 280))
        query = str(
            tool_args.get("query")
            or tool_args.get("image_url")
            or tool_args.get("image")
            or tool_args.get("url")
            or ""
        )
        summary = "; ".join(snippets) if snippets else _compact_text(raw_result, 800)
        return RetrievalRecord(
            step_id=step_id,
            tool_name=tool_name,
            tool_args=dict(tool_args),
            ok=not bool(error),
            error=error,
            query=query,
            result_count=len(entries),
            refs=refs[:8],
            summary=summary,
        )

    def persist_case(
        self,
        *,
        task_id: str,
        task_type: str,
        instruction: str,
        records: list[RetrievalRecord],
        decisions: list[dict[str, Any]],
        pred: str,
        eval_status: str,
        exit_status: str,
        failure_reason: str,
    ) -> dict[str, Any]:
        if not self.enabled or not records:
            return {"written": False, "reason": "disabled_or_no_records"}
        row = {
            "timestamp": _now_iso(),
            "task_id": task_id,
            "task_type": task_type,
            "instruction": instruction[:1500],
            "pred": pred[:500],
            "eval_status": eval_status,
            "exit_status": exit_status,
            "failure_reason": failure_reason[:800],
            "records": [record.as_dict() for record in records],
            "answerability_decisions": decisions,
        }
        with _FILE_LOCK:
            self.case_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.case_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return {"written": True, "path": str(self.case_log_path), "records": len(records)}

    def review_candidate_answer(
        self,
        *,
        instruction: str,
        task_type: str,
        records: list[RetrievalRecord],
        candidate_answer: str,
        issue_type: str,
        applied_memory_context: list[str] | None = None,
    ) -> dict[str, Any]:
        """Review only when 9B stops searching; never emits or edits the answer."""
        if not self.enabled:
            return {
                "accept": True,
                "issue_type": "none",
                "retry_memory_block": "",
                "missing_slots": [],
                "source": "disabled",
            }
        if not self.enable_model:
            return self._heuristic_candidate_review(
                task_type=task_type,
                records=records,
                candidate_answer=candidate_answer,
                issue_type=issue_type,
                source="heuristic",
            )
        try:
            content = self._chat_text(
                system=CANDIDATE_ANSWER_REVIEW_PROMPT,
                payload={
                    "task": instruction,
                    "task_type": task_type,
                    "candidate_answer": _compact_text(candidate_answer, 900),
                    "local_issue_signal": issue_type,
                    "applied_memory_context": [
                        _compact_text(item, 600) for item in (applied_memory_context or [])[:8]
                    ],
                    "retrieval_records": self._compact_records(records),
                    "instruction": "Return only whether the candidate is submit-ready and supported.",
                },
                max_tokens=120,
            )
            data = self._parse_candidate_answer_review_text(content, issue_type)
            accept = self._as_bool(data.get("accept"))
            return {
                "accept": accept,
                "can_answer": accept,
                "confidence": str(data.get("confidence") or ("medium" if accept else "low")),
                "issue_type": str(data.get("issue_type") or ("none" if accept else issue_type)),
                "missing_slots": self._as_string_list(data.get("missing_slots") or [])[:6],
                "retry_memory_block": "",
                "reason": _compact_text(data.get("reason") or "", 300),
                "source": "model",
                "model": self.model_name,
            }
        except Exception as exc:
            review = self._heuristic_candidate_review(
                task_type=task_type,
                records=records,
                candidate_answer=candidate_answer,
                issue_type=issue_type,
                source="heuristic_fallback",
            )
            review["model"] = self.model_name
            review["error"] = f"{type(exc).__name__}: {exc}"
            return review

    def build_retry_context(
        self,
        *,
        instruction: str,
        task_type: str,
        records: list[RetrievalRecord],
        candidate_answer: str,
        issue_type: str,
        applied_memory_context: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compress observations for a retry without giving directional advice."""
        if not self.enabled:
            return {"context": "", "source": "disabled"}
        if not self.enable_model:
            return {
                "context": self._heuristic_observation_context(records),
                "source": "heuristic",
            }
        try:
            content = self._chat_text(
                system=CASE_RETRY_CONTEXT_PROMPT,
                payload={
                    "task": instruction,
                    "task_type": task_type,
                    "candidate_answer": _compact_text(candidate_answer, 900),
                    "local_issue_signal": issue_type,
                    "applied_memory_context": [
                        _compact_text(item, 600) for item in (applied_memory_context or [])[:8]
                    ],
                    "retrieval_records": self._compact_records(records),
                    "instruction": (
                        "Compress only observed information. Do not give next-step search advice, "
                        "missing-slot hints, final answers, or answer extraction instructions."
                    ),
                },
                max_tokens=600,
            )
            data = self._parse_retry_context_text(content)
            return {
                "context": _compact_text(data.get("context") or "", 1200),
                "source": "model",
                "model": self.model_name,
            }
        except Exception as exc:
            return {
                "context": self._heuristic_observation_context(records),
                "source": "heuristic_fallback",
                "model": self.model_name,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _heuristic_candidate_review(
        self,
        *,
        task_type: str,
        records: list[RetrievalRecord],
        candidate_answer: str,
        issue_type: str,
        source: str,
    ) -> dict[str, Any]:
        answer = (candidate_answer or "").strip()
        if issue_type in {"no_answer", "tool_call_leak", "uncertain_answer", "overbroad_answer", "wrong_format"}:
            accept = False
        elif not answer:
            issue_type = "no_answer"
            accept = False
        elif len(answer) > 180:
            issue_type = "overbroad_answer"
            accept = False
        else:
            accept = True
        if accept:
            return {
                "accept": True,
                "can_answer": True,
                "confidence": "low",
                "issue_type": "none",
                "missing_slots": [],
                "retry_memory_block": "",
                "reason": "heuristic_submit_ready_candidate",
                "source": source,
            }
        latest = records[-1] if records else RetrievalRecord(0, "", {}, False, error="no_records")
        decision = self._heuristic_answerability(records, latest)
        return {
            "accept": False,
            "can_answer": decision.can_answer,
            "confidence": decision.confidence,
            "issue_type": issue_type,
            "missing_slots": decision.missing_slots,
            "retry_memory_block": "",
            "reason": decision.reason,
            "source": source,
        }

    def _heuristic_answerability(
        self, records: list[RetrievalRecord], latest: RetrievalRecord
    ) -> AnswerabilityDecision:
        if not latest.ok:
            return AnswerabilityDecision(False, reason="latest_tool_error")
        evidence_text = " ".join(record.summary for record in records[-3:])
        has_named_or_numeric = bool(
            re.search(r"\b[A-Z][A-Za-z0-9'&.-]+(?:\s+[A-Z][A-Za-z0-9'&.-]+){0,5}\b|\d{3,4}", evidence_text)
        )
        enough = len(records) >= 1 and latest.result_count > 0 and has_named_or_numeric
        return AnswerabilityDecision(
            enough,
            confidence="low" if enough else "low",
            reason="heuristic_named_or_numeric_evidence" if enough else "need_more_evidence",
        )

    # ------------------------------------------------------------------
    # Cross-case strategy selection and writing
    # ------------------------------------------------------------------
    def select_strategies(
        self,
        *,
        instruction: str,
        task_type: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"selected": [], "source": "disabled"}
        top_k = self.strategy_top_k if top_k is None else top_k
        candidates = self.strategy_store.load()
        if not candidates or top_k <= 0:
            return {"selected": [], "source": "empty"}
        candidates = candidates[-40:]
        if not self.enable_model:
            selected = self.strategy_store.lexical_select(instruction, task_type, top_k=top_k)
            return {"selected": selected, "source": "lexical"}
        try:
            compact_candidates = [
                {
                    "strategy_id": row.get("strategy_id"),
                    "task_type": row.get("task_type"),
                    "strategy_type": row.get("strategy_type"),
                    "title": row.get("title"),
                    "trigger": row.get("trigger"),
                    "text_block": _compact_text(row.get("text_block") or "", 900),
                }
                for row in candidates
            ]
            content = self._chat_text(
                system=STRATEGY_SELECTION_PROMPT,
                payload={
                    "task": instruction,
                    "task_type": task_type,
                    "max_select": top_k,
                    "candidate_strategies": compact_candidates,
                },
                max_tokens=600,
            )
            data = self._parse_strategy_selection_text(content)
            selected_ids = {
                str(item)
                for item in self._as_string_list(data.get("selected_strategy_ids") or data.get("selected_ids") or [])
            }
            by_id = {str(row.get("strategy_id")): row for row in candidates}
            selected = [by_id[item] for item in selected_ids if item in by_id][:top_k]
            if not selected:
                selected = self.strategy_store.lexical_select(instruction, task_type, top_k=top_k)
            return {
                "selected": selected,
                "source": "model",
                "reason": _compact_text(data.get("reason") or "", 300),
                "model": self.model_name,
            }
        except Exception as exc:
            selected = self.strategy_store.lexical_select(instruction, task_type, top_k=top_k)
            return {
                "selected": selected,
                "source": "lexical_fallback",
                "error": f"{type(exc).__name__}: {exc}",
                "model": self.model_name,
            }

    def maybe_write_strategy(
        self,
        *,
        task_id: str,
        instruction: str,
        task_type: str,
        records: list[RetrievalRecord],
        decisions: list[dict[str, Any]],
        pred: str,
        eval_status: str,
        exit_status: str,
        failure_reason: str,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"operation": "IGNORE", "reason": "disabled"}
        if not records:
            return {"operation": "IGNORE", "reason": "no_retrieval_records"}
        if self._looks_accidental_failure(exit_status, failure_reason):
            return {"operation": "IGNORE", "reason": "accidental_failure_use_reflection_fallback"}
        if eval_status == "incorrect" or not pred:
            return {"operation": "IGNORE", "reason": "not_a_successful_or_unknown_success_case"}

        if not self.enable_model:
            candidate = self._heuristic_strategy_candidate(
                task_id=task_id,
                instruction=instruction,
                task_type=task_type,
                records=records,
                decisions=decisions,
            )
        else:
            try:
                content = self._chat_text(
                    system=STRATEGY_WRITE_PROMPT,
                    payload={
                        "task": instruction,
                        "task_type": task_type,
                        "prediction_present": bool(pred),
                        "eval_status": eval_status,
                        "exit_status": exit_status,
                        "retrieval_records": self._compact_records(records),
                        "answerability_decisions": decisions[-8:],
                        "signals": {
                            "late_search_threshold": self.late_search_threshold,
                            "record_count": len(records),
                            "first_can_answer_record_index": self._first_can_answer_index(decisions),
                        },
                    },
                    max_tokens=900,
                )
                data = self._parse_strategy_write_text(content)
                candidate = self._candidate_from_model_data(task_id, task_type, data)
            except Exception as exc:
                candidate = self._heuristic_strategy_candidate(
                    task_id=task_id,
                    instruction=instruction,
                    task_type=task_type,
                    records=records,
                    decisions=decisions,
                )
                candidate["review_error"] = f"{type(exc).__name__}: {exc}"
                candidate["review_source"] = "heuristic_fallback"

        if not candidate.get("should_write"):
            return {
                "operation": "IGNORE",
                "reason": candidate.get("reason", "reviewer_declined"),
                "review": candidate,
            }
        block = {
            "strategy_type": candidate.get("strategy_type") or "search_strategy",
            "task_type": task_type,
            "title": candidate.get("title") or "Reusable search strategy",
            "trigger": candidate.get("trigger") or candidate.get("trigger_pattern") or "",
            "tags": self._as_string_list(candidate.get("tags") or []),
            "text_block": candidate.get("text_block") or "",
            "priority": int(candidate.get("priority", 1) or 1),
            "source_case_ids": [task_id],
            "reason": candidate.get("reason", ""),
            "review_source": candidate.get("review_source", "model" if self.enable_model else "heuristic"),
            "review_model": self.model_name if self.enable_model else "",
        }
        result = self.strategy_store.upsert(block)
        result["review"] = candidate
        result["strategy_path"] = str(self.strategy_store.path)
        return result

    def format_strategies_for_prompt(self, strategies: list[dict[str, Any]]) -> list[str]:
        formatted = []
        for row in strategies:
            title = row.get("title") or "Reusable strategy"
            trigger = row.get("trigger") or ""
            text = row.get("text_block") or ""
            formatted.append(
                "\n".join(
                    part
                    for part in (
                        f"[Strategy Memory: {title}]",
                        f"Trigger: {trigger}" if trigger else "",
                        "Use this as a retrieval/reasoning strategy only, not as evidence or an answer.",
                        text,
                    )
                    if part
                )
            )
        return formatted

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------
    def _chat_text(self, *, system: str, payload: dict[str, Any], max_tokens: int) -> str:
        client = self._client()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            max_tokens=max_tokens,
            temperature=0.1,
            extra_body=self.generation_config,
        )
        return response.choices[0].message.content or ""

    def _client(self):
        if self.client is not None:
            return self.client
        from openai import OpenAI

        self.client = OpenAI(
            base_url=normalize_openai_base_url(self.base_url),
            api_key=self.api_key,
        )
        return self.client

    def _extract_json(self, text: str) -> dict[str, Any]:
        text = (text or "").strip()
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

    def _parse_answerability_text(self, text: str) -> dict[str, Any]:
        data = self._try_extract_json(text)
        if data:
            return data
        fields = self._parse_field_block(text)
        first = self._first_nonempty_line(text).upper()
        return {
            "can_answer": first.startswith("CAN") and not first.startswith("CANNOT"),
            "confidence": fields.get("CONFIDENCE", "low"),
            "missing_slots": self._split_items(fields.get("MISSING", "")),
            "reason": fields.get("REASON", ""),
        }

    def _parse_answer_issue_text(self, text: str, fallback_issue: str) -> dict[str, Any]:
        data = self._try_extract_json(text)
        if data:
            return data
        fields = self._parse_field_block(text)
        first = self._first_nonempty_line(text).upper()
        return {
            "can_answer": first.startswith("CAN") and not first.startswith("CANNOT"),
            "confidence": fields.get("CONFIDENCE", "low"),
            "issue_type": fields.get("ISSUE", fallback_issue),
            "missing_slots": self._split_items(fields.get("MISSING", "")),
            "retry_memory_block": fields.get("RETRY", ""),
            "reason": fields.get("REASON", ""),
        }

    def _parse_candidate_answer_review_text(self, text: str, fallback_issue: str) -> dict[str, Any]:
        data = self._try_extract_json(text)
        if data:
            if "accept" not in data and "pass" in data:
                data["accept"] = data.get("pass")
            return data
        fields = self._parse_field_block(text)
        first = self._first_nonempty_line(text).upper()
        return {
            "accept": first.startswith("ACCEPT"),
            "confidence": fields.get("CONFIDENCE", "low"),
            "issue_type": fields.get("ISSUE", fallback_issue),
            "missing_slots": self._split_items(fields.get("MISSING", "")),
            "retry_memory_block": fields.get("MEMORY", ""),
            "reason": fields.get("REASON", ""),
        }

    def _parse_retry_context_text(self, text: str) -> dict[str, Any]:
        data = self._try_extract_json(text)
        if data:
            return data
        fields = self._parse_field_block(text, multiline_keys={"CONTEXT"})
        return {"context": fields.get("CONTEXT") or text.strip()}

    def _parse_strategy_selection_text(self, text: str) -> dict[str, Any]:
        data = self._try_extract_json(text)
        if data:
            return data
        fields = self._parse_field_block(text)
        selected = [
            item
            for item in self._split_items(fields.get("SELECT", ""))
            if item.lower() != "none"
        ]
        return {
            "selected_strategy_ids": selected,
            "reason": fields.get("REASON", ""),
        }

    def _parse_strategy_write_text(self, text: str) -> dict[str, Any]:
        data = self._try_extract_json(text)
        if data:
            return data
        fields = self._parse_field_block(text, multiline_keys={"TEXT_BLOCK"})
        first = self._first_nonempty_line(text).upper()
        should_write = first.startswith("ADD")
        priority_text = fields.get("PRIORITY", "1")
        try:
            priority = int(re.search(r"\d+", priority_text).group(0)) if re.search(r"\d+", priority_text) else 1
        except ValueError:
            priority = 1
        return {
            "should_write": should_write,
            "strategy_type": fields.get("TYPE", "none"),
            "title": fields.get("TITLE", ""),
            "trigger": fields.get("TRIGGER", ""),
            "tags": self._split_items(fields.get("TAGS", "")),
            "priority": priority,
            "reason": fields.get("REASON", ""),
            "text_block": fields.get("TEXT_BLOCK", ""),
        }

    def _try_extract_json(self, text: str) -> dict[str, Any]:
        if "{" not in (text or ""):
            return {}
        try:
            return self._extract_json(text)
        except Exception:
            return {}

    def _first_nonempty_line(self, text: str) -> str:
        for line in (text or "").splitlines():
            if line.strip():
                return line.strip()
        return ""

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
                value = match.group(2).strip()
                fields[key] = value
                current_key = key if key in multiline_keys else ""
                continue
            if current_key:
                fields[current_key] = (fields.get(current_key, "") + "\n" + line).strip()
        return fields

    def _split_items(self, text: str) -> list[str]:
        if not text:
            return []
        parts = re.split(r"[;,，；]|\s+->\s+", text)
        return [part.strip(" -\t\r\n") for part in parts if part.strip(" -\t\r\n")]

    def _compact_records(self, records: list[RetrievalRecord]) -> list[dict[str, Any]]:
        return [
            {
                **record.as_dict(),
                "summary": _compact_text(record.summary, 700),
            }
            for record in records[-self.max_records_prompt :]
        ]

    def _result_entries(self, raw_result: Any) -> list[dict[str, Any]]:
        if isinstance(raw_result, list):
            return [item for item in raw_result if isinstance(item, dict)]
        if isinstance(raw_result, dict):
            if isinstance(raw_result.get("results"), list):
                return [item for item in raw_result["results"] if isinstance(item, dict)]
            if isinstance(raw_result.get("recovery_results"), list):
                entries: list[dict[str, Any]] = []
                for recovery in raw_result["recovery_results"]:
                    if not isinstance(recovery, dict):
                        continue
                    entries.extend(self._result_entries(recovery.get("result")))
                return entries
        return []

    def _looks_accidental_failure(self, exit_status: str, failure_reason: str) -> bool:
        text = f"{exit_status} {failure_reason}".lower()
        markers = (
            "llm_call_failed",
            "tool_dispatch_failed",
            "gate_blocked",
            "proxy-error",
            "timeout",
            "timed out",
            "auth",
            "permission",
            "search_budget_exhausted",
            "tool_budget_exhausted",
            "uncaught",
        )
        return any(marker in text for marker in markers)

    def _heuristic_strategy_candidate(
        self,
        *,
        task_id: str,
        instruction: str,
        task_type: str,
        records: list[RetrievalRecord],
        decisions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        first_can = self._first_can_answer_index(decisions)
        successful_early = 0 < len(records) <= 2
        late = first_can >= self.late_search_threshold or len(records) >= self.late_search_threshold + 1
        if not late and not successful_early:
            return {"should_write": False, "reason": "no_late_or_strong_success_signal"}
        queries = [record.query for record in records if record.query][:5]
        first_good = records[min(max(first_can - 1, 0), len(records) - 1)] if first_can > 0 else records[-1]
        strategy_type = "late_decisive_query" if late else "successful_reasoning"
        text_block = (
            f"When a {task_type} task has similar wording or evidence needs, prioritize the query pattern "
            f"that first produced answerable evidence instead of repeating broad searches. "
            f"Early query candidates should combine the rare entity/visible anchor with the requested fact slot. "
            f"Observed useful tool: {first_good.tool_name}; useful query shape: {self._placeholder_query(first_good.query)}. "
            f"Avoid repeating earlier broad query order: {' -> '.join(self._placeholder_query(q) for q in queries[:3])}."
        )
        return {
            "should_write": True,
            "strategy_type": strategy_type,
            "title": "Prioritize decisive query pattern" if late else "Reuse concise successful retrieval pattern",
            "trigger": f"{task_type} task requiring search evidence for a requested fact slot",
            "tags": [task_type, strategy_type],
            "text_block": text_block,
            "priority": 2 if late else 1,
            "reason": f"heuristic_{strategy_type}",
            "source_case_ids": [task_id],
            "review_source": "heuristic",
        }

    def _candidate_from_model_data(self, task_id: str, task_type: str, data: dict[str, Any]) -> dict[str, Any]:
        should = self._as_bool(data.get("should_write") or data.get("useful"))
        text_block = _compact_text(data.get("strategy") or data.get("text_block") or "", 2200)
        return {
            "should_write": should,
            "strategy_type": str(data.get("strategy_type") or "search_strategy"),
            "title": str(data.get("title") or self.strategy_store._title_from_block(text_block))[:100],
            "trigger": str(data.get("trigger") or data.get("trigger_pattern") or "")[:500],
            "tags": self._as_string_list(data.get("tags") or [task_type])[:12],
            "text_block": text_block,
            "priority": int(data.get("priority") or 1),
            "reason": str(data.get("reason") or "model_review"),
            "source_case_ids": [task_id],
            "review_source": "model",
        }

    def _first_can_answer_index(self, decisions: list[dict[str, Any]]) -> int:
        for idx, decision in enumerate(decisions, start=1):
            payload = decision.get("decision") if isinstance(decision.get("decision"), dict) else decision
            if payload and self._as_bool(payload.get("can_answer")):
                return idx
        return 0

    def _placeholder_query(self, query: str) -> str:
        query = re.sub(r"https?://\S+", "<url>", query or "")
        query = re.sub(r"\b\d{4}\b", "<year>", query)
        query = re.sub(r"\b[A-Z][A-Za-z0-9'&.-]+(?:\s+[A-Z][A-Za-z0-9'&.-]+){1,6}\b", "<rare entity>", query)
        query = re.sub(r"\s+", " ", query).strip()
        return query[:120] or "<current evidence anchor> <requested field>"

    def _heuristic_observation_context(self, records: list[RetrievalRecord]) -> str:
        if not records:
            return "当前没有可复用的检索观察记录。"
        chunks = []
        for record in records[-5:]:
            if record.ok:
                query_part = f"query={record.query}" if record.query else "query=none"
                summary = _compact_text(record.summary, 240)
                chunks.append(
                    f"{record.tool_name}({query_part}) 返回 {record.result_count} 条结果；观察摘要：{summary or '无摘要'}。"
                )
            else:
                chunks.append(f"{record.tool_name} 调用失败；错误：{_compact_text(record.error, 180)}。")
        return " ".join(chunks)

    def _as_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value or "").strip().lower() in {"1", "true", "yes", "y", "can", "可以"}

    def _as_string_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if value:
            return [str(value).strip()]
        return []

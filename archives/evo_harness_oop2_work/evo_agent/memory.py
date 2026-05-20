"""Skeptical long-term memory with typed schema and hybrid retrieval."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ACTIVE_MEMORY_TYPES = {"skill", "bad_pattern", "reflection"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", (text or "").lower()))


def _jaccard(a: str, b: str) -> float:
    ta = _tokens(a)
    tb = _tokens(b)
    return len(ta & tb) / max(len(ta | tb), 1)


def _stable_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value:
        return [str(value).strip()]
    return []


MemoryAdvisor = Callable[[str, list[dict[str, Any]], str], dict[str, Any]]


class SkepticalMemoryManager:
    """Natural-language memory with typed schema, RRF retrieval, and feedback.

    Storage stays JSON for the current lightweight harness, but entries follow
    the formal long-term memory schema. Dense retrieval is optional: set
    MEMORY_EMBEDDING_ENABLED=1 plus EMBEDDING_BASE_URL/EMBEDDING_MODEL to call
    an OpenAI-compatible embedding endpoint. Without that, a deterministic
    hashing vector keeps the same interface usable offline.
    """

    def __init__(
        self,
        db_path: str = "memory.json",
        max_rules: int = 64,
        usage_log_path: str | None = None,
        backup_path: str | None = None,
    ):
        self.db_path = Path(db_path)
        self.max_rules = max_rules
        self.usage_log_path = Path(usage_log_path) if usage_log_path else self.db_path.with_name("usage_logs.jsonl")
        self.backup_path = Path(backup_path) if backup_path else self.db_path.with_name("memory_backup.jsonl")
        self.embedding_dim = int(os.getenv("MEMORY_EMBEDDING_DIM", "384"))
        self.embedding_enabled = os.getenv("MEMORY_EMBEDDING_ENABLED", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.embedding_base_url = os.getenv("EMBEDDING_BASE_URL", os.getenv("LLM_BASE_URL", ""))
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.retrieve_threshold = float(os.getenv("MEMORY_RETRIEVE_THRESHOLD", "0.45"))
        self.memory_db: list[dict[str, Any]] = self._load_db()

    # ------------------------------------------------------------------
    # Persistence and normalization
    # ------------------------------------------------------------------
    def _load_db(self) -> list[dict[str, Any]]:
        if not self.db_path.exists():
            return []
        try:
            data = json.loads(self.db_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if isinstance(data, dict):
            data = data.get("memories", [])
        if not isinstance(data, list):
            return []
        normalized = [self._normalize_entry(item) for item in data if isinstance(item, dict)]
        return [item for item in normalized if item.get("content") or item.get("rule")]

    def _save_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.db_path.with_suffix(self.db_path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.memory_db, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self.db_path)

    def _append_backup(self, entry: dict[str, Any], operation: str) -> None:
        self.backup_path.parent.mkdir(parents=True, exist_ok=True)
        row = {"operation": operation, "timestamp": _now_iso(), "memory": entry}
        with open(self.backup_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _normalize_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        if "memory_id" not in entry:
            entry = self._legacy_rule_to_memory(entry)
        now = _now_iso()
        entry.setdefault("memory_type", "reflection")
        if entry["memory_type"] not in ACTIVE_MEMORY_TYPES:
            entry["memory_type"] = "reflection"
        entry.setdefault("task_type", "general")
        entry.setdefault("title", self._title_from_content(entry.get("content") or entry.get("rule", "")))
        entry.setdefault("trigger_pattern", entry.get("title", "General reusable task-solving pattern."))
        entry["trigger_keywords"] = _as_list(entry.get("trigger_keywords") or entry.get("keywords"))[:80]
        entry.setdefault("content", entry.get("rule", ""))
        entry["procedure"] = _as_list(entry.get("procedure"))
        entry["avoid"] = _as_list(entry.get("avoid"))
        entry["example_patterns"] = _as_list(entry.get("example_patterns"))
        entry.setdefault("source", {"source_type": entry.get("source", "reflection"), "source_episode_ids": []})
        if not isinstance(entry.get("source"), dict):
            entry["source"] = {"source_type": str(entry.get("source")), "source_episode_ids": []}
        entry.setdefault("stats", {})
        stats = entry["stats"] if isinstance(entry["stats"], dict) else {}
        if "up" in entry or "down" in entry:
            stats.setdefault("success_count", int(entry.get("up", 0)))
            stats.setdefault("failure_count", int(entry.get("down", 0)))
        stats.setdefault("usage_count", int(entry.get("usage_count", 0)))
        stats.setdefault("success_count", int(entry.get("success_count", 0)))
        stats.setdefault("failure_count", int(entry.get("failure_count", 0)))
        stats["confidence"] = self._confidence(stats)
        entry["stats"] = stats
        entry.setdefault("status", "active")
        entry.setdefault("created_at", entry.get("created_at") or now)
        entry.setdefault("updated_at", entry.get("updated_at") or now)
        entry.setdefault("last_used_at", entry.get("last_used_at"))
        entry.setdefault("embedding_text", self.build_embedding_text(entry))
        entry["keywords"] = entry["trigger_keywords"]
        entry["rule"] = self._rule_alias(entry)
        entry["id"] = entry["memory_id"]
        entry["weight"] = stats["confidence"]
        return entry

    def _legacy_rule_to_memory(self, entry: dict[str, Any]) -> dict[str, Any]:
        rule = str(entry.get("rule") or entry.get("content") or "").strip()
        keywords = _as_list(entry.get("keywords")) or sorted(_tokens(rule))[:20]
        stats = {
            "usage_count": int(entry.get("usage_count", 0)),
            "success_count": int(entry.get("up", entry.get("success_count", 0))),
            "failure_count": int(entry.get("down", entry.get("failure_count", 0))),
            "confidence": float(entry.get("weight", 0.5)),
        }
        return {
            "memory_id": entry.get("id") or _stable_id("reflect", rule),
            "memory_type": "reflection",
            "task_type": "general",
            "title": self._title_from_content(rule),
            "trigger_pattern": self._trigger_from_rule(rule),
            "trigger_keywords": keywords,
            "content": rule,
            "procedure": self._procedure_from_rule(rule),
            "avoid": self._avoid_from_rule(rule),
            "example_patterns": [],
            "failure_type": entry.get("failure_type"),
            "source": {"source_type": str(entry.get("source", "legacy")), "source_episode_ids": []},
            "stats": stats,
            "status": "active",
            "created_at": entry.get("created_at") or _now_iso(),
            "updated_at": entry.get("updated_at") or _now_iso(),
            "last_used_at": entry.get("last_used_at"),
        }

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve_top_rules(
        self, instruction: str, task_type: str = "general", top_k: int = 3
    ) -> list[str]:
        """Return prompt-ready memory snippets for the planner/executor."""
        return self.format_memories_for_prompt(
            self.retrieve_memories(instruction, task_type=task_type, top_k=top_k)
        )

    def retrieve_memories(
        self, instruction: str, task_type: str = "general", top_k: int = 3
    ) -> list[dict[str, Any]]:
        """Hybrid dense/lexical retrieval with skeptical reranking."""
        query_text = self.build_query_text(instruction, task_type)
        active = [
            entry
            for entry in self.memory_db
            if entry.get("status", "active") == "active"
            and float(entry.get("stats", {}).get("confidence", 0.5)) >= 0.30
        ]
        if not active:
            return []

        query_vec = self.get_embedding(query_text)
        dense_rank = self._rank_dense(active, query_vec)
        lexical_rank = self._rank_lexical(active, instruction)
        rrf = self._reciprocal_rank_fusion([dense_rank, lexical_rank])
        scored = []
        for idx, entry in enumerate(active):
            vector_similarity = self._cosine(query_vec, self._entry_embedding(entry))
            task_score = self._task_type_score(entry.get("task_type", "general"), task_type)
            keyword_score = self._keyword_score(entry, instruction)
            confidence = float(entry.get("stats", {}).get("confidence", 0.5))
            recency_success = self._recency_success_score(entry)
            score = (
                0.45 * max(vector_similarity, 0.0)
                + 0.25 * task_score
                + 0.15 * keyword_score
                + 0.10 * confidence
                + 0.05 * recency_success
                + 0.10 * rrf.get(idx, 0.0)
            )
            if score < self.retrieve_threshold:
                continue
            view = dict(entry)
            view["_score"] = score
            view["_vector_similarity"] = vector_similarity
            view["_keyword_score"] = keyword_score
            scored.append(view)
        scored.sort(key=lambda item: item["_score"], reverse=True)
        return self._dedupe_content(scored)[:top_k]

    def build_query_text(self, question: str, task_type: str = "general") -> str:
        pattern = self._likely_pattern(question, task_type)
        return (
            f"Task type: {self._normalize_task_type(task_type)}. "
            f"Question: {question}. "
            f"Likely pattern: {pattern}. "
            "Need: reusable strategy, warnings, and short supported answer."
        )

    def format_memories_for_prompt(self, memories: list[dict[str, Any]]) -> list[str]:
        formatted = []
        for mem in memories:
            prefix = "Relevant Skill" if mem.get("memory_type") == "skill" else "Execution Warning"
            procedure = "; ".join(_as_list(mem.get("procedure"))[:5])
            avoid = "; ".join(_as_list(mem.get("avoid"))[:4])
            parts = [
                f"[{prefix}: {mem.get('title', 'Untitled')}]",
                f"Trigger: {mem.get('trigger_pattern', '')}",
                f"Do: {mem.get('content', '')}",
            ]
            if procedure:
                parts.append(f"Procedure: {procedure}")
            if avoid:
                parts.append(f"Avoid: {avoid}")
            parts.append("Use as a general hint, not as a factual answer.")
            formatted.append(" ".join(part for part in parts if part.strip()))
        return formatted

    # ------------------------------------------------------------------
    # Writing, validation, and merge
    # ------------------------------------------------------------------
    def auto_dream_deduplication(
        self,
        new_rule: str,
        base_keywords: list[str] | None = None,
        advisor: MemoryAdvisor | None = None,
        task_instruction: str = "",
        memory_type: str = "reflection",
        task_type: str = "general",
        episode_id: str | None = None,
    ) -> dict[str, Any]:
        """Validate and ADD/EDIT/IGNORE a candidate memory."""
        clean_rule = re.sub(r"\s+", " ", (new_rule or "").strip())
        if not clean_rule:
            return {"operation": "IGNORE", "reason": "empty_rule"}
        candidate = self._candidate_from_rule(
            clean_rule,
            base_keywords=base_keywords,
            memory_type=memory_type,
            task_type=task_type,
            episode_id=episode_id,
        )
        validation = self.validate_candidate_memory(candidate)
        if not validation["ok"]:
            return {"operation": "IGNORE", "rule": clean_rule, "reason": validation["reason"]}

        similar = self.find_similar_rules(candidate["embedding_text"], threshold=0.45)
        advice = (
            advisor(clean_rule, similar, task_instruction)
            if advisor
            else self._heuristic_advice(candidate, similar)
        )
        operation = str(advice.get("operation", "ADD")).upper()
        if operation not in {"ADD", "EDIT", "IGNORE"}:
            operation = "ADD"
        if operation == "IGNORE":
            return {
                "operation": "IGNORE",
                "target_id": advice.get("target_id"),
                "memory_id": advice.get("target_id"),
                "rule": str(advice.get("rule") or clean_rule),
                "reason": advice.get("reason", "ignored_by_advisor"),
                **self._advisor_metadata(advice),
            }
        if operation == "EDIT":
            target = self._find_target(advice.get("target_id"), similar)
            if target is not None:
                return self._merge_entry(target, candidate, advice)
            operation = "ADD"
        result = self._add_entry(candidate, advice)
        self.prune_memory()
        return result

    def validate_candidate_memory(self, memory: dict[str, Any]) -> dict[str, Any]:
        if memory.get("memory_type") not in ACTIVE_MEMORY_TYPES:
            return {"ok": False, "reason": "unsupported_memory_type"}
        if not str(memory.get("trigger_pattern", "")).strip():
            return {"ok": False, "reason": "missing_trigger_pattern"}
        if self._is_generic_trigger(str(memory.get("trigger_pattern", ""))):
            return {"ok": False, "reason": "generic_trigger_pattern"}
        if not memory.get("procedure") and not memory.get("avoid"):
            return {"ok": False, "reason": "missing_procedure_or_avoid"}
        if len(str(memory.get("content", ""))) > 1200:
            return {"ok": False, "reason": "content_too_long"}
        if self._is_low_quality_reflection_memory(memory):
            return {"ok": False, "reason": "low_quality_reflection_memory"}
        if self._looks_like_specific_answer(memory):
            return {"ok": False, "reason": "possible_answer_leakage"}
        return {"ok": True, "reason": "valid"}

    def find_similar_rules(
        self, new_rule: str, threshold: float = 0.45, limit: int = 5
    ) -> list[dict[str, Any]]:
        matches = []
        new_vec = self.get_embedding(new_rule)
        for entry in self.memory_db:
            sim = max(
                _jaccard(new_rule, entry.get("embedding_text", "") or entry.get("rule", "")),
                self._cosine(new_vec, self._entry_embedding(entry)),
            )
            if sim >= threshold:
                view = dict(entry)
                view["similarity"] = sim
                matches.append(view)
        matches.sort(key=lambda item: item["similarity"], reverse=True)
        return matches[:limit]

    def _add_entry(self, candidate: dict[str, Any], advice: dict[str, Any]) -> dict[str, Any]:
        now = _now_iso()
        candidate["created_at"] = now
        candidate["updated_at"] = now
        candidate["last_operation"] = "ADD"
        candidate["last_reason"] = advice.get("reason", "")
        candidate.update(self._advisor_metadata(advice, prefix="last_"))
        candidate = self._normalize_entry(candidate)
        self.memory_db.append(candidate)
        self._save_db()
        self._append_backup(candidate, "ADD")
        return {"operation": "ADD", **candidate, **self._advisor_metadata(advice)}

    def _merge_entry(
        self, entry: dict[str, Any], candidate: dict[str, Any], advice: dict[str, Any]
    ) -> dict[str, Any]:
        old_rule = entry.get("rule", "")
        entry["trigger_keywords"] = sorted(
            set(entry.get("trigger_keywords", [])) | set(candidate.get("trigger_keywords", []))
        )[:80]
        entry["keywords"] = entry["trigger_keywords"]
        entry["procedure"] = self._merge_lists(entry.get("procedure"), candidate.get("procedure"), max_items=8)
        entry["avoid"] = self._merge_lists(entry.get("avoid"), candidate.get("avoid"), max_items=8)
        entry["example_patterns"] = self._merge_lists(
            entry.get("example_patterns"), candidate.get("example_patterns"), max_items=8
        )
        if len(candidate.get("content", "")) > len(entry.get("content", "")) * 1.25:
            entry["content"] = candidate["content"]
        entry["embedding_text"] = self.build_embedding_text(entry)
        entry["rule"] = self._rule_alias(entry)
        entry["updated_at"] = _now_iso()
        entry["edit_count"] = int(entry.get("edit_count", 0)) + 1
        entry["merge_count"] = int(entry.get("merge_count", 0)) + 1
        entry["last_operation"] = "EDIT"
        entry["last_reason"] = advice.get("reason", "")
        entry["stats"]["confidence"] = min(0.95, float(entry["stats"].get("confidence", 0.5)) + 0.03)
        self._merge_source(entry, candidate)
        self._save_db()
        self._append_backup(entry, "EDIT")
        return {
            "operation": "EDIT",
            "id": entry.get("memory_id"),
            "memory_id": entry.get("memory_id"),
            "old_rule": old_rule,
            "rule": entry.get("rule"),
            "reason": advice.get("reason", ""),
            **self._advisor_metadata(advice),
        }

    def _advisor_metadata(self, advice: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        metadata = {}
        for key in ("advisor_source", "advisor_model", "advisor_error"):
            value = advice.get(key)
            if value:
                metadata[f"{prefix}{key}"] = value
        return metadata

    def _heuristic_advice(
        self, candidate: dict[str, Any], similar_entries: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if not similar_entries:
            return {"operation": "ADD", "rule": candidate["rule"], "reason": "no_similar_memory"}
        best = similar_entries[0]
        sim = float(best.get("similarity", 0.0))
        if sim >= 0.92:
            return {
                "operation": "EDIT",
                "target_id": best.get("memory_id"),
                "rule": candidate["rule"],
                "reason": f"merge_duplicate_memory_{sim:.2f}",
            }
        if sim >= 0.78 and not self._looks_conflicting(best, candidate):
            return {
                "operation": "EDIT",
                "target_id": best.get("memory_id"),
                "rule": candidate["rule"],
                "reason": f"merge_similar_memory_{sim:.2f}",
            }
        if sim >= 0.78 and self._looks_conflicting(best, candidate):
            return {
                "operation": "IGNORE",
                "target_id": best.get("memory_id"),
                "rule": best.get("rule", candidate["rule"]),
                "reason": f"conflict_with_existing_higher_confidence_{sim:.2f}",
            }
        return {"operation": "ADD", "rule": candidate["rule"], "reason": f"weak_similarity_{sim:.2f}"}

    # ------------------------------------------------------------------
    # Usage feedback
    # ------------------------------------------------------------------
    def log_usage(
        self,
        memory_ids: list[str],
        episode_id: str,
        task_type: str,
        question: str,
        used_position: str = "planner",
    ) -> list[str]:
        usage_ids = []
        self.usage_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.usage_log_path, "a", encoding="utf-8") as f:
            for memory_id in memory_ids:
                usage_id = _stable_id("use", f"{memory_id}:{episode_id}:{used_position}:{time.time()}")
                usage_ids.append(usage_id)
                row = {
                    "usage_id": usage_id,
                    "memory_id": memory_id,
                    "episode_id": episode_id,
                    "task_type": self._normalize_task_type(task_type),
                    "question": question[:1000],
                    "used_position": used_position,
                    "outcome": "pending",
                    "judged_relevant": None,
                    "judged_helpful": None,
                    "comment": None,
                    "created_at": _now_iso(),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return usage_ids

    def update_after_episode(
        self,
        memory_ids: list[str],
        episode_id: str,
        outcome: str,
        judged_helpful: bool | None = None,
        comment: str = "",
    ) -> None:
        success = outcome == "success"
        id_set = set(memory_ids)
        if not id_set:
            return
        now = _now_iso()
        for entry in self.memory_db:
            if entry.get("memory_id") not in id_set and entry.get("rule") not in id_set:
                continue
            stats = entry.setdefault("stats", {})
            stats["usage_count"] = int(stats.get("usage_count", 0)) + 1
            if success:
                stats["success_count"] = int(stats.get("success_count", 0)) + 1
                entry["last_operation"] = "UPVOTE"
            else:
                stats["failure_count"] = int(stats.get("failure_count", 0)) + 1
                entry["last_operation"] = "DOWNVOTE"
            stats["confidence"] = self._confidence(stats)
            entry["weight"] = stats["confidence"]
            entry["last_used_at"] = now
            entry["updated_at"] = now
            if int(stats.get("usage_count", 0)) >= 3 and stats["confidence"] < 0.30:
                entry["status"] = "deprecated"
                entry["last_operation"] = "DEPRECATED"
        self._append_usage_outcomes(memory_ids, episode_id, outcome, judged_helpful, comment)
        self.prune_memory(save=False)
        self._save_db()

    def apply_expel_reward(self, applied_ids_or_rules: list[str], is_success: bool) -> None:
        outcome = "success" if is_success else "failure"
        self.update_after_episode(applied_ids_or_rules, "unknown_episode", outcome)

    def _append_usage_outcomes(
        self,
        memory_ids: list[str],
        episode_id: str,
        outcome: str,
        judged_helpful: bool | None,
        comment: str,
    ) -> None:
        self.usage_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.usage_log_path, "a", encoding="utf-8") as f:
            for memory_id in memory_ids:
                row = {
                    "usage_id": _stable_id("use_update", f"{memory_id}:{episode_id}:{outcome}:{time.time()}"),
                    "memory_id": memory_id,
                    "episode_id": episode_id,
                    "outcome": outcome,
                    "judged_relevant": 1,
                    "judged_helpful": None if judged_helpful is None else int(judged_helpful),
                    "comment": comment,
                    "created_at": _now_iso(),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def prune_memory(self, save: bool = True) -> dict[str, Any]:
        before = len(self.memory_db)
        for entry in self.memory_db:
            stats = entry.setdefault("stats", {})
            stats["confidence"] = self._confidence(stats)
            entry["weight"] = stats["confidence"]
            if int(stats.get("usage_count", 0)) >= 3 and stats["confidence"] < 0.30:
                entry["status"] = "deprecated"
        active = [e for e in self.memory_db if e.get("status", "active") == "active"]
        deprecated = [e for e in self.memory_db if e.get("status", "active") != "active"]
        if self.max_rules > 0 and len(active) > self.max_rules:
            active.sort(
                key=lambda e: (
                    float(e.get("stats", {}).get("confidence", 0.5)),
                    int(e.get("merge_count", 0)) + int(e.get("edit_count", 0)),
                    str(e.get("updated_at", "")),
                ),
                reverse=True,
            )
            for removed in active[self.max_rules :]:
                removed["status"] = "deprecated"
                removed["last_operation"] = "REMOVE"
                removed["last_reason"] = "memory_max_rules_prune"
            active = active[: self.max_rules]
        self.memory_db = active + deprecated
        removed_count = before - len(active)
        if save:
            self._save_db()
        return {
            "operation": "PRUNE",
            "before": before,
            "after": len(active),
            "removed": max(0, removed_count),
            "max_rules": self.max_rules,
        }

    # ------------------------------------------------------------------
    # Embedding and scoring
    # ------------------------------------------------------------------
    def get_embedding(self, text: str) -> list[float]:
        if self.embedding_enabled and self.embedding_base_url:
            try:
                from openai import OpenAI

                client = OpenAI(base_url=self.embedding_base_url, api_key=os.getenv("EMBEDDING_API_KEY", "EMPTY"))
                response = client.embeddings.create(model=self.embedding_model, input=text)
                vector = list(response.data[0].embedding)
                return self._normalize_vector(vector)
            except Exception:
                pass
        return self._hash_embedding(text)

    def _entry_embedding(self, entry: dict[str, Any]) -> list[float]:
        embedding = entry.get("embedding")
        if isinstance(embedding, list) and embedding:
            return self._normalize_vector([float(x) for x in embedding])
        embedding_text = entry.get("embedding_text") or self.build_embedding_text(entry)
        embedding = self.get_embedding(embedding_text)
        entry["embedding"] = embedding
        return embedding

    def _hash_embedding(self, text: str) -> list[float]:
        vec = [0.0] * self.embedding_dim
        toks = list(_tokens(text))
        for token in toks:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.embedding_dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        return self._normalize_vector(vec)

    def _normalize_vector(self, vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def _cosine(self, a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        n = min(len(a), len(b))
        return sum(a[i] * b[i] for i in range(n)) / (
            math.sqrt(sum(a[i] * a[i] for i in range(n)))
            * math.sqrt(sum(b[i] * b[i] for i in range(n)))
            + 1e-9
        )

    def _rank_dense(self, entries: list[dict[str, Any]], query_vec: list[float]) -> list[int]:
        scores = [(idx, self._cosine(query_vec, self._entry_embedding(entry))) for idx, entry in enumerate(entries)]
        scores.sort(key=lambda item: item[1], reverse=True)
        return [idx for idx, _ in scores]

    def _rank_lexical(self, entries: list[dict[str, Any]], instruction: str) -> list[int]:
        scores = [(idx, self._keyword_score(entry, instruction)) for idx, entry in enumerate(entries)]
        scores.sort(key=lambda item: item[1], reverse=True)
        return [idx for idx, _ in scores]

    def _reciprocal_rank_fusion(self, rankings: list[list[int]], k: int = 60) -> dict[int, float]:
        scores: dict[int, float] = {}
        for ranking in rankings:
            for rank, idx in enumerate(ranking, start=1):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
        return scores

    def _task_type_score(self, memory_task_type: str, current_task_type: str) -> float:
        memory_task_type = self._normalize_task_type(memory_task_type)
        current_task_type = self._normalize_task_type(current_task_type)
        if memory_task_type == current_task_type:
            return 1.0
        if memory_task_type == "general":
            return 0.6
        if {memory_task_type, current_task_type} <= {"2wiki", "multi_hop", "comparison"}:
            return 0.75
        return 0.0

    def _keyword_score(self, entry: dict[str, Any], instruction: str) -> float:
        keywords = set(str(k).lower() for k in entry.get("trigger_keywords", []) if str(k).strip())
        if not keywords:
            keywords = _tokens(entry.get("embedding_text", "") or entry.get("content", ""))
        if not keywords:
            return 0.0
        instruction_lower = instruction.lower()
        matched = sum(1 for keyword in keywords if keyword in instruction_lower or keyword in _tokens(instruction))
        return matched / max(len(keywords), 1)

    def _recency_success_score(self, entry: dict[str, Any]) -> float:
        stats = entry.get("stats", {})
        successes = int(stats.get("success_count", 0))
        failures = int(stats.get("failure_count", 0))
        if successes + failures == 0:
            return 0.5
        return successes / max(successes + failures, 1)

    def _dedupe_content(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        kept = []
        seen: list[str] = []
        for entry in entries:
            text = entry.get("embedding_text", "") or entry.get("content", "")
            if any(_jaccard(text, old) > 0.88 for old in seen):
                continue
            seen.append(text)
            kept.append(entry)
        return kept

    # ------------------------------------------------------------------
    # Candidate construction helpers
    # ------------------------------------------------------------------
    def _candidate_from_rule(
        self,
        rule: str,
        base_keywords: list[str] | None,
        memory_type: str,
        task_type: str,
        episode_id: str | None,
    ) -> dict[str, Any]:
        clean = re.sub(r"\s+", " ", rule.strip())
        source_ids = [episode_id] if episode_id else []
        candidate = {
            "memory_id": _stable_id(memory_type[:5] or "mem", clean),
            "memory_type": memory_type if memory_type in ACTIVE_MEMORY_TYPES else "reflection",
            "task_type": self._normalize_task_type(task_type),
            "title": self._title_from_content(clean),
            "trigger_pattern": self._trigger_from_rule(clean),
            "trigger_keywords": (base_keywords or sorted(_tokens(clean))[:20])[:80],
            "content": self._content_from_rule(clean),
            "procedure": self._procedure_from_rule(clean),
            "avoid": self._avoid_from_rule(clean),
            "example_patterns": [],
            "source": {"source_type": "reflection", "source_episode_ids": source_ids},
            "stats": {"usage_count": 0, "success_count": 0, "failure_count": 0, "confidence": 0.5},
            "status": "active",
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "last_used_at": None,
        }
        candidate["embedding_text"] = self.build_embedding_text(candidate)
        candidate["rule"] = clean
        candidate["keywords"] = candidate["trigger_keywords"]
        candidate["id"] = candidate["memory_id"]
        candidate["weight"] = 0.5
        return candidate

    def build_embedding_text(self, memory: dict[str, Any]) -> str:
        return re.sub(
            r"\s+",
            " ",
            " ".join(
                [
                    f"Task type: {memory.get('task_type', 'general')}.",
                    f"Memory type: {memory.get('memory_type', 'reflection')}.",
                    f"Trigger: {memory.get('trigger_pattern', '')}.",
                    f"Lesson: {memory.get('content', '')}.",
                    f"Procedure: {'; '.join(_as_list(memory.get('procedure')))}.",
                    f"Avoid: {'; '.join(_as_list(memory.get('avoid')))}.",
                ]
            ).strip(),
        )

    def _rule_alias(self, memory: dict[str, Any]) -> str:
        content = re.sub(r"^(?:\s*Action:\s*)+", "", str(memory.get("content", "")).strip(), flags=re.I)
        trigger = str(memory.get("trigger_pattern", "")).strip() or "When this task pattern appears"
        relation = "Does Not Contribute" if memory.get("memory_type") == "bad_pattern" else "Necessary"
        return f"[{trigger}] -> Action: {content} is {relation} to achieve Goal G (Confidence: should)"

    def _content_from_rule(self, rule: str) -> str:
        if "->" in rule:
            right = rule.split("->", 1)[1]
            right = re.sub(r"^(?:\s*Action:\s*)+", "", right, flags=re.I).strip()
            right = re.sub(r"\s+is\s+(Necessary|Does Not Contribute).*", "", right, flags=re.I).strip()
            return right or rule
        return rule

    def _trigger_from_rule(self, rule: str) -> str:
        match = re.search(r"\[(.*?)\]", rule)
        if match:
            return match.group(1).strip()
        text = rule.lower()
        if "vqa" in text or "image" in text or "visual" in text or "图" in text:
            return "When a visual question requires identifying a subject before checking an attribute"
        if "browser" in text or "http" in text or "timeout" in text or "session" in text:
            return "When browser or search tools return repeated runtime failures"
        if "search" in text or "query" in text or "关键词" in text:
            return "When equivalent search queries stop adding new evidence"
        if "multi-hop" in text or "intermediate" in text or "director" in text or "founder" in text:
            return "When the question requires resolving an intermediate entity before the final attribute"
        return "When a task-solving strategy must be chosen from incomplete evidence"

    def _procedure_from_rule(self, rule: str) -> list[str]:
        text = rule.lower()
        if "multi-hop" in text or "intermediate" in text or "relation" in text:
            return [
                "Identify the final answer type required by the question.",
                "Resolve the intermediate entity first.",
                "Continue searching for the final requested attribute.",
                "Check that the candidate answer matches the requested type.",
                "Return only the final short answer.",
            ]
        if "browser" in text or "search" in text or "api" in text:
            return [
                "Inspect the latest observation before calling another tool.",
                "Change the query or switch tools after repeated errors.",
                "Answer from reliable existing evidence when enough information is available.",
            ]
        return [
            "Use the memory only as a general strategy.",
            "Verify the current task evidence before answering.",
            "Return only the final short answer.",
        ]

    def _avoid_from_rule(self, rule: str) -> list[str]:
        text = rule.lower()
        avoid = ["Do not treat this memory as a factual answer."]
        if "intermediate" in text or "multi-hop" in text:
            avoid.append("Do not answer the intermediate entity as the final answer.")
        if "repeating" in text or "repeated" in text or "error" in text:
            avoid.append("Do not repeat equivalent failing tool calls.")
        return avoid

    def _is_generic_trigger(self, trigger: str) -> bool:
        normalized = re.sub(r"\s+", " ", trigger.strip().lower())
        generic = {
            "this reusable task-solving pattern is detected.",
            "this reusable task-solving pattern is detected",
            "general reusable task-solving pattern.",
            "general reusable task-solving pattern",
            "when this task pattern appears",
        }
        if normalized in generic:
            return True
        if len(_tokens(normalized)) < 3 and not re.search(r"\d|[A-Z]", trigger):
            return True
        return False

    def _is_low_quality_reflection_memory(self, memory: dict[str, Any]) -> bool:
        if memory.get("memory_type") != "reflection":
            return False
        trigger = str(memory.get("trigger_pattern", "")).strip().lower()
        content = str(memory.get("content", "")).strip().lower()
        rule = str(memory.get("rule", "")).strip().lower()
        if (
            "equivalent search queries stop adding new evidence" in trigger
            and "candidate validation" in content
            and "candidate validation is necessary" in rule
        ):
            return True
        if (
            "incomplete evidence" in trigger
            and content
            in {
                "candidate validation",
                "subject validation",
                "cross-entity validation",
                "cross-checking with all relational constraints",
            }
        ):
            return True
        return False

    def _title_from_content(self, text: str) -> str:
        clean = re.sub(r"[\[\]<>]", "", text or "").strip()
        clean = re.sub(r"\s+", " ", clean)
        return clean[:90] or "Reusable task-solving memory"

    def _likely_pattern(self, question: str, task_type: str) -> str:
        text = question.lower()
        normalized = self._normalize_task_type(task_type)
        if normalized in {"2wiki", "multi_hop"} or any(
            marker in text for marker in ("director of", "founder of", "author of", "birthplace", "nationality")
        ):
            return "multi-hop question with intermediate entity"
        if normalized == "visual_qa":
            return "visual question requiring image grounding and possible visual search"
        return "single-hop or open factual verification question"

    def _normalize_task_type(self, task_type: str) -> str:
        text = (task_type or "general").lower()
        if text in {"multi_hop", "comparison", "2wiki"}:
            return "2wiki"
        if text in {"visual_qa", "vqa", "simplevqa"}:
            return "visual_qa"
        if text in {"open_search", "simpleqa"}:
            return "simpleqa"
        return text or "general"

    def _confidence(self, stats: dict[str, Any]) -> float:
        successes = int(stats.get("success_count", 0))
        failures = int(stats.get("failure_count", 0))
        return (successes + 1) / (successes + failures + 2)

    def _merge_lists(self, a: Any, b: Any, max_items: int = 8) -> list[str]:
        merged = []
        for item in _as_list(a) + _as_list(b):
            if item not in merged:
                merged.append(item)
            if len(merged) >= max_items:
                break
        return merged

    def _merge_source(self, entry: dict[str, Any], candidate: dict[str, Any]) -> None:
        source = entry.setdefault("source", {"source_type": "reflection", "source_episode_ids": []})
        cand_source = candidate.get("source", {})
        ids = _as_list(source.get("source_episode_ids")) + _as_list(cand_source.get("source_episode_ids"))
        source["source_episode_ids"] = self._merge_lists(ids, [], max_items=50)

    def _find_target(
        self, target_id: str | None, similar_entries: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        candidate_id = target_id or (similar_entries[0].get("memory_id") if similar_entries else None)
        if not candidate_id:
            return None
        for entry in self.memory_db:
            if entry.get("memory_id") == candidate_id or entry.get("id") == candidate_id:
                return entry
        return None

    def _looks_like_specific_answer(self, memory: dict[str, Any]) -> bool:
        text = " ".join(
            [
                str(memory.get("content", "")),
                " ".join(_as_list(memory.get("example_patterns"))),
            ]
        )
        capitalized = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b", text)
        return len(set(capitalized)) >= 5

    def _looks_conflicting(self, old: dict[str, Any], new: dict[str, Any]) -> bool:
        old_text = str(old.get("content", "")).lower()
        new_text = str(new.get("content", "")).lower()
        conflict_pairs = [
            ("do not", "should"),
            ("avoid", "use"),
            ("does not contribute", "necessary"),
        ]
        return any(a in old_text and b in new_text for a, b in conflict_pairs)


class WorkingMemory:
    """Per-case state that is never written to long-term memory."""

    def __init__(self, episode_id: str, task_type: str = "general", question: str = "") -> None:
        self.state = {
            "episode_id": episode_id,
            "task_type": task_type,
            "question": question,
            "step_count": 0,
            "searched_queries": [],
            "visited_urls": [],
            "intermediate_results": [],
            "candidate_answers": [],
            "open_questions": [],
            "tool_errors": [],
            "repeated_query_count": 0,
        }

    def add_search_query(self, query: str) -> bool:
        repeated = self.detect_repeated_query(query)
        if repeated:
            self.state["repeated_query_count"] += 1
        self.state["searched_queries"].append(query)
        return repeated

    def add_intermediate_result(self, name: str, value: str, evidence: str = "") -> None:
        self.state["intermediate_results"].append(
            {"name": name, "value": value, "evidence": evidence}
        )

    def add_candidate_answer(self, answer: str, answer_type: str, status: str) -> None:
        self.state["candidate_answers"].append(
            {"answer": answer, "answer_type": answer_type, "status": status}
        )

    def add_tool_error(self, tool_name: str, error: str) -> None:
        self.state["tool_errors"].append({"tool_name": tool_name, "error": error})

    def detect_repeated_query(self, query: str) -> bool:
        return query.strip().lower() in {q.strip().lower() for q in self.state["searched_queries"]}

    def summarize_for_prompt(self) -> str:
        return json.dumps(self.state, ensure_ascii=False)

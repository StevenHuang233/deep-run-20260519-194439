"""Skeptical CLIN-style long-term memory manager."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", (text or "").lower()))


def _jaccard(a: str, b: str) -> float:
    ta = _tokens(a)
    tb = _tokens(b)
    return len(ta & tb) / max(len(ta | tb), 1)


MemoryAdvisor = Callable[[str, list[dict[str, Any]], str], dict[str, Any]]


class SkepticalMemoryManager:
    """Natural-language memory with keyword retrieval and ExpeL-style editing."""

    def __init__(self, db_path: str = "memory.json", max_rules: int = 64):
        self.db_path = Path(db_path)
        self.max_rules = max_rules
        self.memory_db: list[dict[str, Any]] = self._load_db()

    def _load_db(self) -> list[dict[str, Any]]:
        if not self.db_path.exists():
            return []
        try:
            data = json.loads(self.db_path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                return []
            return [self._normalize_entry(item) for item in data if isinstance(item, dict)]
        except json.JSONDecodeError:
            return []

    def _normalize_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        rule = str(entry.get("rule", "")).strip()
        entry.setdefault("id", self._entry_id(rule))
        entry.setdefault("keywords", sorted(_tokens(rule))[:80])
        entry.setdefault("up", 1)
        entry.setdefault("down", 0)
        entry.setdefault("source", "reflection")
        entry["weight"] = self._weight(entry)
        return entry

    def _save_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.db_path.with_suffix(self.db_path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.memory_db, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self.db_path)

    def _entry_id(self, rule: str) -> str:
        return hashlib.sha1(rule.encode("utf-8")).hexdigest()[:12]

    def _weight(self, entry: dict[str, Any]) -> float:
        up = int(entry.get("up", 0))
        down = int(entry.get("down", 0))
        return (up - down) / (up + down + 0.1)

    def retrieve_top_rules(self, instruction: str) -> list[str]:
        """Fast keyword collision retrieval, returning at most two skeptical hints."""
        query_tokens = _tokens(instruction)
        if not query_tokens:
            return []
        scored: list[tuple[float, dict[str, Any]]] = []
        for entry in self.memory_db:
            weight = self._weight(entry)
            if weight < 0.20:
                continue
            keywords = set(entry.get("keywords", [])) | _tokens(entry.get("rule", ""))
            overlap = len(query_tokens & keywords)
            if overlap <= 0:
                continue
            density = overlap / max(len(query_tokens), 1)
            score = density + max(weight, 0) + 0.05 * int(entry.get("merge_count", 0))
            scored.append((score, entry))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1].get("rule", "") for item in scored[:2] if item[1].get("rule")]

    def find_similar_rules(
        self, new_rule: str, threshold: float = 0.45, limit: int = 5
    ) -> list[dict[str, Any]]:
        matches = []
        for entry in self.memory_db:
            sim = _jaccard(new_rule, entry.get("rule", ""))
            if sim >= threshold:
                view = dict(entry)
                view["similarity"] = sim
                matches.append(view)
        matches.sort(key=lambda item: item["similarity"], reverse=True)
        return matches[:limit]

    def auto_dream_deduplication(
        self,
        new_rule: str,
        base_keywords: list[str] | None = None,
        advisor: MemoryAdvisor | None = None,
        task_instruction: str = "",
    ) -> dict[str, Any]:
        """ADD/EDIT/IGNORE candidate memories with optional 32B model advice."""
        clean_rule = re.sub(r"\s+", " ", (new_rule or "").strip())
        base_keywords = (base_keywords or sorted(_tokens(clean_rule)))[:80]
        if not clean_rule:
            return {"operation": "IGNORE", "reason": "empty_rule"}

        similar = self.find_similar_rules(clean_rule)
        advice = (
            advisor(clean_rule, similar, task_instruction)
            if advisor
            else self._heuristic_advice(clean_rule, similar)
        )
        operation = str(advice.get("operation", "ADD")).upper()
        if operation not in {"ADD", "EDIT", "IGNORE"}:
            operation = "ADD"
        advised_rule = re.sub(r"\s+", " ", str(advice.get("rule") or clean_rule).strip())
        target_id = advice.get("target_id")

        if operation == "IGNORE":
            return {
                "operation": "IGNORE",
                "target_id": target_id,
                "rule": advised_rule,
                "reason": advice.get("reason", "ignored_by_advisor"),
            }
        if operation == "EDIT":
            target = self._find_target(target_id, similar)
            if target is not None:
                result = self._edit_entry(target, advised_rule, base_keywords, advice)
                self.prune_memory()
                return result
            operation = "ADD"

        result = self._add_entry(advised_rule, base_keywords, advice)
        self.prune_memory()
        return result

    def _heuristic_advice(
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
                "rule": new_rule if len(new_rule) > len(best.get("rule", "")) else best.get("rule", new_rule),
                "reason": f"similar_rule_merge_{sim:.2f}",
            }
        return {"operation": "ADD", "rule": new_rule, "reason": f"weak_similarity_{sim:.2f}"}

    def _find_target(
        self, target_id: str | None, similar_entries: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        candidate_id = target_id or (similar_entries[0].get("id") if similar_entries else None)
        if not candidate_id:
            return None
        for entry in self.memory_db:
            if entry.get("id") == candidate_id:
                return entry
        return None

    def _edit_entry(
        self,
        entry: dict[str, Any],
        new_rule: str,
        base_keywords: list[str],
        advice: dict[str, Any],
    ) -> dict[str, Any]:
        now = time.time()
        old_rule = entry.get("rule", "")
        entry["rule"] = new_rule
        entry["keywords"] = sorted(set(entry.get("keywords", [])) | set(base_keywords) | _tokens(new_rule))[:80]
        entry["updated_at"] = now
        entry["edit_count"] = int(entry.get("edit_count", 0)) + 1
        entry["merge_count"] = int(entry.get("merge_count", 0)) + 1
        entry["last_operation"] = "EDIT"
        entry["last_reason"] = advice.get("reason", "")
        entry["weight"] = self._weight(entry)
        self._save_db()
        return {
            "operation": "EDIT",
            "id": entry.get("id"),
            "old_rule": old_rule,
            "rule": entry.get("rule"),
            "reason": advice.get("reason", ""),
        }

    def _add_entry(
        self, rule: str, base_keywords: list[str], advice: dict[str, Any]
    ) -> dict[str, Any]:
        now = time.time()
        entry = {
            "id": self._entry_id(rule),
            "rule": rule,
            "keywords": sorted(set(base_keywords) | _tokens(rule))[:80],
            "up": 1,
            "down": 0,
            "weight": 1 / 1.1,
            "created_at": now,
            "updated_at": now,
            "source": "reflection",
            "last_operation": "ADD",
            "last_reason": advice.get("reason", ""),
        }
        self.memory_db.append(entry)
        self._save_db()
        return {"operation": "ADD", **entry}

    def apply_expel_reward(self, applied_rules: list[str], is_success: bool) -> None:
        if not applied_rules:
            return
        rule_set = set(applied_rules)
        for entry in self.memory_db:
            if entry.get("rule") not in rule_set:
                continue
            if is_success:
                entry["up"] = int(entry.get("up", 0)) + 1
                entry["last_operation"] = "UPVOTE"
            else:
                entry["down"] = int(entry.get("down", 0)) + 1
                entry["last_operation"] = "DOWNVOTE"
            entry["weight"] = self._weight(entry)
            entry["updated_at"] = time.time()
        self.memory_db = [e for e in self.memory_db if self._weight(e) >= 0.20]
        self.prune_memory(save=False)
        self._save_db()

    def prune_memory(self, save: bool = True) -> dict[str, Any]:
        """Keep only the strongest rules, similar to ExpeL's bounded rule list."""
        before = len(self.memory_db)
        self.memory_db = [e for e in self.memory_db if self._weight(e) >= 0.20]
        if self.max_rules > 0 and len(self.memory_db) > self.max_rules:
            self.memory_db.sort(
                key=lambda e: (
                    self._weight(e),
                    int(e.get("merge_count", 0)) + int(e.get("edit_count", 0)),
                    float(e.get("updated_at", 0)),
                ),
                reverse=True,
            )
            for removed in self.memory_db[self.max_rules :]:
                removed["last_operation"] = "REMOVE"
                removed["last_reason"] = "memory_max_rules_prune"
            self.memory_db = self.memory_db[: self.max_rules]
        removed_count = before - len(self.memory_db)
        if save and removed_count:
            self._save_db()
        return {
            "operation": "PRUNE",
            "before": before,
            "after": len(self.memory_db),
            "removed": removed_count,
            "max_rules": self.max_rules,
        }

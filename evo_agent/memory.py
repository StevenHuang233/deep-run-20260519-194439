"""Skeptical CLIN-style long-term memory manager."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", (text or "").lower()))


def _jaccard(a: str, b: str) -> float:
    ta = _tokens(a)
    tb = _tokens(b)
    return len(ta & tb) / max(len(ta | tb), 1)


class SkepticalMemoryManager:
    """Natural-language memory with keyword retrieval and ExpeL-style weights."""

    def __init__(self, db_path: str = "memory.json"):
        self.db_path = Path(db_path)
        self.memory_db: list[dict[str, Any]] = self._load_db()

    def _load_db(self) -> list[dict[str, Any]]:
        if not self.db_path.exists():
            return []
        try:
            data = json.loads(self.db_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []

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
        query_tokens = _tokens(instruction)
        if not query_tokens:
            return []
        scored: list[tuple[float, dict[str, Any]]] = []
        for entry in self.memory_db:
            if self._weight(entry) < 0.20:
                continue
            keywords = set(entry.get("keywords", [])) | _tokens(entry.get("rule", ""))
            overlap = len(query_tokens & keywords)
            if overlap <= 0:
                continue
            score = overlap / max(len(query_tokens), 1) + max(self._weight(entry), 0)
            scored.append((score, entry))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1].get("rule", "") for item in scored[:2] if item[1].get("rule")]

    def auto_dream_deduplication(
        self, new_rule: str, base_keywords: list[str] | None = None
    ) -> dict[str, Any]:
        base_keywords = base_keywords or sorted(_tokens(new_rule))
        now = time.time()
        for entry in self.memory_db:
            if _jaccard(new_rule, entry.get("rule", "")) > 0.75:
                merged = sorted(set(entry.get("keywords", [])) | set(base_keywords))
                entry["keywords"] = merged[:80]
                entry["updated_at"] = now
                entry["merge_count"] = int(entry.get("merge_count", 0)) + 1
                self._save_db()
                return entry

        entry = {
            "id": self._entry_id(new_rule),
            "rule": new_rule.strip(),
            "keywords": base_keywords[:80],
            "up": 1,
            "down": 0,
            "weight": 1 / 1.1,
            "created_at": now,
            "updated_at": now,
            "source": "reflection",
        }
        self.memory_db.append(entry)
        self._save_db()
        return entry

    def apply_expel_reward(self, applied_rules: list[str], is_success: bool) -> None:
        if not applied_rules:
            return
        rule_set = set(applied_rules)
        for entry in self.memory_db:
            if entry.get("rule") not in rule_set:
                continue
            if is_success:
                entry["up"] = int(entry.get("up", 0)) + 1
            else:
                entry["down"] = int(entry.get("down", 0)) + 1
            entry["weight"] = self._weight(entry)
            entry["updated_at"] = time.time()
        self.memory_db = [e for e in self.memory_db if self._weight(e) >= 0.20]
        self._save_db()

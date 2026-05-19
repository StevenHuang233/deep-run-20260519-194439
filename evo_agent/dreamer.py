"""Offline memory consolidation inspired by AutoDream-style systems."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from .compiler import CognitiveCompiler
from .config import HarnessConfig, ensure_output_dirs
from .memory import SkepticalMemoryManager
from .types import Reflection


class MemoryDreamer:
    """Review recent trajectories and consolidate repeated failures into memory.

    This module is deliberately offline: it never writes predictions and never
    changes a running case. It only turns failed trajectories into generalized
    CLIN-style rules, then relies on SkepticalMemoryManager to deduplicate,
    merge, and prune them.
    """

    def __init__(
        self,
        config: HarnessConfig,
        memory: SkepticalMemoryManager | None = None,
        compiler: CognitiveCompiler | None = None,
    ) -> None:
        self.config = config
        ensure_output_dirs(config)
        self.memory = memory or SkepticalMemoryManager(
            config.memory_db_path, max_rules=config.memory_max_rules
        )
        self.compiler = compiler or CognitiveCompiler(
            ref_model=config.reflection_model_name,
            base_url=config.reflection_base_url,
            api_key=config.reflection_api_key,
            enable_model=config.reflection_model_enabled or config.memory_model_enabled,
            max_model_billion=config.reflection_model_max_b,
        )

    def run(
        self,
        trajectory_dir: str | None = None,
        report_path: str | None = None,
        limit: int | None = None,
        min_failures: int | None = None,
    ) -> dict[str, Any]:
        """Run one consolidation pass and write a JSON report."""
        trajectory_dir = trajectory_dir or self.config.trajectory_dir
        report_path = report_path or self.config.dream_report_path
        limit = self.config.dream_max_trajectories if limit is None else limit
        min_failures = self.config.dream_min_failures if min_failures is None else min_failures

        trajectories = self._load_recent_trajectories(Path(trajectory_dir), limit)
        failures = [item for item in trajectories if item["failed"]]
        memory_updates: list[dict[str, Any]] = []
        reflections: list[dict[str, Any]] = []

        if len(failures) >= max(1, min_failures):
            for item in failures:
                reflection = self.compiler.compile_failed_trace(
                    item["rows"], item["instruction"]
                )
                reflection_view = reflection.as_dict()
                reflection_view["trajectory_path"] = item["path"]
                reflection_view["exit_status"] = item["exit_status"]
                reflections.append(reflection_view)
                if not reflection.memory_worthy:
                    continue
                update = self.memory.auto_dream_deduplication(
                    reflection.clin_rule,
                    self._memory_keywords(item, reflection),
                    advisor=self.compiler.advise_memory_update
                    if self.config.memory_model_enabled
                    else None,
                    task_instruction=item["instruction"],
                )
                update["trajectory_path"] = item["path"]
                update["failure_type"] = reflection.failure_type
                memory_updates.append(update)

        prune_report = self.memory.prune_memory()
        report = {
            "created_at": time.time(),
            "trajectory_dir": str(Path(trajectory_dir)),
            "trajectories_reviewed": len(trajectories),
            "failures_considered": len(failures),
            "min_failures": min_failures,
            "memory_updates": memory_updates,
            "reflections": reflections,
            "prune": prune_report,
            "memory_size": len(self.memory.memory_db),
        }
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return report

    def _load_recent_trajectories(self, trajectory_dir: Path, limit: int) -> list[dict[str, Any]]:
        if not trajectory_dir.exists():
            return []
        paths = sorted(
            trajectory_dir.glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if limit and limit > 0:
            paths = paths[:limit]
        items = []
        for path in paths:
            rows = self._read_jsonl(path)
            if not rows:
                continue
            status = self._run_status(rows)
            items.append(
                {
                    "path": str(path),
                    "task_id": path.stem,
                    "rows": rows,
                    "instruction": self._extract_instruction(rows),
                    "exit_status": status.get("exit_status", ""),
                    "failure_reason": status.get("failure_reason", ""),
                    "failed": self._is_failed(rows, status),
                }
            )
        return items

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows

    def _run_status(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        for row in reversed(rows):
            if row.get("event_type") == "run_status" and isinstance(row.get("content"), dict):
                return row["content"]
        return {}

    def _is_failed(self, rows: list[dict[str, Any]], status: dict[str, Any]) -> bool:
        if status:
            exit_status = str(status.get("exit_status", ""))
            if status.get("success") is False:
                return True
            if status.get("failure_reason"):
                return True
            return exit_status not in {"submitted", "self_reflection_submitted"}
        assistant_text = " ".join(
            str(row.get("content", "")) for row in rows if row.get("role") == "assistant"
        )
        return not bool(re.search(r"<answer>.*?</answer>", assistant_text, flags=re.S | re.I))

    def _extract_instruction(self, rows: list[dict[str, Any]]) -> str:
        for row in rows:
            if row.get("role") != "user":
                continue
            content = row.get("content", "")
            if isinstance(content, str):
                return content[:4000]
            if isinstance(content, list):
                texts = [
                    str(item.get("text", ""))
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                return "\n".join(texts)[:4000]
        return ""

    def _memory_keywords(self, item: dict[str, Any], reflection: Reflection) -> list[str]:
        text = " ".join(
            [
                item.get("instruction", ""),
                item.get("exit_status", ""),
                item.get("failure_reason", ""),
                reflection.failure_type,
                reflection.root_cause,
            ]
        )
        return sorted(set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())))[:80]

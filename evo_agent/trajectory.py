"""JSONL trajectory recording compatible with the provided harness format."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from .types import Role


class Trajectory:
    """Append-only JSONL trajectory store.

    The core fields match the provided harness:
    timestamp, step_id, role, content, tool_call_id.
    Extra reflection/memory events are stored with role="event" and are not
    replayed back into the model context.
    """

    def __init__(self, task_id: str, output_dir: str = "trajectories", reset: bool = False) -> None:
        self.task_id = task_id
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.path = out / f"{task_id}.jsonl"
        if reset and self.path.exists():
            self.path.unlink()

    def write(
        self,
        role: Role,
        content,
        step_id: Optional[int] = None,
        tool_call_id: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> None:
        entry = {
            "timestamp": time.time(),
            "step_id": step_id,
            "role": role.value,
            "content": content,
            "tool_call_id": tool_call_id,
        }
        if extra:
            entry.update(extra)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def write_event(self, event_type: str, payload: dict, step_id: Optional[int]) -> None:
        self.write(
            Role.EVENT,
            payload,
            step_id=step_id,
            extra={"event_type": event_type, "include_in_context": False},
        )

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def to_messages(
        self,
        recent_steps: int | None = None,
        include_images: bool = True,
    ) -> list[dict]:
        keep_steps: set[int | None] | None = None
        if recent_steps and recent_steps > 0:
            step_ids = sorted(
                {
                    item.get("step_id")
                    for item in self.read_all()
                    if isinstance(item.get("step_id"), int) and item.get("step_id") > 0
                }
            )
            keep_steps = set(step_ids[-recent_steps:])
            keep_steps.update({0, None})

        messages = []
        for entry in self.read_all():
            if entry.get("include_in_context") is False:
                continue
            if keep_steps is not None and entry.get("step_id") not in keep_steps:
                continue
            role = entry.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                continue
            content = entry.get("content") or ""
            if not include_images:
                content = self._content_without_images(content)
            msg = {"role": role, "content": content}
            if role == "assistant" and entry.get("tool_calls"):
                msg["tool_calls"] = entry["tool_calls"]
            if entry.get("tool_call_id"):
                msg["tool_call_id"] = entry["tool_call_id"]
            messages.append(msg)
        return messages

    def _content_without_images(self, content):
        """Drop expensive multimodal image payloads while keeping task text."""
        if not isinstance(content, list):
            return content
        text_parts = []
        image_count = 0
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                text = item.get("text") or ""
                if text:
                    text_parts.append(str(text))
            elif item.get("type") == "image_url":
                image_count += 1
        if image_count:
            text_parts.append(
                "[Image omitted from this later turn to reduce cost. "
                "Use the provided image_url/local_image_path text or prior visual observations if needed.]"
            )
        return "\n".join(text_parts).strip()

    def summary(self) -> dict:
        entries = self.read_all()
        role_counts: dict[str, int] = {}
        for item in entries:
            role = item.get("role", "")
            role_counts[role] = role_counts.get(role, 0) + 1
        return {
            "task_id": self.task_id,
            "total_turns": len(entries),
            "role_counts": role_counts,
            "path": str(self.path),
        }

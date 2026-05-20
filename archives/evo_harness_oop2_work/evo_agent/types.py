"""Shared small data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    EVENT = "event"


@dataclass
class TaskCase:
    index: int
    instruction: str
    answer: str = ""
    task_id: Optional[str] = None
    image: str = ""
    image_b64: Optional[str] = None
    image_url: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Reflection:
    failure_type: str
    root_cause: str
    correct_strategy: list[str]
    memory_worthy: bool
    clin_rule: str
    confidence: str = "should"
    compiler_source: str = "heuristic"
    compiler_model: str = ""
    compiler_error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_type": self.failure_type,
            "root_cause": self.root_cause,
            "correct_strategy": self.correct_strategy,
            "memory_worthy": self.memory_worthy,
            "clin_rule": self.clin_rule,
            "confidence": self.confidence,
            "compiler_source": self.compiler_source,
            "compiler_model": self.compiler_model,
            "compiler_error": self.compiler_error,
        }

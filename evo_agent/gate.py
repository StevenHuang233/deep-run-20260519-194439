"""Physical tool gate middleware for loop and error interception."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any


class ToolGateBlocked(RuntimeError):
    """Raised when the physical gate decides the loop should be cut off."""


class ToolGateMiddleware:
    """Detect repeated tool calls and persistent tool failures."""

    def __init__(self, loop_limit: int = 3, similarity_threshold: float = 0.85):
        self.loop_limit = loop_limit
        self.similarity_threshold = similarity_threshold
        self.call_history: list[str] = []
        self.error_tracker: dict[str, int] = defaultdict(int)

    def _tokens(self, text: str) -> set[str]:
        return set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower()))

    def _calculate_jaccard(self, str1: str, str2: str) -> float:
        words1 = self._tokens(str1)
        words2 = self._tokens(str2)
        return len(words1 & words2) / max(len(words1 | words2), 1)

    def _canonical_call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        try:
            arg_text = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
        except TypeError:
            arg_text = str(arguments)
        return f"{tool_name} {arg_text}"

    def inspect_call(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        call = self._canonical_call(tool_name, arguments)
        self.call_history.append(call)
        if len(self.call_history) < self.loop_limit:
            return True

        recent = self.call_history[-self.loop_limit :]
        similarities = [
            self._calculate_jaccard(recent[i], recent[i + 1])
            for i in range(len(recent) - 1)
        ]
        if similarities and min(similarities) >= self.similarity_threshold:
            raise ToolGateBlocked(
                "repeated tool calls detected: "
                f"last_{self.loop_limit}_jaccard={similarities}"
            )
        return True

    def track_error(self, tool_name: str, error_message: str) -> None:
        if not error_message:
            self.error_tracker[tool_name] = 0
            return

        text = error_message.lower()
        hard_error = any(
            marker in text
            for marker in (
                "401",
                "403",
                "auth",
                "unauthorized",
                "forbidden",
                "api key",
                "invalid token",
            )
        )
        retryable_but_costly = any(
            marker in text
            for marker in (
                "timeout",
                "timed out",
                "500",
                "internal server error",
                "connection",
                "session",
                "proxy-error",
            )
        )

        if not (hard_error or retryable_but_costly):
            self.error_tracker[tool_name] = 0
            return

        self.error_tracker[tool_name] += 1
        if hard_error or self.error_tracker[tool_name] >= 2:
            raise ToolGateBlocked(
                f"persistent tool error for {tool_name}: {error_message[:300]}"
            )

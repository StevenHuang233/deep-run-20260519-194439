"""Physical tool gate middleware for loop and error interception."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any


class ToolGateBlocked(RuntimeError):
    """Raised when the physical gate decides the loop should be cut off."""


class ToolGateMiddleware:
    """Detect repeated tool calls and persistent tool failures.

    The first threshold detects likely no-progress loops. The critical threshold
    is a second, OpenClaw-style hard cap for exact or near-exact repetitions.
    """

    def __init__(
        self,
        loop_limit: int = 3,
        similarity_threshold: float = 0.85,
        critical_limit: int = 6,
        history_size: int = 20,
    ):
        self.loop_limit = loop_limit
        self.critical_limit = max(critical_limit, loop_limit)
        self.history_size = max(history_size, self.critical_limit)
        self.similarity_threshold = similarity_threshold
        self.call_history: list[str] = []
        self.error_tracker: dict[str, int] = defaultdict(int)
        self.loop_warning_count = 0

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
        self.call_history = self.call_history[-self.history_size :]
        if len(self.call_history) < self.loop_limit:
            return True

        recent = self.call_history[-self.loop_limit :]
        similarities = [
            self._calculate_jaccard(recent[i], recent[i + 1])
            for i in range(len(recent) - 1)
        ]
        if similarities and min(similarities) >= self.similarity_threshold:
            self.loop_warning_count += 1
            raise ToolGateBlocked(
                "repeated tool calls detected: "
                f"last_{self.loop_limit}_jaccard={similarities}"
            )
        critical_recent = self.call_history[-self.critical_limit :]
        if len(critical_recent) >= self.critical_limit:
            first = critical_recent[0]
            if all(
                self._calculate_jaccard(first, other) >= self.similarity_threshold
                for other in critical_recent[1:]
            ):
                raise ToolGateBlocked(
                    "critical repeated tool-call loop detected: "
                    f"critical_limit={self.critical_limit}"
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

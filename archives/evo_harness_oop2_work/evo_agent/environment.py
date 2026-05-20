"""Tool environment that reuses the existing service adapters."""

from __future__ import annotations

import json
import os
import re
import time
import sys
import threading
from pathlib import Path
from typing import Any, Callable

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
TOOL_ROOTS = [
    WORKSPACE_ROOT / "harness-sii" / "harness-sii",
    WORKSPACE_ROOT / "harness-sii",
    WORKSPACE_ROOT,
]
for tool_root in reversed(TOOL_ROOTS):
    if tool_root.exists() and str(tool_root) not in sys.path:
        sys.path.insert(0, str(tool_root))


_SEMAPHORES_LOCK = threading.Lock()
_SEMAPHORES: dict[tuple[str, int], threading.BoundedSemaphore] = {}


def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: str) -> int:
    return int(os.getenv(name, default))


def _shared_semaphore(name: str, limit: int) -> threading.BoundedSemaphore:
    limit = max(1, int(limit or 1))
    key = (name, limit)
    with _SEMAPHORES_LOCK:
        semaphore = _SEMAPHORES.get(key)
        if semaphore is None:
            semaphore = threading.BoundedSemaphore(limit)
            _SEMAPHORES[key] = semaphore
        return semaphore

# The current project already provides browser/search services. This module
# keeps those modules as the single source of truth and only adapts OpenAI
# tool-call JSON to their Python functions.


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": "Web text search. Returns a list of {rank,title,url,snippet,content?}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 3},
                    "fetch": {"type": "boolean", "default": True},
                    "max_chars": {"type": "integer", "default": 500},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_image",
            "description": "Reverse image search. Use image_url when an online URL is available; otherwise use image with a local file path so the proxy can upload it. Returns ranked web results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_url": {"type": "string"},
                    "image": {"type": "string"},
                    "top_k": {"type": "integer", "default": 1},
                    "fetch": {"type": "boolean", "default": True},
                    "max_chars": {"type": "integer", "default": 500},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_navigate",
            "description": "Open a URL through the existing browser service.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "wait_until": {
                        "type": "string",
                        "enum": ["domcontentloaded", "load", "networkidle"],
                        "default": "domcontentloaded",
                    },
                    "include_text": {"type": "boolean", "default": True},
                    "max_text": {"type": "integer", "default": 2000},
                    "timeout": {"type": "integer", "default": 30},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_get_text",
            "description": "Return text from the current browser page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_chars": {"type": "integer", "default": 5000},
                    "timeout": {"type": "integer", "default": 15},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_click",
            "description": "Click an element by CSS selector in the current browser page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string"},
                    "nth": {"type": "integer", "default": 0},
                    "timeout": {"type": "integer", "default": 10},
                },
                "required": ["selector"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_type",
            "description": "Type into an input selected by CSS selector.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string"},
                    "text": {"type": "string"},
                    "submit": {"type": "boolean", "default": False},
                    "clear": {"type": "boolean", "default": True},
                    "timeout": {"type": "integer", "default": 10},
                },
                "required": ["selector", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_parallel",
            "description": "Open or extract text from multiple URLs through the existing browser service.",
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {"type": "array", "items": {"type": "string"}},
                    "mode": {
                        "type": "string",
                        "enum": ["navigate", "get_text"],
                        "default": "navigate",
                    },
                    "max_chars": {"type": "integer"},
                    "wait_until": {
                        "type": "string",
                        "enum": ["domcontentloaded", "load", "networkidle"],
                        "default": "domcontentloaded",
                    },
                    "max_concurrency": {"type": "integer", "default": 4},
                    "timeout": {"type": "integer", "default": 30},
                },
                "required": ["urls"],
            },
        },
    },
]


class ToolEnvironment:
    """Dispatch OpenAI tool calls to the existing Python tool functions."""

    def __init__(
        self,
        disable_browser_tools: bool = False,
        retry_attempts: int = 3,
        retry_min_seconds: float = 1,
        retry_max_seconds: float = 8,
        search_text_max_top_k: int | None = None,
        search_text_max_chars: int | None = None,
        search_tool_max_concurrency: int | None = None,
        browser_tool_max_concurrency: int | None = None,
        search_text_default_fetch: bool | None = None,
        search_image_default_fetch: bool | None = None,
        search_text_broad_query_fetch: bool | None = None,
    ) -> None:
        self.tool_map: dict[str, Callable[[dict[str, Any]], Any]] = {
            "search_text": self._search_text,
            "search_image": self._search_image,
        }
        if not disable_browser_tools:
            self.tool_map.update(
                {
                    "browser_navigate": self._browser_navigate,
                    "browser_get_text": self._browser_get_text,
                    "browser_click": self._browser_click,
                    "browser_type": self._browser_type,
                    "browser_parallel": self._browser_parallel,
                }
            )
        self.disable_browser_tools = disable_browser_tools
        self.retry_attempts = max(1, int(retry_attempts or 1))
        self.retry_min_seconds = max(0.0, float(retry_min_seconds or 0.0))
        self.retry_max_seconds = max(self.retry_min_seconds, float(retry_max_seconds or 0.0))
        self.search_text_max_top_k = max(
            1, int(search_text_max_top_k or os.getenv("SEARCH_TEXT_MAX_TOP_K", "3"))
        )
        self.search_text_max_chars = max(
            100, int(search_text_max_chars or os.getenv("SEARCH_TEXT_MAX_CHARS", "350"))
        )
        self.search_tool_max_concurrency = max(
            1, int(search_tool_max_concurrency or _env_int("SEARCH_TOOL_MAX_CONCURRENCY", "2"))
        )
        self.browser_tool_max_concurrency = max(
            1, int(browser_tool_max_concurrency or _env_int("BROWSER_TOOL_MAX_CONCURRENCY", "2"))
        )
        self.search_semaphore = _shared_semaphore("search", self.search_tool_max_concurrency)
        self.browser_semaphore = _shared_semaphore("browser", self.browser_tool_max_concurrency)
        if search_text_default_fetch is None:
            search_text_default_fetch = _env_bool("SEARCH_TEXT_DEFAULT_FETCH", "0")
        self.search_text_default_fetch = bool(search_text_default_fetch)
        if search_image_default_fetch is None:
            search_image_default_fetch = _env_bool("SEARCH_IMAGE_DEFAULT_FETCH", "0")
        self.search_image_default_fetch = bool(search_image_default_fetch)
        if search_text_broad_query_fetch is None:
            search_text_broad_query_fetch = _env_bool("SEARCH_TEXT_BROAD_QUERY_FETCH", "0")
        self.search_text_broad_query_fetch = bool(search_text_broad_query_fetch)

    @property
    def schemas(self) -> list[dict[str, Any]]:
        if not self.disable_browser_tools:
            return TOOLS_SCHEMA
        return [
            tool
            for tool in TOOLS_SCHEMA
            if not tool.get("function", {}).get("name", "").startswith("browser_")
        ]

    def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name not in self.tool_map:
            return {"ok": False, "error": f"Unknown tool: {tool_name}"}
        attempts = self._attempts_for(tool_name)
        last_result: Any = None
        last_error = ""
        for attempt in range(1, attempts + 1):
            try:
                with self._semaphore_for(tool_name):
                    result = self.tool_map[tool_name](arguments)
                error = self.result_error(result)
                if not error:
                    if attempt > 1 and isinstance(result, dict):
                        result = {**result, "retry_attempts": attempt}
                    return result
                last_result = result
                last_error = error
                if attempt >= attempts or self._is_non_retryable_error(error) or not self._is_retryable_error(error):
                    break
            except Exception as exc:
                last_result = {
                    "ok": False,
                    "error": f"Tool {tool_name} raised {type(exc).__name__}: {exc}",
                }
                last_error = str(last_result["error"])
                if (
                    attempt >= attempts
                    or self._is_non_retryable_error(last_error)
                    or not self._is_retryable_error(last_error)
                ):
                    break
            self._sleep_before_retry(attempt)
        if isinstance(last_result, dict):
            return {**last_result, "retry_attempts": min(attempts, max(1, attempt))}
        return last_result

    def _attempts_for(self, tool_name: str) -> int:
        if tool_name.startswith(("browser_", "search_")):
            return self.retry_attempts
        return 1

    def _semaphore_for(self, tool_name: str) -> threading.BoundedSemaphore:
        if tool_name.startswith("search_"):
            return self.search_semaphore
        if tool_name.startswith("browser_"):
            return self.browser_semaphore
        return _shared_semaphore("other", 10_000)

    def _is_non_retryable_error(self, error: str) -> bool:
        text = error.lower()
        hard_markers = (
            "api key",
            "invalid token",
            "permission denied",
            "authentication failed",
        )
        return any(marker in text for marker in hard_markers)

    def _sleep_before_retry(self, attempt: int) -> None:
        delay = min(self.retry_max_seconds, self.retry_min_seconds * (2 ** (attempt - 1)))
        if delay > 0:
            time.sleep(delay)

    def _is_retryable_error(self, error: str) -> bool:
        text = error.lower()
        retryable_markers = (
            "timeout",
            "timed out",
            "500",
            "502",
            "503",
            "504",
            "401",
            "403",
            "unauthorized",
            "forbidden",
            "service temporarily unavailable",
            "internal server error",
            "connection",
            "session",
            "proxy-error",
            "temporarily",
            "rate limit",
            "too many requests",
        )
        return any(marker in text for marker in retryable_markers)

    def serialize_result(self, result: Any) -> str:
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False)
        return str(result)

    def result_error(self, result: Any) -> str:
        if isinstance(result, dict) and result.get("ok") is False:
            return str(result.get("error", "unknown error"))
        if isinstance(result, list):
            for item in result:
                if isinstance(item, dict):
                    text = " ".join(str(item.get(k, "")) for k in ("error", "snippet", "content"))
                    if "proxy-error" in text.lower() or item.get("ok") is False:
                        return text[:500]
        text = str(result)
        lower_text = text.lower()
        if (
            text.startswith("[ERROR]")
            or "proxy-error" in lower_text
            or "ok=false" in lower_text
            or "timeout" in lower_text
            or "timed out" in lower_text
            or "http 500" in lower_text
            or "http 502" in lower_text
            or "http 503" in lower_text
            or "http 504" in lower_text
        ):
            return text[:500]
        return ""

    def _search_text(self, args: dict[str, Any]) -> Any:
        from tools.search_tool import search_text

        query = str(args.get("query", "")).strip()
        top_k = max(1, min(int(args.get("top_k", 3)), self.search_text_max_top_k))
        force_fetch = _env_bool("SEARCH_TEXT_FORCE_FETCH", "0")
        requested_fetch = bool(args.get("fetch", self.search_text_default_fetch or force_fetch))
        fetch = requested_fetch and self._should_fetch_search_text(query)
        return search_text(
            query=query,
            top_k=top_k,
            fetch=fetch,
            max_chars=max(
                100,
                min(int(args.get("max_chars", self.search_text_max_chars)), self.search_text_max_chars),
            ),
        )

    def _should_fetch_search_text(self, query: str) -> bool:
        if _env_bool("SEARCH_TEXT_FORCE_FETCH", "0"):
            return True
        if not self.search_text_default_fetch:
            return False
        words = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", query.lower())
        if len(words) <= 4 and not re.search(r"https?://|site:|\"[^\"]+\"", query):
            return self.search_text_broad_query_fetch
        return True

    def _search_image(self, args: dict[str, Any]) -> Any:
        from tools.search_tool import search_image

        image = args.get("image_url") or args.get("image") or ""
        if not str(image).strip():
            return {
                "ok": False,
                "error": "search_image requires image_url or image; no image was available",
                "error_type": "missing_image",
            }
        fetch = self._should_fetch_search_image(args)
        return search_image(
            image=image,
            top_k=int(args.get("top_k", 1)),
            fetch=fetch,
            max_chars=int(args.get("max_chars", 500)),
        )

    def _should_fetch_search_image(self, args: dict[str, Any]) -> bool:
        if _env_bool("SEARCH_IMAGE_FORCE_FETCH", "0"):
            return bool(args.get("fetch", True))
        if not self.search_image_default_fetch:
            return False
        return bool(args.get("fetch", True))

    def _browser_navigate(self, args: dict[str, Any]) -> Any:
        from tools.browser_tool import browser_navigate

        return browser_navigate(**args)

    def _browser_get_text(self, args: dict[str, Any]) -> Any:
        from tools.browser_tool import browser_get_text

        return browser_get_text(**args)

    def _browser_click(self, args: dict[str, Any]) -> Any:
        from tools.browser_tool import browser_click

        return browser_click(**args)

    def _browser_type(self, args: dict[str, Any]) -> Any:
        from tools.browser_tool import browser_type

        return browser_type(**args)

    def _browser_parallel(self, args: dict[str, Any]) -> Any:
        from tools.browser_tool import browser_parallel

        bounded_args = dict(args)
        requested = int(bounded_args.get("max_concurrency", self.browser_tool_max_concurrency))
        bounded_args["max_concurrency"] = max(1, min(requested, self.browser_tool_max_concurrency))
        return browser_parallel(**bounded_args)

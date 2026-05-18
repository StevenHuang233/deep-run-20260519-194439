"""Tool environment that reuses the existing service adapters."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

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
            "description": "Reverse image search. Accepts image or image_url and returns ranked web results.",
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

    def __init__(self) -> None:
        self.tool_map: dict[str, Callable[[dict[str, Any]], Any]] = {
            "search_text": self._search_text,
            "search_image": self._search_image,
            "browser_navigate": self._browser_navigate,
            "browser_get_text": self._browser_get_text,
            "browser_click": self._browser_click,
            "browser_type": self._browser_type,
            "browser_parallel": self._browser_parallel,
        }

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return TOOLS_SCHEMA

    def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name not in self.tool_map:
            return {"ok": False, "error": f"Unknown tool: {tool_name}"}
        try:
            return self.tool_map[tool_name](arguments)
        except Exception as exc:
            return {
                "ok": False,
                "error": f"Tool {tool_name} raised {type(exc).__name__}: {exc}",
            }

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
        if text.startswith("[ERROR]") or "proxy-error" in text.lower():
            return text[:500]
        return ""

    def _search_text(self, args: dict[str, Any]) -> Any:
        from tools.search_tool import search_text

        return search_text(
            query=args.get("query", ""),
            top_k=int(args.get("top_k", 3)),
            fetch=bool(args.get("fetch", True)),
            max_chars=int(args.get("max_chars", 500)),
        )

    def _search_image(self, args: dict[str, Any]) -> Any:
        from tools.search_tool import search_image

        image = args.get("image") or args.get("image_url") or ""
        return search_image(
            image=image,
            top_k=int(args.get("top_k", 1)),
            fetch=bool(args.get("fetch", True)),
            max_chars=int(args.get("max_chars", 500)),
        )

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

        return browser_parallel(**args)

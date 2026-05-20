"""Model policy helpers for bounded auxiliary models."""

from __future__ import annotations

import re


def model_size_billion(model_name: str) -> float | None:
    """Best-effort parser for names such as qwen3-32b or Qwen2.5-7B."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", model_name or "")
    if not match:
        return None
    return float(match.group(1))


def assert_model_within_limit(model_name: str, max_billion: float) -> None:
    """Reject auxiliary model names that explicitly exceed the project limit."""
    size = model_size_billion(model_name)
    if size is not None and size > max_billion:
        raise ValueError(
            f"Auxiliary model {model_name!r} appears to be {size:g}B, "
            f"which exceeds the configured <= {max_billion:g}B limit."
        )

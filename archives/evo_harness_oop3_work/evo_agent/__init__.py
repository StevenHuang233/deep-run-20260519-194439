"""From-scratch OOP self-evolving agent harness.

This package intentionally leaves the original harness files untouched.
It reuses the existing browser/search tool modules as service adapters while
rebuilding the agent control, memory, gate, and benchmark flow.
"""

from .harness import HarnessOrchestrator, run_task

__all__ = ["HarnessOrchestrator", "run_task"]

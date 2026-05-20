"""Benchmark CSV runner."""

from __future__ import annotations

from .config import HarnessConfig
from .harness import HarnessOrchestrator


class BenchmarkRunner:
    """Runs the target benchmark.csv through the OOP harness."""

    def __init__(self, task_file: str, config: HarnessConfig | None = None):
        self.config = config or HarnessConfig()
        self.orchestrator = HarnessOrchestrator(task_file=task_file, config=self.config)

    def run(
        self,
        output_path: str | None = None,
        limit: int | None = None,
        start: int = 0,
        trajectory_dir: str | None = None,
        workers: int = 1,
    ) -> list[dict]:
        return self.orchestrator.run_file(
            output_path=output_path,
            limit=limit,
            start=start,
            trajectory_dir=trajectory_dir,
            workers=workers,
        )

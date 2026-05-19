"""CLI entry point compatible with the original task_runner shape."""

from __future__ import annotations

import argparse
import base64
import logging
from pathlib import Path

try:
    from evo_agent.config import HarnessConfig
    from evo_agent.dreamer import MemoryDreamer
    from evo_agent.harness import HarnessOrchestrator, run_task
except ImportError:
    from .evo_agent.config import HarnessConfig
    from .evo_agent.dreamer import MemoryDreamer
    from .evo_agent.harness import HarnessOrchestrator, run_task


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OOP self-evolving Qwen Agent Harness")
    p.add_argument("--instruction", "-i", default=None, help="Task instruction text")
    p.add_argument("--task-id", "-t", default=None, help="Optional task ID")
    p.add_argument("--max-steps", "-s", type=int, default=None, help="Max ReAct steps")
    p.add_argument("--llm-url", default=None, help="OpenAI-compatible base URL")
    p.add_argument("--model", default=None, help="Model name")
    p.add_argument("--traj-dir", default=None, help="Trajectory output directory")
    p.add_argument("--image", default=None, help="Local path to input image or base64")
    p.add_argument("--image-url", default=None, help="Online input image URL")

    p.add_argument("--task-file", default=None, help="CSV/JSONL task file for batch run")
    p.add_argument("--output", default=None, help="Prediction JSONL output path")
    p.add_argument("--limit", type=int, default=None, help="Optional batch limit")
    p.add_argument("--start", type=int, default=0, help="Start index for batch run")
    p.add_argument("--resume", action="store_true", help="Append to output and skip existing indices")
    p.add_argument("--strict", action="store_true", help="Stop batch immediately on uncaught per-case errors")
    p.add_argument("--dream-memory", action="store_true", help="Consolidate failed trajectories into memory")
    p.add_argument("--dream-after-run", action="store_true", help="Run memory consolidation after a batch run")
    p.add_argument("--dream-report", default=None, help="Dream memory report JSON path")
    p.add_argument("--dream-limit", type=int, default=None, help="Max recent trajectories to review")
    p.add_argument("--dream-min-failures", type=int, default=None, help="Minimum failed trajectories before writing memory")
    return p.parse_args()


def _read_image_b64(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if p.exists() and p.is_file():
        return base64.b64encode(p.read_bytes()).decode("ascii")
    # Accept raw base64 for convenience, used by benchmark loader internally.
    return path


def main() -> None:
    args = _parse_args()
    config = HarnessConfig()
    if args.llm_url:
        config.llm_base_url = args.llm_url
    if args.model:
        config.model_name = args.model
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.traj_dir:
        config.trajectory_dir = args.traj_dir
    if args.dream_report:
        config.dream_report_path = args.dream_report

    def run_dream_memory() -> dict:
        dreamer = MemoryDreamer(config)
        report = dreamer.run(
            trajectory_dir=args.traj_dir or config.trajectory_dir,
            report_path=args.dream_report or config.dream_report_path,
            limit=args.dream_limit,
            min_failures=args.dream_min_failures,
        )
        print(
            "Dream memory complete: "
            f"{report['failures_considered']} failures, "
            f"{len(report['memory_updates'])} memory updates"
        )
        print(f"Dream report: {args.dream_report or config.dream_report_path}")
        return report

    if args.dream_memory and not args.task_file:
        run_dream_memory()
        return

    if args.task_file:
        orch = HarnessOrchestrator(task_file=args.task_file, config=config)
        results = orch.run_file(
            output_path=args.output,
            limit=args.limit,
            start=args.start,
            trajectory_dir=args.traj_dir,
            resume=args.resume,
            continue_on_error=not args.strict,
        )
        print(f"Batch complete: {len(results)} cases")
        print(f"Output: {args.output or Path(config.result_dir) / 'predictions.jsonl'}")
        if args.dream_after_run or args.dream_memory:
            run_dream_memory()
        return

    if not args.instruction:
        raise SystemExit("--instruction is required unless --task-file is provided")

    task = {
        "instruction": args.instruction,
        "image": args.image or "",
        "image_b64": _read_image_b64(args.image),
        "image_url": args.image_url,
    }
    if args.task_id:
        task["id"] = args.task_id

    result = run_task(
        task,
        max_steps=args.max_steps,
        llm_base_url=args.llm_url,
        model_name=args.model,
        trajectory_dir=args.traj_dir,
    )
    print("\n" + "=" * 60)
    print("TASK COMPLETE")
    print("=" * 60)
    print(f"Task ID:  {result['task_id']}")
    print(f"Steps:    {result['steps']}")
    print(f"Traj:     {result['trajectory_path']}")
    print(f"\nAnswer:\n{result['answer']}")


if __name__ == "__main__":
    main()

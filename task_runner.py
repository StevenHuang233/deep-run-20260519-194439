"""CLI entry point compatible with the original task_runner shape."""

from __future__ import annotations

import argparse
import base64
import logging
import os
from pathlib import Path
import time

try:
    from evo_agent.config import HarnessConfig
    from evo_agent.dataset_adapter import attach_gold_answers, prepare_answered_dataset
    from evo_agent.dreamer import MemoryDreamer
    from evo_agent.harness import HarnessOrchestrator, run_task
    from evo_agent.judge import JudgeConfig, judge_predictions
    from evo_agent.submission import export_group_submission
except ImportError:
    from .evo_agent.config import HarnessConfig
    from .evo_agent.dataset_adapter import attach_gold_answers, prepare_answered_dataset
    from .evo_agent.dreamer import MemoryDreamer
    from .evo_agent.harness import HarnessOrchestrator, run_task
    from .evo_agent.judge import JudgeConfig, judge_predictions
    from .evo_agent.submission import export_group_submission


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
    p.add_argument("--image-dir", default=None, help="Base directory for relative image paths in JSONL tasks")
    p.add_argument("--dataset", choices=["2wiki", "simplevqa"], default=None, help="Adapt a local answered dataset for safe inference")
    p.add_argument("--dataset-root", default="/inspire/qb-ilm2/project/26summer-camp-01/26210500/datasets", help="Root containing 2wiki.jsonl and simpleVQA/SimpleVQA.jsonl")
    p.add_argument("--dataset-task-output", default=None, help="Prepared no-answer task JSONL path")
    p.add_argument("--dataset-gold-output", default=None, help="Prepared gold-answer sidecar JSONL path")
    p.add_argument("--output", default=None, help="Prediction JSONL output path")
    p.add_argument("--limit", type=int, default=None, help="Optional batch limit")
    p.add_argument("--start", type=int, default=0, help="Start index for batch run")
    p.add_argument("--workers", type=int, default=1, help="Number of cases to run concurrently")
    p.add_argument("--run-id", default=None, help="Optional run ID used for per-run outputs and memory")
    p.add_argument("--shared-memory", action="store_true", help="Use MEMORY_DB_PATH/global memory instead of per-run batch memory")
    p.add_argument("--fresh-memory", action="store_true", help="Reset this run's per-run memory before starting")
    p.add_argument("--resume", action="store_true", help="Append to output and skip existing indices")
    p.add_argument("--strict", action="store_true", help="Stop batch immediately on uncaught per-case errors")
    p.add_argument("--dream-memory", action="store_true", help="Consolidate failed trajectories into memory")
    p.add_argument("--dream-after-run", action="store_true", help="Run memory consolidation after a batch run")
    p.add_argument("--dream-report", default=None, help="Dream memory report JSON path")
    p.add_argument("--dream-limit", type=int, default=None, help="Max recent trajectories to review")
    p.add_argument("--dream-min-failures", type=int, default=None, help="Minimum failed trajectories before writing memory")

    p.add_argument("--judge-after-run", action="store_true", help="Run LLM-as-a-judge scoring after batch predictions")
    p.add_argument("--judge-file", default=None, help="Existing prediction JSONL to score without running tasks")
    p.add_argument("--judge-gold-file", default=None, help="Gold-answer sidecar JSONL to join before judging")
    p.add_argument("--judge-output", default=None, help="Judge detail JSONL output path")
    p.add_argument("--judge-llm-url", default=None, help="OpenAI-compatible judge base URL; /v1 is appended if omitted")
    p.add_argument("--judge-model", default=None, help="Judge model name")
    p.add_argument("--judge-limit", type=int, default=None, help="Optional judge row limit")
    p.add_argument("--judge-start", type=int, default=0, help="Start index for judging")
    p.add_argument("--judge-resume", action="store_true", help="Append judge output and skip existing indices")
    p.add_argument("--group-id", default=None, help="Group id; when set, export group_{id}.json/csv/zip after batch")
    p.add_argument("--submission-dir", default=None, help="Directory for group submission files")
    p.add_argument("--submission-from", default=None, help="Existing prediction JSONL to export without running tasks")
    p.add_argument("--submission-benchmark", default=None, help="Benchmark CSV for submission export; defaults to --task-file")
    p.add_argument("--submission-traj-dir", default=None, help="Trajectory dir for submission export; defaults to --traj-dir/config")
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

    def run_judge(prediction_path: str) -> dict:
        judge_config = JudgeConfig()
        if args.judge_llm_url:
            judge_config.base_url = args.judge_llm_url
        if args.judge_model:
            judge_config.model_name = args.judge_model
        judge_input_path = prediction_path
        if args.judge_gold_file:
            joined_path = str(Path(prediction_path).with_suffix(Path(prediction_path).suffix + ".with_gold.jsonl"))
            judge_input_path = str(
                attach_gold_answers(
                    prediction_path=prediction_path,
                    gold_path=args.judge_gold_file,
                    output_path=joined_path,
                )
            )
        summary = judge_predictions(
            prediction_path=judge_input_path,
            output_path=args.judge_output,
            config=judge_config,
            limit=args.judge_limit,
            start=args.judge_start,
            resume=args.judge_resume,
        )
        print(
            "Judge complete: "
            f"{summary['correct']}/{summary['total']} correct, "
            f"accuracy={summary['accuracy']:.4f}"
        )
        if judge_input_path != prediction_path:
            print(f"Judge input with gold: {judge_input_path}")
        print(f"Judge output: {summary['judge_output_path']}")
        return summary

    if args.judge_file and not args.task_file:
        run_judge(args.judge_file)
        return

    def export_submission(prediction_path: str, benchmark_path: str) -> dict:
        if not args.group_id:
            raise SystemExit("--group-id is required for submission export")
        summary = export_group_submission(
            prediction_path=prediction_path,
            benchmark_path=args.submission_benchmark or benchmark_path,
            trajectory_dir=args.submission_traj_dir or args.traj_dir or config.trajectory_dir,
            group_id=args.group_id,
            output_dir=args.submission_dir or config.result_dir,
        )
        print(
            "Submission export complete: "
            f"{summary['answered']}/{summary['total_questions']} answered"
        )
        print(f"Submission CSV: {summary['csv_path']}")
        print(f"Submission JSON: {summary['json_path']}")
        print(f"Submission ZIP: {summary['zip_path']}")
        if summary["missing_answer_indices"]:
            print(f"Missing answers: {summary['missing_answer_indices']}")
        if summary["missing_trajectory_indices"]:
            print(f"Missing trajectories: {summary['missing_trajectory_indices']}")
        return summary

    if args.submission_from and not args.task_file:
        benchmark_path = args.submission_benchmark
        if not benchmark_path:
            raise SystemExit("--submission-benchmark is required with --submission-from")
        export_submission(args.submission_from, benchmark_path)
        return

    if args.dataset:
        prediction_output = args.output or str(
            Path(config.result_dir) / f"{args.dataset}_predictions.jsonl"
        )
        prepared_dir = Path(prediction_output).resolve().parent / "prepared_tasks"
        gold_dir = Path(prediction_output).resolve().parent / "gold"
        prepared = prepare_answered_dataset(
            args.dataset,
            dataset_root=args.dataset_root,
            task_output=args.dataset_task_output or prepared_dir / f"{args.dataset}.tasks.jsonl",
            gold_output=args.dataset_gold_output or gold_dir / f"{args.dataset}.gold.jsonl",
            limit=args.limit,
            start=args.start,
        )
        args.task_file = str(prepared.task_path)
        if prepared.image_dir and not args.image_dir:
            args.image_dir = str(prepared.image_dir)
        args.judge_gold_file = args.judge_gold_file or str(prepared.gold_path)
        # --start/--limit already sliced the source dataset while preparing.
        # The batch runner must process the prepared task file from its row 0.
        args.start = 0
        args.limit = prepared.total
        print(
            "Prepared dataset: "
            f"{prepared.name} total={prepared.total} "
            f"tasks={prepared.task_path} gold={prepared.gold_path}"
        )

    if args.task_file:
        prediction_output = args.output or str(Path(config.result_dir) / "predictions.jsonl")
        if not args.shared_memory and not os.getenv("MEMORY_DB_PATH"):
            run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
            memory_dir = Path(prediction_output).resolve().parent / "run_memories"
            config.memory_db_path = str(memory_dir / f"memory_{run_id}.json")
            memory_dir.mkdir(parents=True, exist_ok=True)
            memory_path = Path(config.memory_db_path)
            if args.fresh_memory:
                memory_path.write_text("[]", encoding="utf-8")
            elif not memory_path.exists():
                memory_path.write_text("[]", encoding="utf-8")
        orch = HarnessOrchestrator(task_file=args.task_file, image_dir=args.image_dir, config=config)
        results = orch.run_file(
            output_path=prediction_output,
            limit=args.limit,
            start=args.start,
            trajectory_dir=args.traj_dir,
            resume=args.resume,
            continue_on_error=not args.strict,
            workers=args.workers,
        )
        print(f"Batch complete: {len(results)} cases")
        print(f"Output: {prediction_output}")
        if args.group_id:
            export_submission(prediction_output, args.task_file)
        if args.judge_after_run:
            run_judge(prediction_output)
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

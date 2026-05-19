"""Contract checks for the from-scratch harness."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from evo_agent.config import HarnessConfig
from evo_agent.environment import ToolEnvironment
from evo_agent.harness import HarnessOrchestrator, _detect_image_suffix
from evo_agent.memory import SkepticalMemoryManager
from evo_agent.model_policy import assert_model_within_limit
from evo_agent.trajectory import Trajectory
from evo_agent.types import Role, TaskCase


class InterfaceContractTests(unittest.TestCase):
    def test_tool_schema_names_match_existing_services(self) -> None:
        names = [tool["function"]["name"] for tool in ToolEnvironment().schemas]
        self.assertEqual(
            names,
            [
                "search_text",
                "search_image",
                "browser_navigate",
                "browser_get_text",
                "browser_click",
                "browser_type",
                "browser_parallel",
            ],
        )

    def test_config_reads_environment_at_instance_creation(self) -> None:
        old_value = os.environ.get("MODEL_NAME")
        try:
            os.environ["MODEL_NAME"] = "contract-model"
            self.assertEqual(HarnessConfig().model_name, "contract-model")
        finally:
            if old_value is None:
                os.environ.pop("MODEL_NAME", None)
            else:
                os.environ["MODEL_NAME"] = old_value

    def test_auxiliary_model_limit_blocks_larger_models(self) -> None:
        assert_model_within_limit("qwen3-32b", 32)
        with self.assertRaises(ValueError):
            assert_model_within_limit("qwen3-72b", 32)

    def test_data_uri_image_detection(self) -> None:
        png_1x1 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
            "/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
        )
        suffix, mime = _detect_image_suffix(f"data:image/png;base64,{png_1x1}")
        self.assertEqual((suffix, mime), (".png", "image/png"))

    def test_mock_batch_output_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "mini.csv"
            csv_path.write_text("problem,image,answer\nWhat?,,Answer\n", encoding="utf-8")
            output_path = root / "predictions.jsonl"
            config = HarnessConfig(
                mock_llm=True,
                result_dir=str(root / "outputs"),
                trajectory_dir=str(root / "trajectories"),
                memory_db_path=str(root / "memory.json"),
            )
            orch = HarnessOrchestrator(task_file=str(csv_path), config=config)
            orch.run_file(output_path=str(output_path))
            row = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(set(row), {"index", "instruction", "image", "answer", "pred"})
            self.assertEqual(row["pred"], "answer")

    def test_recent_step_context_keeps_system_user_and_latest_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            traj = Trajectory("ctx", output_dir=tmp)
            traj.write(Role.SYSTEM, "system", step_id=0)
            traj.write(Role.USER, "user", step_id=0)
            traj.write(Role.ASSISTANT, "old", step_id=1)
            traj.write(Role.TOOL, "old observation", step_id=1, tool_call_id="a")
            traj.write(Role.ASSISTANT, "new", step_id=2)
            messages = traj.to_messages(recent_steps=1)
            self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant"])
            self.assertEqual(messages[-1]["content"], "new")

    def test_resume_skips_existing_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "mini.csv"
            csv_path.write_text(
                "problem,image,answer\nFirst,,A\nSecond,,B\n",
                encoding="utf-8",
            )
            output_path = root / "predictions.jsonl"
            output_path.write_text(
                json.dumps(
                    {
                        "index": 0,
                        "instruction": "First",
                        "image": "",
                        "answer": "A",
                        "pred": "a",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = HarnessConfig(
                mock_llm=True,
                result_dir=str(root / "outputs"),
                trajectory_dir=str(root / "trajectories"),
                memory_db_path=str(root / "memory.json"),
            )
            orch = HarnessOrchestrator(task_file=str(csv_path), config=config)
            results = orch.run_file(output_path=str(output_path), resume=True)
            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(results), 1)
            self.assertEqual([row["index"] for row in rows], [0, 1])

    def test_llm_retry_counts_transient_failures(self) -> None:
        class _Completions:
            def __init__(self) -> None:
                self.calls = 0

            def create(self, **_kwargs):
                self.calls += 1
                if self.calls < 3:
                    raise RuntimeError("temporary 500")
                return {"ok": True}

        class _Chat:
            def __init__(self) -> None:
                self.completions = _Completions()

        class _Client:
            def __init__(self) -> None:
                self.chat = _Chat()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = HarnessConfig(
                mock_llm=True,
                llm_retry_attempts=3,
                llm_retry_min_seconds=0,
                llm_retry_max_seconds=0,
                result_dir=str(root / "outputs"),
                trajectory_dir=str(root / "trajectories"),
                memory_db_path=str(root / "memory.json"),
            )
            orch = HarnessOrchestrator(config=config)
            orch.client = _Client()
            response, attempts = orch._call_llm_with_retry({})
            self.assertEqual(response, {"ok": True})
            self.assertEqual(attempts, 3)

    def test_single_case_reflection_retry_recovers_empty_prediction(self) -> None:
        class _Message:
            def __init__(self, content: str) -> None:
                self.content = content
                self.tool_calls = None
                self.reasoning_content = ""

        class _Choice:
            def __init__(self, content: str) -> None:
                self.message = _Message(content)

        class _Response:
            def __init__(self, content: str) -> None:
                self.choices = [_Choice(content)]
                self.usage = None

        class _Completions:
            def __init__(self) -> None:
                self.contents = ["", "<answer>Recovered</answer>"]

            def create(self, **_kwargs):
                return _Response(self.contents.pop(0))

        class _Chat:
            def __init__(self) -> None:
                self.completions = _Completions()

        class _Client:
            def __init__(self) -> None:
                self.chat = _Chat()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = HarnessConfig(
                mock_llm=True,
                max_steps=1,
                case_reflection_attempts=1,
                case_reflection_max_steps=1,
                result_dir=str(root / "outputs"),
                trajectory_dir=str(root / "trajectories"),
                memory_db_path=str(root / "memory.json"),
            )
            orch = HarnessOrchestrator(config=config)
            orch.config.mock_llm = False
            orch.client = _Client()
            result = orch.run_case(
                TaskCase(index=0, instruction="Need an answer", answer="Recovered", task_id="retry")
            )
            events = [
                row.get("event_type")
                for row in Path(result["trajectory_path"]).read_text(encoding="utf-8").splitlines()
                for row in [json.loads(row)]
            ]
            self.assertEqual(result["pred"], "recovered")
            self.assertEqual(result["exit_status"], "self_reflection_submitted")
            self.assertIn("self_reflection_retry_start", events)

    def test_batch_continues_on_uncaught_case_error(self) -> None:
        class _FailingOrchestrator(HarnessOrchestrator):
            def run_case(self, case, max_steps=None, trajectory_dir=None):
                if case.index == 0:
                    raise RuntimeError("boom")
                return {
                    "task_id": case.task_id,
                    "answer": "<answer>B</answer>",
                    "pred": "b",
                    "steps": 1,
                    "trajectory_path": "",
                    "summary": {},
                    "success": True,
                    "failure_reason": "",
                    "exit_status": "submitted",
                    "api_calls": 0,
                    "total_tokens": 0,
                    "reflection": None,
                    "memory_write": None,
                    "applied_rules": [],
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "mini.csv"
            csv_path.write_text(
                "problem,image,answer\nFirst,,A\nSecond,,B\n",
                encoding="utf-8",
            )
            output_path = root / "predictions.jsonl"
            config = HarnessConfig(
                mock_llm=True,
                result_dir=str(root / "outputs"),
                trajectory_dir=str(root / "trajectories"),
                memory_db_path=str(root / "memory.json"),
            )
            orch = _FailingOrchestrator(task_file=str(csv_path), config=config)
            results = orch.run_file(output_path=str(output_path), continue_on_error=True)
            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            status = json.loads((root / "predictions.jsonl.status.json").read_text(encoding="utf-8"))
            self.assertEqual(len(results), 2)
            self.assertEqual(rows[0]["pred"], "unknown")
            self.assertEqual(rows[1]["pred"], "b")
            self.assertEqual(status["exit_status_counts"]["uncaught_RuntimeError_fallback"], 1)

    def test_export_logs_fills_empty_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_path = root / "predictions.jsonl"
            config = HarnessConfig(
                mock_llm=True,
                fallback_answer="fallback",
                result_dir=str(root / "outputs"),
                trajectory_dir=str(root / "trajectories"),
                memory_db_path=str(root / "memory.json"),
            )
            orch = HarnessOrchestrator(config=config)
            orch.export_logs(str(output_path), 0, "Q", "", "", "")
            row = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(row["pred"], "fallback")

    def test_memory_prunes_to_max_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = SkepticalMemoryManager(str(Path(tmp) / "memory.json"), max_rules=2)
            for idx in range(4):
                memory.auto_dream_deduplication(
                    f"[Context {idx}] -> Action: strategy {idx} is Necessary to achieve Goal G (Confidence: should)",
                    base_keywords=[f"k{idx}"],
                    advisor=lambda rule, _similar, _task: {
                        "operation": "ADD",
                        "rule": rule,
                        "reason": "force_add_for_prune_contract",
                    },
                )
            self.assertEqual(len(memory.memory_db), 2)
            self.assertLessEqual(len(json.loads((Path(tmp) / "memory.json").read_text(encoding="utf-8"))), 2)


if __name__ == "__main__":
    unittest.main()

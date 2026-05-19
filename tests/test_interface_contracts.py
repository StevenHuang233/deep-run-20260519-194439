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
from evo_agent.model_policy import assert_model_within_limit
from evo_agent.trajectory import Trajectory
from evo_agent.types import Role


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


if __name__ == "__main__":
    unittest.main()

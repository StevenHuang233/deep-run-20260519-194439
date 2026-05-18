"""Contract checks for the from-scratch harness."""

from __future__ import annotations

import base64
import json
import os
import tempfile
import unittest
from pathlib import Path

from evo_agent.config import HarnessConfig
from evo_agent.environment import ToolEnvironment
from evo_agent.harness import HarnessOrchestrator, _detect_image_suffix
from evo_agent.model_policy import assert_model_within_limit


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


if __name__ == "__main__":
    unittest.main()

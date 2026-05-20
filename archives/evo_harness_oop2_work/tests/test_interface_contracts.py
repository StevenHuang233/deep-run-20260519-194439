"""Contract checks for the from-scratch harness."""

from __future__ import annotations

import json
import os
import csv
import subprocess
import tempfile
import unittest
from pathlib import Path

from evo_agent.config import HarnessConfig, normalize_openai_base_url
from evo_agent.dreamer import MemoryDreamer
from evo_agent.environment import ToolEnvironment
from evo_agent.harness import HarnessOrchestrator, _detect_image_suffix
from evo_agent.judge import JudgeConfig, judge_predictions
from evo_agent.memory import SkepticalMemoryManager
from evo_agent.model_policy import assert_model_within_limit
from evo_agent.submission import export_group_submission
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

    def test_browser_tools_can_be_disabled_independently(self) -> None:
        names = [tool["function"]["name"] for tool in ToolEnvironment(disable_browser_tools=True).schemas]
        self.assertEqual(names, ["search_text", "search_image"])

    def test_search_image_schema_accepts_url_or_local_image(self) -> None:
        search_image_schema = next(
            tool for tool in ToolEnvironment().schemas if tool["function"]["name"] == "search_image"
        )
        parameters = search_image_schema["function"]["parameters"]
        self.assertIn("image_url", parameters["properties"])
        self.assertIn("image", parameters["properties"])
        self.assertEqual(parameters["required"], [])

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

    def test_proxy_base_url_is_normalized_to_openai_v1(self) -> None:
        self.assertEqual(normalize_openai_base_url("https://host/proxy/8000/"), "https://host/proxy/8000/v1")
        self.assertEqual(normalize_openai_base_url("https://host/proxy/8000/v1"), "https://host/proxy/8000/v1")

    def test_tool_dispatch_retries_transient_errors(self) -> None:
        env = ToolEnvironment(retry_attempts=3, retry_min_seconds=0, retry_max_seconds=0)
        calls = {"count": 0}

        def flaky_tool(_args):
            calls["count"] += 1
            if calls["count"] < 3:
                return {"ok": False, "error": "HTTP 503 Service Temporarily Unavailable"}
            return {"ok": True, "value": "done"}

        env.tool_map["search_text"] = flaky_tool
        self.assertEqual(env.dispatch("search_text", {"query": "x"}), {"ok": True, "value": "done", "retry_attempts": 3})
        self.assertEqual(calls["count"], 3)

    def test_tool_dispatch_does_not_retry_auth_errors(self) -> None:
        env = ToolEnvironment(retry_attempts=3, retry_min_seconds=0, retry_max_seconds=0)
        calls = {"count": 0}

        def auth_error(_args):
            calls["count"] += 1
            return {"ok": False, "error": "invalid API key"}

        env.tool_map["search_text"] = auth_error
        result = env.dispatch("search_text", {"query": "x"})
        self.assertEqual(calls["count"], 1)
        self.assertEqual(result["retry_attempts"], 1)

    def test_tool_dispatch_retries_http_status_errors(self) -> None:
        env = ToolEnvironment(retry_attempts=2, retry_min_seconds=0, retry_max_seconds=0)
        calls = {"count": 0}

        def http_error(_args):
            calls["count"] += 1
            return {"ok": False, "error": "HTTP 403 forbidden"}

        env.tool_map["browser_navigate"] = http_error
        result = env.dispatch("browser_navigate", {"url": "https://example.invalid"})
        self.assertEqual(calls["count"], 2)
        self.assertEqual(result["retry_attempts"], 2)

    def test_broad_search_text_fetch_is_disabled_by_default(self) -> None:
        env = ToolEnvironment(search_text_broad_query_fetch=False)
        self.assertFalse(env._should_fetch_search_text("found a box"))
        self.assertFalse(env._should_fetch_search_text('"found a box"'))

    def test_search_text_fetch_can_be_enabled_for_specific_runs(self) -> None:
        env = ToolEnvironment(search_text_default_fetch=True, search_text_broad_query_fetch=False)
        self.assertFalse(env._should_fetch_search_text("found a box"))
        self.assertTrue(env._should_fetch_search_text('"found a box"'))

    def test_refusal_or_timeout_text_is_not_a_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = HarnessConfig(
                mock_llm=True,
                result_dir=str(root / "outputs"),
                trajectory_dir=str(root / "trajectories"),
                memory_db_path=str(root / "memory.json"),
            )
            orch = HarnessOrchestrator(config=config)
            self.assertFalse(orch._has_parseable_prediction("<answer>Unable to answer due to search timeout</answer>"))
            self.assertFalse(orch._has_parseable_prediction("<answer>Unable to definitively identify the person with certainty</answer>"))
            self.assertFalse(orch._has_parseable_prediction("<answer>Cannot identify the specific source from available results</answer>"))
            self.assertFalse(orch._has_parseable_prediction("<answer>Unknown - insufficient evidence to identify the founder</answer>"))
            self.assertFalse(orch._has_parseable_prediction("<answer>Based on the search results, the most likely answer is **Jane Doe**, though definitive confirmation requires additional verification.</answer>"))
            self.assertFalse(orch._has_parseable_prediction("<answer>工具失败，无法确定</answer>"))
            self.assertTrue(orch._has_parseable_prediction("<answer>上海创智学院</answer>"))

    def test_resume_valid_only_requeues_empty_and_bad_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_path = root / "predictions.jsonl"
            output_path.write_text(
                json.dumps({"index": 0, "pred": "good answer"})
                + "\n"
                + json.dumps({"index": 1, "pred": ""})
                + "\n"
                + json.dumps({"index": 2, "pred": "Unknown - insufficient evidence"})
                + "\n",
                encoding="utf-8",
            )
            config = HarnessConfig(
                mock_llm=True,
                result_dir=str(root / "outputs"),
                trajectory_dir=str(root / "trajectories"),
                memory_db_path=str(root / "memory.json"),
            )
            orch = HarnessOrchestrator(config=config)
            self.assertEqual(orch._read_completed_indices(str(output_path)), {0, 1, 2})
            self.assertEqual(orch._read_completed_indices(str(output_path), valid_only=True), {0})

    def test_final_main_step_disables_tools_and_adds_force_answer_prompt(self) -> None:
        class _Message:
            content = "<answer>Best guess</answer>"
            tool_calls = None
            reasoning_content = ""

        class _Choice:
            message = _Message()

        class _Response:
            choices = [_Choice()]
            usage = None

        class _Completions:
            def __init__(self) -> None:
                self.kwargs = []

            def create(self, **kwargs):
                self.kwargs.append(kwargs)
                return _Response()

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
                reflection_enabled=False,
                result_dir=str(root / "outputs"),
                trajectory_dir=str(root / "trajectories"),
                memory_db_path=str(root / "memory.json"),
            )
            orch = HarnessOrchestrator(config=config)
            orch.config.mock_llm = False
            client = _Client()
            orch.client = client
            result = orch.run_case(TaskCase(index=0, instruction="Q", task_id="force"))
            self.assertEqual(result["pred"], "best guess")
            self.assertNotIn("tools", client.chat.completions.kwargs[0])
            rows = [
                json.loads(line)
                for line in Path(result["trajectory_path"]).read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(any(row.get("event_type") == "force_answer_prompt:1" for row in rows))

    def test_bad_final_answer_gets_short_answer_repair_turn(self) -> None:
        class _Message:
            tool_calls = None
            reasoning_content = ""

            def __init__(self, content: str) -> None:
                self.content = content

        class _Choice:
            def __init__(self, content: str) -> None:
                self.message = _Message(content)

        class _Response:
            usage = None

            def __init__(self, content: str) -> None:
                self.choices = [_Choice(content)]

        class _Completions:
            def __init__(self) -> None:
                self.contents = [
                    "<answer>Based on the search results, the most likely answer is **Jane Doe**, though definitive confirmation requires additional verification.</answer>",
                    "<answer>Jane Doe</answer>",
                ]
                self.kwargs = []

            def create(self, **kwargs):
                self.kwargs.append(kwargs)
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
                reflection_enabled=False,
                result_dir=str(root / "outputs"),
                trajectory_dir=str(root / "trajectories"),
                memory_db_path=str(root / "memory.json"),
            )
            orch = HarnessOrchestrator(config=config)
            orch.config.mock_llm = False
            client = _Client()
            orch.client = client
            result = orch.run_case(TaskCase(index=0, instruction="Who?", task_id="repair"))
            self.assertEqual(result["pred"], "jane doe")
            self.assertEqual(result["exit_status"], "answer_repaired")
            self.assertEqual(len(client.chat.completions.kwargs), 2)
            self.assertNotIn("tools", client.chat.completions.kwargs[-1])
            rows = [
                json.loads(line)
                for line in Path(result["trajectory_path"]).read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(any(row.get("event_type") == "answer_repair_start" for row in rows))
            self.assertTrue(any(row.get("event_type") == "answer_repair_status" for row in rows))

    def test_search_tool_concurrency_limit_is_configurable(self) -> None:
        env = ToolEnvironment(
            retry_attempts=1,
            search_tool_max_concurrency=1,
            retry_min_seconds=0,
            retry_max_seconds=0,
        )
        active = {"count": 0, "max": 0}
        lock = __import__("threading").Lock()

        def slow_tool(_args):
            with lock:
                active["count"] += 1
                active["max"] = max(active["max"], active["count"])
            __import__("time").sleep(0.05)
            with lock:
                active["count"] -= 1
            return {"ok": True}

        env.tool_map["search_text"] = slow_tool
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda _: env.dispatch("search_text", {"query": "x"}), range(3)))
        self.assertEqual(active["max"], 1)

    def test_tool_gate_allows_multiple_transient_failures_before_blocking(self) -> None:
        from evo_agent.gate import ToolGateBlocked, ToolGateMiddleware

        gate = ToolGateMiddleware()
        gate.track_error("browser_navigate", "HTTP 500")
        gate.track_error("browser_navigate", "HTTP 500")
        with self.assertRaises(ToolGateBlocked):
            gate.track_error("browser_navigate", "HTTP 500")

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

    def test_trajectory_can_omit_images_after_first_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            traj = Trajectory("image_ctx", output_dir=tmp)
            traj.write(Role.SYSTEM, "system", step_id=0)
            traj.write(
                Role.USER,
                [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    {"type": "text", "text": "Question text"},
                ],
                step_id=0,
            )
            full_messages = traj.to_messages(include_images=True)
            slim_messages = traj.to_messages(include_images=False)
            self.assertIsInstance(full_messages[1]["content"], list)
            self.assertIsInstance(slim_messages[1]["content"], str)
            self.assertIn("Question text", slim_messages[1]["content"])
            self.assertIn("Image omitted", slim_messages[1]["content"])

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

    def test_no_gold_non_empty_prediction_has_unknown_eval_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = HarnessConfig(
                mock_llm=True,
                result_dir=str(root / "outputs"),
                trajectory_dir=str(root / "trajectories"),
                memory_db_path=str(root / "memory.json"),
            )
            result = HarnessOrchestrator(config=config).run_case(
                TaskCase(index=0, instruction="Q", answer="", task_id="no_gold")
            )
            self.assertTrue(result["has_pred"])
            self.assertEqual(result["eval_status"], "unknown")
            self.assertFalse(result["success"])

    def test_pseudo_tool_call_is_recovered_from_reasoning_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = HarnessConfig(
                mock_llm=True,
                result_dir=str(root / "outputs"),
                trajectory_dir=str(root / "trajectories"),
                memory_db_path=str(root / "memory.json"),
            )
            orch = HarnessOrchestrator(config=config)
            pseudo_text = """
            <tool_call>
            <function=search_text>
            <parameter=query>
            found a box listening to music
            </parameter>
            <parameter=top_k>5</parameter>
            <parameter=fetch>true</parameter>
            </function>
            </tool_call>
            """
            calls = orch._extract_pseudo_tool_calls("", pseudo_text)
            self.assertEqual(len(calls), 1)
            name, args, _tc_id = orch._parse_tool_call(calls[0])
            self.assertEqual(name, "search_text")
            self.assertEqual(args["query"], "found a box listening to music")
            self.assertEqual(args["top_k"], 5)
            self.assertTrue(args["fetch"])
            self.assertFalse(orch._has_parseable_prediction(pseudo_text))

    def test_csv_image_url_is_preserved_for_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "mini.csv"
            image_url = "https://example.invalid/image.jpg"
            csv_path.write_text(
                "problem,image,image_url,answer\n"
                f"Which object?,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=,{image_url},object\n",
                encoding="utf-8",
            )
            config = HarnessConfig(
                mock_llm=True,
                result_dir=str(root / "outputs"),
                trajectory_dir=str(root / "trajectories"),
                memory_db_path=str(root / "memory.json"),
            )
            orch = HarnessOrchestrator(task_file=str(csv_path), config=config)
            case = next(orch.load_next_case())
            self.assertEqual(case.image_url, image_url)
            self.assertTrue(case.image_b64)
            self.assertTrue(case.image.endswith(".png"))
            user_json = json.dumps(orch._build_user_content(case), ensure_ascii=False)
            self.assertIn(f"image_url: {image_url}", user_json)
            self.assertIn("local_image_path_for_vision", user_json)
            self.assertIn("local_image_path_for_search", user_json)

    def test_csv_image_column_url_is_used_as_image_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "mini.csv"
            image_url = "https://example.invalid/direct.jpg"
            csv_path.write_text(
                "problem,image,answer\n"
                f"Which object?,{image_url},object\n",
                encoding="utf-8",
            )
            config = HarnessConfig(
                mock_llm=True,
                result_dir=str(root / "outputs"),
                trajectory_dir=str(root / "trajectories"),
                memory_db_path=str(root / "memory.json"),
            )
            case = next(HarnessOrchestrator(task_file=str(csv_path), config=config).load_next_case())
            self.assertEqual(case.image_url, image_url)
            self.assertFalse(case.image_b64)
            self.assertEqual(case.image, "")

    def test_csv_base64_image_exposes_local_path_for_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "mini.csv"
            csv_path.write_text(
                "problem,image,answer\n"
                "Which object?,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=,object\n",
                encoding="utf-8",
            )
            config = HarnessConfig(
                mock_llm=True,
                result_dir=str(root / "outputs"),
                trajectory_dir=str(root / "trajectories"),
                memory_db_path=str(root / "memory.json"),
            )
            orch = HarnessOrchestrator(task_file=str(csv_path), config=config)
            case = next(orch.load_next_case())
            user_content = orch._build_user_content(case)
            user_json = json.dumps(user_content, ensure_ascii=False)
            user_text = user_content[1]["text"]
            self.assertEqual(case.image_url, "")
            self.assertTrue(case.image)
            self.assertIn(f"local_image_path_for_search: {case.image}", user_text)

    def test_search_image_recovery_uses_local_image_when_no_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "sample.jpg"
            image_path.write_bytes(b"\xff\xd8\xffmock")
            config = HarnessConfig(
                mock_llm=True,
                result_dir=str(root / "outputs"),
                trajectory_dir=str(root / "trajectories"),
                memory_db_path=str(root / "memory.json"),
            )
            orch = HarnessOrchestrator(config=config)
            calls = []

            def fake_dispatch(_name, args):
                calls.append(dict(args))
                return {"ok": True, "results": []}

            orch.env.dispatch = fake_dispatch
            result, error = orch._dispatch_tool_with_recovery(
                "search_image",
                {"top_k": 1},
                TaskCase(index=0, instruction="Q", image=str(image_path)),
            )
            self.assertEqual(error, "")
            self.assertEqual(result, {"ok": True, "results": []})
            self.assertEqual(calls[0]["image"], str(image_path))

    def test_browser_failed_url_is_blocked_on_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = HarnessConfig(
                mock_llm=True,
                browser_url_failure_limit=1,
                result_dir=str(root / "outputs"),
                trajectory_dir=str(root / "trajectories"),
                memory_db_path=str(root / "memory.json"),
            )
            orch = HarnessOrchestrator(config=config)
            state = orch._new_tool_state()
            args = {"url": "https://www.wbaltv.com/article/kaylies-gift-of-life/23547566?x=1"}
            self.assertEqual(orch._precheck_tool_call("browser_navigate", args, state), "")
            orch._record_tool_outcome("browser_navigate", args, "HTTP 500 Internal Server Error", state)
            blocked = orch._precheck_tool_call(
                "browser_navigate",
                {"url": "https://www.wbaltv.com/article/kaylies-gift-of-life/23547566?y=2"},
                state,
            )
            self.assertIn("browser_url_blocked", blocked)

    def test_search_budget_blocks_duplicate_and_excess_searches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = HarnessConfig(
                mock_llm=True,
                max_search_calls_per_case=2,
                result_dir=str(root / "outputs"),
                trajectory_dir=str(root / "trajectories"),
                memory_db_path=str(root / "memory.json"),
            )
            orch = HarnessOrchestrator(config=config)
            state = orch._new_tool_state()
            args = {"query": "found a box listening to music"}
            orch._record_tool_outcome("search_text", args, "", state)
            self.assertIn("repeated_search_blocked", orch._precheck_tool_call("search_text", args, state))
            self.assertEqual(
                orch._precheck_tool_call("search_text", {"query": "different query"}, state),
                "",
            )
            orch._record_tool_outcome("search_text", {"query": "different query"}, "", state)
            self.assertIn(
                "search_budget_exhausted",
                orch._precheck_tool_call("search_text", {"query": "third query"}, state),
            )

    def test_search_results_are_filtered_for_case_novelty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = HarnessConfig(
                mock_llm=True,
                search_stale_result_limit=1,
                result_dir=str(root / "outputs"),
                trajectory_dir=str(root / "trajectories"),
                memory_db_path=str(root / "memory.json"),
            )
            orch = HarnessOrchestrator(config=config)
            state = orch._new_tool_state()
            first, first_error = orch._enforce_search_novelty(
                [
                    {"title": "A", "url": "https://example.com/a", "snippet": "one"},
                    {"title": "B", "url": "https://example.com/b", "snippet": "two"},
                ],
                state,
            )
            self.assertEqual(first_error, "")
            self.assertEqual(len(first), 2)
            second, second_error = orch._enforce_search_novelty(
                [
                    {"title": "A again", "url": "https://example.com/a?utm=1", "snippet": "dup"},
                    {"title": "C", "url": "https://other.example/c", "snippet": "new"},
                ],
                state,
            )
            self.assertEqual(second_error, "")
            self.assertEqual([item["url"] for item in second], ["https://other.example/c"])
            third, third_error = orch._enforce_search_novelty(
                [{"title": "A", "url": "https://example.com/a", "snippet": "dup"}],
                state,
            )
            self.assertEqual(third_error, "stale_search_results")
            self.assertFalse(third["ok"])
            self.assertIn(
                "stale_search_loop_blocked",
                orch._precheck_tool_call("search_text", {"query": "new query"}, state),
            )

    def test_2wiki_jsonl_string_context_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jsonl_path = root / "2wiki.jsonl"
            jsonl_path.write_text(
                json.dumps(
                    {
                        "_id": "abc",
                        "question": "Where was the director born?",
                        "context": json.dumps([["Film", ["Film was directed by A."]], ["A", ["A was born in B."]]]),
                        "answer": "B",
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
            case = next(HarnessOrchestrator(task_file=str(jsonl_path), config=config).load_next_case())
            self.assertEqual(case.task_id, "2wiki_abc")
            self.assertIn("Film: Film was directed by A.", case.instruction)

    def test_simplevqa_jsonl_uses_local_image_for_vision_and_url_for_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            image_dir.mkdir()
            image_path = image_dir / "sample.jpg"
            image_path.write_bytes(b"\xff\xd8\xffmock")
            jsonl_path = root / "simplevqa.jsonl"
            jsonl_path.write_text(
                json.dumps(
                    {
                        "data_id": 7,
                        "question": "What is shown?",
                        "image": "sample.jpg",
                        "image_url": "https://example.invalid/expired.jpg",
                        "answer": "sample",
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
            case = next(
                HarnessOrchestrator(
                    task_file=str(jsonl_path), image_dir=str(image_dir), config=config
                ).load_next_case()
            )
            self.assertEqual(case.image, str(image_path))
            self.assertTrue(case.image_b64)
            self.assertEqual(case.image_url, "https://example.invalid/expired.jpg")
            user_content = HarnessOrchestrator(config=config)._build_user_content(case)
            user_json = json.dumps(user_content, ensure_ascii=False)
            self.assertIn("image_url: https://example.invalid/expired.jpg", user_json)
            self.assertIn("local_image_path_for_vision", user_json)
            self.assertIn("local_image_path_for_search", user_json)

    def test_llm_judge_scores_prediction_jsonl_with_mock_client(self) -> None:
        class _Message:
            content = '{"score": 1, "verdict": "correct", "reason": "same answer"}'

        class _Choice:
            message = _Message()

        class _Response:
            choices = [_Choice()]

        class _Completions:
            def create(self, **_kwargs):
                return _Response()

        class _Chat:
            completions = _Completions()

        class _Client:
            chat = _Chat()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                json.dumps({"index": 0, "instruction": "Q", "answer": "A", "pred": "A"})
                + "\n"
                + json.dumps({"index": 1, "instruction": "Q", "answer": "B", "pred": ""})
                + "\n",
                encoding="utf-8",
            )
            output = root / "judge.jsonl"
            summary = judge_predictions(
                predictions,
                output_path=output,
                config=JudgeConfig(base_url="https://judge.invalid/proxy/8001", model_name="Qwen3-32B"),
                client=_Client(),
            )
            self.assertEqual(summary["total"], 2)
            self.assertEqual(summary["correct"], 1)
            self.assertEqual(summary["accuracy"], 0.5)
            self.assertTrue((root / "judge.jsonl.summary.json").exists())

    def test_llm_judge_falls_back_when_model_returns_non_json(self) -> None:
        class _Message:
            content = "not json"

        class _Choice:
            message = _Message()

        class _Response:
            choices = [_Choice()]

        class _Completions:
            def create(self, **_kwargs):
                return _Response()

        class _Chat:
            completions = _Completions()

        class _Client:
            chat = _Chat()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                json.dumps({"index": 0, "instruction": "Q", "answer": "1934年", "pred": "1934年"}) + "\n",
                encoding="utf-8",
            )
            summary = judge_predictions(
                predictions,
                output_path=root / "judge.jsonl",
                config=JudgeConfig(base_url="https://judge.invalid/proxy/8001", model_name="Qwen3-32B"),
                client=_Client(),
            )
            self.assertEqual(summary["correct"], 1)

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

    def test_trajectory_can_reset_existing_case_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            traj = Trajectory("same_case", output_dir=tmp)
            traj.write(Role.USER, "old", step_id=1)
            reset = Trajectory("same_case", output_dir=tmp, reset=True)
            reset.write(Role.SYSTEM, "new", step_id=0)
            rows = reset.read_all()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["content"], "new")

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
            self.assertEqual(rows[0]["pred"], "")
            self.assertEqual(rows[1]["pred"], "b")
            self.assertEqual(status["exit_status_counts"]["uncaught_RuntimeError"], 1)

    def test_parallel_batch_records_fast_completed_case_first(self) -> None:
        class _SlowFirstOrchestrator(HarnessOrchestrator):
            def run_case(self, case, max_steps=None, trajectory_dir=None):
                if case.index == 0:
                    import time as _time

                    _time.sleep(0.25)
                return {
                    "task_id": case.task_id,
                    "answer": f"<answer>{case.index}</answer>",
                    "pred": str(case.index),
                    "steps": 1,
                    "trajectory_path": "",
                    "summary": {},
                    "success": True,
                    "has_pred": True,
                    "eval_status": "unknown",
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
                "problem,image,answer\nFirst,,\nSecond,,\n",
                encoding="utf-8",
            )
            output_path = root / "predictions.jsonl"
            config = HarnessConfig(
                mock_llm=True,
                result_dir=str(root / "outputs"),
                trajectory_dir=str(root / "trajectories"),
                memory_db_path=str(root / "memory.json"),
            )
            orch = _SlowFirstOrchestrator(task_file=str(csv_path), config=config)
            orch.run_file(output_path=str(output_path), workers=2)
            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["index"], 1)
            self.assertEqual({row["index"] for row in rows}, {0, 1})

    def test_batch_cli_uses_per_run_memory_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jsonl_path = root / "simplevqa.jsonl"
            image_path = root / "sample.jpg"
            image_path.write_bytes(b"\xff\xd8\xffmock")
            jsonl_path.write_text(
                json.dumps(
                    {
                        "data_id": 1,
                        "question": "What?",
                        "image": "sample.jpg",
                        "image_url": "https://example.invalid/image.jpg",
                        "answer": "A",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_path = root / "predictions.jsonl"
            env = os.environ.copy()
            env["MOCK_LLM"] = "1"
            result = subprocess.run(
                [
                    os.sys.executable,
                    "-m",
                    "task_runner",
                    "--task-file",
                    str(jsonl_path),
                    "--image-dir",
                    str(root),
                    "--output",
                    str(output_path),
                    "--run-id",
                    "contract",
                ],
                cwd=str(Path(__file__).resolve().parents[1]),
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("Batch complete", result.stdout)
            self.assertTrue((root / "run_memories" / "memory_contract.json").exists())

    def test_fresh_memory_resets_existing_per_run_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "mini.csv"
            csv_path.write_text("problem,image,answer\nWhat?,,Answer\n", encoding="utf-8")
            output_path = root / "predictions.jsonl"
            memory_dir = root / "run_memories"
            memory_dir.mkdir()
            memory_path = memory_dir / "memory_contract.json"
            memory_path.write_text(json.dumps([{"rule": "stale"}]), encoding="utf-8")
            env = os.environ.copy()
            env["MOCK_LLM"] = "1"
            subprocess.run(
                [
                    os.sys.executable,
                    "-m",
                    "task_runner",
                    "--task-file",
                    str(csv_path),
                    "--output",
                    str(output_path),
                    "--run-id",
                    "contract",
                    "--fresh-memory",
                ],
                cwd=str(Path(__file__).resolve().parents[1]),
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertNotIn("stale", memory_path.read_text(encoding="utf-8"))

    def test_export_logs_preserves_empty_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_path = root / "predictions.jsonl"
            config = HarnessConfig(
                mock_llm=True,
                result_dir=str(root / "outputs"),
                trajectory_dir=str(root / "trajectories"),
                memory_db_path=str(root / "memory.json"),
            )
            orch = HarnessOrchestrator(config=config)
            orch.export_logs(str(output_path), 0, "Q", "", "", "")
            row = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(row["pred"], "")

    def test_group_submission_export_fills_answer_and_zips_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark_path = root / "benchmark.csv"
            benchmark_path.write_text(
                "problem,image,answer\nQ0,,\nQ1,,\n",
                encoding="utf-8",
            )
            predictions_path = root / "predictions.jsonl"
            predictions_path.write_text(
                json.dumps({"index": 0, "instruction": "Q0", "answer": "", "pred": "A0"})
                + "\n"
                + json.dumps({"index": 1, "instruction": "Q1", "answer": "", "pred": "A1"})
                + "\n",
                encoding="utf-8",
            )
            traj_dir = root / "trajectories"
            traj_dir.mkdir()
            Trajectory("benchmark_000", output_dir=str(traj_dir)).write(Role.USER, "Q0", step_id=0)
            Trajectory("benchmark_001", output_dir=str(traj_dir)).write(Role.USER, "Q1", step_id=0)

            summary = export_group_submission(
                prediction_path=predictions_path,
                benchmark_path=benchmark_path,
                trajectory_dir=traj_dir,
                group_id="42",
                output_dir=root / "submission",
            )

            self.assertEqual(summary["answered"], 2)
            self.assertTrue(Path(summary["zip_path"]).exists())
            with open(summary["csv_path"], newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([row["answer"] for row in rows], ["A0", "A1"])
            payload = json.loads(Path(summary["json_path"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["total_questions"], 2)
            self.assertEqual(payload["trajectories"][0]["trajectory"][0]["content"], "Q0")

    def test_failed_case_gets_five_model_attempts_then_empty_pred(self) -> None:
        class _Message:
            content = ""
            tool_calls = None
            reasoning_content = ""

        class _Choice:
            message = _Message()

        class _Response:
            choices = [_Choice()]
            usage = None

        class _Completions:
            def __init__(self) -> None:
                self.calls = 0

            def create(self, **_kwargs):
                self.calls += 1
                return _Response()

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
                min_model_attempts=5,
                case_reflection_attempts=0,
                case_reflection_max_steps=1,
                result_dir=str(root / "outputs"),
                trajectory_dir=str(root / "trajectories"),
                memory_db_path=str(root / "memory.json"),
            )
            orch = HarnessOrchestrator(config=config)
            orch.config.mock_llm = False
            client = _Client()
            orch.client = client
            result = orch.run_case(TaskCase(index=0, instruction="No answer", task_id="empty"))
            self.assertEqual(result["pred"], "")
            self.assertEqual(result["api_calls"], 8)
            self.assertEqual(client.chat.completions.calls, 8)
            self.assertEqual(result["answer_repair"]["attempts"], 2)
            self.assertIsNotNone(result["forced_answer"])
            self.assertEqual(result["exit_status"], "self_reflection_limits_exceeded")

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

    def test_memory_validator_rejects_generic_trigger_and_normalizes_action(self) -> None:
        from evo_agent.memory import _tokens

        with tempfile.TemporaryDirectory() as tmp:
            memory = SkepticalMemoryManager(str(Path(tmp) / "memory.json"), max_rules=2)
            generic = {
                "memory_type": "reflection",
                "trigger_pattern": "This reusable task-solving pattern is detected.",
                "procedure": ["Use the memory only as a general strategy."],
                "avoid": ["Do not treat this memory as a factual answer."],
                "content": "Change query terms.",
            }
            self.assertFalse(memory.validate_candidate_memory(generic)["ok"])
            specific = memory._candidate_from_rule(
                "[When browser tools repeatedly fail on the same URL] -> Action: Action: switch to search_text or answer from observations is Necessary to achieve Goal G (Confidence: should)",
                base_keywords=["浏览器失败"],
                memory_type="reflection",
                task_type="general",
                episode_id="case",
            )
            self.assertTrue(memory.validate_candidate_memory(specific)["ok"])
            self.assertFalse(specific["content"].startswith("Action:"))
            self.assertEqual(_tokens("上海旅游"), {"上海旅游"})

    def test_memory_validator_rejects_low_quality_reflection_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = SkepticalMemoryManager(str(Path(tmp) / "memory.json"), max_rules=2)
            noisy = memory._candidate_from_rule(
                "[When equivalent search queries stop adding new evidence] -> Action: Candidate validation is Necessary to achieve Goal G (Confidence: should)",
                base_keywords=["search", "candidate"],
                memory_type="reflection",
                task_type="simpleqa",
                episode_id="case",
            )
            validation = memory.validate_candidate_memory(noisy)
            self.assertFalse(validation["ok"])
            self.assertEqual(validation["reason"], "low_quality_reflection_memory")

    def test_typed_memory_hybrid_retrieval_and_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory = SkepticalMemoryManager(str(root / "memory.json"), max_rules=10)
            memory.memory_db = [
                memory._normalize_entry(
                    {
                        "memory_id": "bad_2wiki_000001",
                        "memory_type": "bad_pattern",
                        "task_type": "2wiki",
                        "title": "Do not stop at the first-hop entity",
                        "trigger_pattern": "The question asks for an attribute of an intermediate entity.",
                        "trigger_keywords": ["director", "birthplace", "founder", "nationality"],
                        "content": "Continue from the intermediate entity to the final requested attribute.",
                        "procedure": ["Find the intermediate entity.", "Search the final requested attribute."],
                        "avoid": ["Do not answer the intermediate entity."],
                        "source": {"source_type": "test", "source_episode_ids": []},
                        "stats": {
                            "usage_count": 0,
                            "success_count": 0,
                            "failure_count": 0,
                            "confidence": 0.5,
                        },
                        "status": "active",
                        "embedding_text": "Task type: 2wiki. Trigger: director birthplace. Avoid answering intermediate entity.",
                    }
                )
            ]
            memory._save_db()
            retrieved = memory.retrieve_memories(
                "What is the birthplace of the director of X?", task_type="2wiki", top_k=3
            )
            self.assertEqual(retrieved[0]["memory_id"], "bad_2wiki_000001")
            memory.log_usage(["bad_2wiki_000001"], "episode_1", "2wiki", "Q", "planner")
            memory.update_after_episode(["bad_2wiki_000001"], "episode_1", "success")
            updated = memory.memory_db[0]
            self.assertEqual(updated["stats"]["usage_count"], 1)
            self.assertEqual(updated["stats"]["success_count"], 1)
            self.assertGreater(updated["stats"]["confidence"], 0.5)
            self.assertTrue((root / "usage_logs.jsonl").exists())

    def test_dreamer_consolidates_failed_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            traj = Trajectory("failed_case", output_dir=str(root / "trajectories"))
            traj.write(Role.SYSTEM, "system", step_id=0)
            traj.write(Role.USER, "Which entity has the final attribute?", step_id=0)
            traj.write(Role.ASSISTANT, "", step_id=1)
            traj.write_event(
                "run_status",
                {
                    "exit_status": "self_reflection_limits_exceeded",
                    "success": False,
                    "failure_reason": "empty_after_5_model_attempts",
                },
                step_id=5,
            )
            config = HarnessConfig(
                mock_llm=True,
                result_dir=str(root / "outputs"),
                trajectory_dir=str(root / "trajectories"),
                memory_db_path=str(root / "memory.json"),
                dream_report_path=str(root / "outputs" / "dream_report.json"),
            )
            dreamer = MemoryDreamer(config)
            report = dreamer.run()
            self.assertEqual(report["trajectories_reviewed"], 1)
            self.assertEqual(report["failures_considered"], 1)
            self.assertGreaterEqual(len(report["memory_updates"]), 1)
            self.assertTrue((root / "outputs" / "dream_report.json").exists())
            memory_rows = json.loads((root / "memory.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(memory_rows), 1)


if __name__ == "__main__":
    unittest.main()

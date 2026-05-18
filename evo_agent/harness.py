"""OOP ReAct harness with memory, physical gate, and reflection hooks."""

from __future__ import annotations

import base64
import csv
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Generator, Optional

from .compiler import CognitiveCompiler
from .config import HarnessConfig, ensure_output_dirs
from .environment import ToolEnvironment
from .gate import ToolGateBlocked, ToolGateMiddleware
from .memory import SkepticalMemoryManager
from .planner import Planner
from .prompts import SYSTEM_PROMPT
from .trajectory import Trajectory
from .types import Reflection, Role, TaskCase

logger = logging.getLogger("evo_harness.harness")


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _normalize_answer(text: str) -> str:
    text = text or ""
    answer_match = re.search(r"<answer>(.*?)</answer>", text, flags=re.S | re.I)
    if answer_match:
        text = answer_match.group(1)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip("\"'`.,;:!? \t\r\n")
    text = re.sub(r"^[\"'“”‘’\s]+|[\"'“”‘’\s]+$", "", text)
    return text


def _detect_image_suffix(image_b64: str) -> tuple[str, str]:
    raw = base64.b64decode(image_b64[:80] + "===")
    if raw.startswith(b"\x89PNG"):
        return ".png", "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return ".jpg", "image/jpeg"
    if raw.startswith(b"RIFF") and b"WEBP" in raw[:16]:
        return ".webp", "image/webp"
    return ".jpg", "image/jpeg"


def _safe_b64_decode(image_b64: str) -> bytes:
    clean = re.sub(r"\s+", "", image_b64 or "")
    if "," in clean and clean.startswith("data:"):
        clean = clean.split(",", 1)[1]
    padding = "=" * (-len(clean) % 4)
    return base64.b64decode(clean + padding)


class HarnessOrchestrator:
    """Coordinates dataset loading, ReAct execution, logs, memory, and reflection."""

    def __init__(
        self,
        task_file: str = "",
        image_dir: Optional[str] = None,
        base_model: Optional[str] = None,
        config: Optional[HarnessConfig] = None,
    ) -> None:
        self.config = config or HarnessConfig()
        if base_model:
            self.config.model_name = base_model
        ensure_output_dirs(self.config)

        self.task_file = task_file
        self.image_dir = image_dir
        self.base_model = self.config.model_name
        self.planner = Planner()
        self.memory = SkepticalMemoryManager(self.config.memory_db_path)
        self.compiler = CognitiveCompiler(
            ref_model=self.config.reflection_model_name,
            base_url=self.config.reflection_base_url,
            api_key=self.config.reflection_api_key,
            enable_model=self.config.reflection_model_enabled
            or self.config.memory_model_enabled,
            max_model_billion=self.config.reflection_model_max_b,
        )
        self.env = ToolEnvironment()
        self.client = None if self.config.mock_llm else self._create_openai_client()
        self.trajectory_log: list[dict[str, Any]] = []

    def _create_openai_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package is required for real LLM calls. "
                "Install requirements.txt or set MOCK_LLM=1 for local smoke tests."
            ) from exc
        return OpenAI(base_url=self.config.llm_base_url, api_key="EMPTY")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_next_case(self) -> Generator[TaskCase, None, None]:
        """Load cases from benchmark CSV, SimpleVQA JSONL, or 2Wiki JSONL."""
        if not self.task_file:
            return
        path = Path(self.task_file)
        if path.suffix.lower() == ".csv":
            yield from self._load_csv(path)
            return
        if path.suffix.lower() == ".jsonl":
            yield from self._load_jsonl(path)
            return
        raise ValueError(f"Unsupported task file: {path}")

    def _load_csv(self, path: Path) -> Generator[TaskCase, None, None]:
        csv.field_size_limit(2_147_483_647)
        with open(path, encoding="utf-8-sig", newline="") as f:
            for idx, row in enumerate(csv.DictReader(f)):
                instruction = row.get("problem") or row.get("instruction") or ""
                image_value = (row.get("image") or "").strip()
                image_path = ""
                image_b64 = None
                if image_value:
                    image_b64 = image_value
                    image_path = self._materialize_base64_image(idx, image_value)
                yield TaskCase(
                    index=idx,
                    instruction=instruction,
                    answer=row.get("answer") or "",
                    task_id=f"benchmark_{idx:03d}",
                    image=image_path,
                    image_b64=image_b64,
                    metadata={
                        "source": str(path),
                        "raw_image_present": bool(image_value),
                        "submission_image": image_value,
                    },
                )

    def _load_jsonl(self, path: Path) -> Generator[TaskCase, None, None]:
        with open(path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                row = json.loads(line)
                if "question" in row and "image_url" in row:
                    image_path = row.get("image") or ""
                    if self.image_dir and image_path:
                        image_path = str(Path(self.image_dir) / image_path)
                    yield TaskCase(
                        index=idx,
                        instruction=row.get("question", ""),
                        answer=row.get("answer", ""),
                        task_id=f"simplevqa_{row.get('data_id', idx)}",
                        image=image_path,
                        image_url=row.get("image_url"),
                        metadata=row,
                    )
                elif "question" in row:
                    context = row.get("context", {})
                    context_text = self._format_2wiki_context(context)
                    instruction = row.get("question", "")
                    if context_text:
                        instruction = f"{instruction}\n\nContext:\n{context_text}"
                    yield TaskCase(
                        index=idx,
                        instruction=instruction,
                        answer=row.get("answer", ""),
                        task_id=f"2wiki_{row.get('id', idx)}",
                        metadata=row,
                    )

    def _format_2wiki_context(self, context: dict[str, Any], max_chars: int = 12000) -> str:
        titles = context.get("title") or []
        sentences = context.get("sentences") or []
        blocks = []
        for title, sent_list in zip(titles, sentences):
            blocks.append(f"{title}: {' '.join(sent_list)}")
        text = "\n".join(blocks)
        return text[:max_chars]

    def _materialize_base64_image(self, index: int, image_b64: str) -> str:
        suffix, _ = _detect_image_suffix(image_b64)
        out_dir = Path(self.config.result_dir) / "benchmark_images"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"benchmark_{index:03d}{suffix}"
        if not out_path.exists():
            out_path.write_bytes(_safe_b64_decode(image_b64))
        return str(out_path)

    # ------------------------------------------------------------------
    # Prompt and execution
    # ------------------------------------------------------------------
    def inject_historical_guidelines(self, system_prompt: str, rules: list[str]) -> str:
        return self.planner.inject_historical_guidelines(system_prompt, rules)

    def execute_react_cycle(
        self, task_instruction: str, image_name: Optional[str] = None
    ) -> tuple[str, str]:
        case = TaskCase(
            index=0,
            instruction=task_instruction,
            image=image_name or "",
            task_id=f"manual_{uuid.uuid4().hex[:8]}",
        )
        result = self.run_case(case)
        return result["answer"], result["trajectory_path"]

    def run_case(
        self,
        case: TaskCase,
        max_steps: Optional[int] = None,
        trajectory_dir: Optional[str] = None,
    ) -> dict[str, Any]:
        max_steps = max_steps or self.config.max_steps
        trajectory_dir = trajectory_dir or self.config.trajectory_dir
        task_id = case.task_id or str(uuid.uuid4())[:8]
        traj = Trajectory(task_id, output_dir=trajectory_dir)
        gate = ToolGateMiddleware(
            loop_limit=self.config.gate_loop_limit,
            similarity_threshold=self.config.gate_similarity_threshold,
            critical_limit=self.config.gate_critical_limit,
        )

        memory_rules = self.memory.retrieve_top_rules(case.instruction)
        system_prompt = self.inject_historical_guidelines(SYSTEM_PROMPT, memory_rules)
        traj.write(Role.SYSTEM, system_prompt, step_id=0)
        traj.write(Role.USER, self._build_user_content(case), step_id=0)
        traj.write_event(
            "memory_retrieval",
            {"rules": memory_rules, "task_kind": self.planner.task_kind(case.instruction, bool(case.image or case.image_url or case.image_b64))},
            step_id=0,
        )

        final_answer = ""
        failure_reason = ""
        steps_done = 0
        api_calls = 0
        total_tokens_accum = 0
        exit_status = "unknown"

        for step in range(1, max_steps + 1):
            steps_done = step
            messages = traj.to_messages()
            if self.config.mock_llm:
                final_answer = self._mock_answer(case)
                exit_status = "submitted"
                traj.write(
                    Role.ASSISTANT,
                    final_answer,
                    step_id=step,
                    extra={"reasoning_content": "MOCK_LLM local validation path."},
                )
                break

            request_kwargs = {
                "model": self.config.model_name,
                "messages": messages,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "extra_body": {"enable_thinking": True},
            }
            if not self.config.disable_tools:
                request_kwargs["tools"] = self.env.schemas
                request_kwargs["tool_choice"] = "auto"

            try:
                if self.client is None:
                    raise RuntimeError("OpenAI client is unavailable")
                response = self.client.chat.completions.create(**request_kwargs)
                api_calls += 1
            except Exception as exc:
                failure_reason = f"llm_call_failed: {type(exc).__name__}: {exc}"
                traj.write(Role.TOOL, f"[HARNESS ERROR] {failure_reason}", step_id=step)
                break

            choice = response.choices[0]
            msg = choice.message
            content = getattr(msg, "content", None) or ""
            reasoning_content = getattr(msg, "reasoning_content", None) or ""
            usage = getattr(response, "usage", None)
            total_tokens = getattr(usage, "total_tokens", None) if usage else None
            if total_tokens:
                total_tokens_accum += int(total_tokens)
            tool_calls = None if self.config.disable_tools else getattr(msg, "tool_calls", None)

            extra: dict[str, Any] = {}
            if tool_calls:
                extra["tool_calls"] = [self._dump_tool_call(tc) for tc in tool_calls]
            if reasoning_content:
                extra["reasoning_content"] = reasoning_content
            if total_tokens:
                extra["total_tokens"] = total_tokens
            traj.write(Role.ASSISTANT, content, step_id=step, extra=extra or None)

            if not tool_calls:
                if content:
                    final_answer = content
                    exit_status = "submitted"
                    break
                continue

            blocked = False
            for tc in tool_calls:
                fn_name, fn_args, tc_id = self._parse_tool_call(tc)
                try:
                    gate.inspect_call(fn_name, fn_args)
                    raw_result = self.env.dispatch(fn_name, fn_args)
                    error = self.env.result_error(raw_result)
                    if error:
                        gate.track_error(fn_name, error)
                    tool_result = self.env.serialize_result(raw_result)
                except ToolGateBlocked as exc:
                    failure_reason = f"gate_blocked: {exc}"
                    tool_result = _json_dumps({"ok": False, "error": failure_reason})
                    blocked = True
                except Exception as exc:
                    failure_reason = f"tool_dispatch_failed: {type(exc).__name__}: {exc}"
                    tool_result = _json_dumps({"ok": False, "error": failure_reason})
                    blocked = True

                traj.write(
                    Role.TOOL,
                    tool_result,
                    step_id=step,
                    tool_call_id=tc_id,
                    extra={"fn_name": fn_name, "fn_args": fn_args},
                )
                if blocked:
                    exit_status = "gate_blocked"
                    break
            if blocked:
                break
        else:
            failure_reason = f"max_steps_reached:{max_steps}"
            final_answer = "[HARNESS] Max steps reached. Last assistant message above."
            exit_status = "limits_exceeded"

        if failure_reason and exit_status == "unknown":
            exit_status = "failed"
        elif exit_status == "unknown":
            exit_status = "submitted" if final_answer else "empty"

        success = self._judge_success(final_answer, case.answer)
        if case.answer and memory_rules:
            self.memory.apply_expel_reward(memory_rules, success)

        reflection_dict = None
        memory_write = None
        if self.config.reflection_enabled and (failure_reason or (case.answer and not success)):
            reflection = self.compiler.compile_failed_trace(
                traj.read_all(), case.instruction
            )
            reflection_dict = reflection.as_dict()
            traj.write_event("reflection", reflection_dict, step_id=steps_done)
            if reflection.memory_worthy:
                memory_write = self.memory.auto_dream_deduplication(
                    reflection.clin_rule,
                    self._memory_keywords(case, reflection),
                    advisor=self.compiler.advise_memory_update
                    if self.config.memory_model_enabled
                    else None,
                    task_instruction=case.instruction,
                )
                traj.write_event("memory_write", memory_write, step_id=steps_done)

        traj.write_event(
            "run_status",
            {
                "exit_status": exit_status,
                "success": success,
                "failure_reason": failure_reason,
                "api_calls": api_calls,
                "total_tokens": total_tokens_accum,
            },
            step_id=steps_done,
        )

        return {
            "task_id": task_id,
            "answer": final_answer,
            "pred": self.extract_pred(final_answer),
            "steps": steps_done,
            "trajectory_path": str(traj.path),
            "summary": traj.summary(),
            "success": success,
            "failure_reason": failure_reason,
            "exit_status": exit_status,
            "api_calls": api_calls,
            "total_tokens": total_tokens_accum,
            "reflection": reflection_dict,
            "memory_write": memory_write,
            "applied_rules": memory_rules,
        }

    def _build_user_content(self, case: TaskCase) -> Any:
        instruction = case.instruction
        if case.image_url:
            instruction = f"{instruction}\nimage_url: {case.image_url}"
        if case.image:
            instruction = f"{instruction}\nlocal_image_path_for_tools: {case.image}"

        if case.image_b64:
            _, mime = _detect_image_suffix(case.image_b64)
            return [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{case.image_b64}"}},
                {"type": "text", "text": instruction},
            ]
        if case.image_url:
            return [
                {"type": "image_url", "image_url": {"url": case.image_url}},
                {"type": "text", "text": instruction},
            ]
        return instruction

    def _dump_tool_call(self, tc: Any) -> dict[str, Any]:
        if hasattr(tc, "model_dump"):
            return tc.model_dump()
        if isinstance(tc, dict):
            return tc
        return json.loads(json.dumps(tc, default=lambda o: getattr(o, "__dict__", str(o))))

    def _parse_tool_call(self, tc: Any) -> tuple[str, dict[str, Any], str]:
        if isinstance(tc, dict):
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args_text = fn.get("arguments", "{}") or "{}"
            tc_id = tc.get("id", "")
        else:
            name = tc.function.name
            args_text = tc.function.arguments or "{}"
            tc_id = tc.id
        try:
            args = json.loads(args_text)
        except json.JSONDecodeError:
            args = {}
        return name, args, tc_id

    def _judge_success(self, pred: str, answer: str) -> bool:
        if not answer:
            return bool(pred.strip())
        p = _normalize_answer(pred)
        a = _normalize_answer(answer)
        return bool(a and (p == a or a in p))

    def extract_pred(self, answer: str) -> str:
        return _normalize_answer(answer) or answer.strip()

    def _mock_answer(self, case: TaskCase) -> str:
        if case.answer:
            return f"<answer>{case.answer}</answer>"
        if case.image or case.image_b64 or case.image_url:
            return "<answer>MOCK_IMAGE_ANSWER</answer>"
        return "<answer>MOCK_TEXT_ANSWER</answer>"

    def _memory_keywords(self, case: TaskCase, reflection: Reflection) -> list[str]:
        text = " ".join(
            [
                case.instruction,
                case.metadata.get("type", "") if isinstance(case.metadata, dict) else "",
                reflection.failure_type,
                reflection.root_cause,
            ]
        )
        return sorted(set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())))[:80]

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------
    def export_logs(
        self,
        output_path: str,
        index: int,
        instruction: str,
        pred: str,
        answer: str,
        image: str = "",
    ) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "index": index,
                        "instruction": instruction,
                        "image": image,
                        "answer": answer,
                        "pred": pred,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def run_file(
        self,
        output_path: Optional[str] = None,
        limit: Optional[int] = None,
        start: int = 0,
        trajectory_dir: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        results = []
        output_path = output_path or str(Path(self.config.result_dir) / "predictions.jsonl")
        if Path(output_path).exists():
            Path(output_path).unlink()
        for case in self.load_next_case():
            if case.index < start:
                continue
            if limit is not None and len(results) >= limit:
                break
            result = self.run_case(case, trajectory_dir=trajectory_dir)
            export_image = case.metadata.get("submission_image") if isinstance(case.metadata, dict) else None
            self.export_logs(
                output_path,
                index=case.index,
                instruction=case.instruction,
                image=export_image if export_image is not None else (case.image or case.image_url or ""),
                answer=case.answer,
                pred=result["pred"],
            )
            results.append(result)
            logger.info(
                "case %s done steps=%s pred=%s",
                case.task_id,
                result["steps"],
                result["pred"][:120],
            )
        return results


def run_task(
    task: dict,
    max_steps: Optional[int] = None,
    llm_base_url: Optional[str] = None,
    model_name: Optional[str] = None,
    trajectory_dir: Optional[str] = None,
) -> dict[str, Any]:
    """Current-harness-compatible single task entry point."""
    config = HarnessConfig()
    if llm_base_url:
        config.llm_base_url = llm_base_url
    if model_name:
        config.model_name = model_name
    if max_steps:
        config.max_steps = max_steps
    orch = HarnessOrchestrator(config=config)
    case = TaskCase(
        index=0,
        instruction=task["instruction"],
        answer=task.get("answer", ""),
        task_id=task.get("id") or str(uuid.uuid4())[:8],
        image=task.get("image", ""),
        image_b64=task.get("image_b64"),
        image_url=task.get("image_url"),
        metadata=task,
    )
    return orch.run_case(case, max_steps=max_steps, trajectory_dir=trajectory_dir)

"""OOP ReAct harness with memory, physical gate, and reflection hooks."""

from __future__ import annotations

import base64
import csv
import json
import logging
import re
import time
import traceback
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


class LLMCallFailure(RuntimeError):
    """Wrap model-call errors with the number of attempted API calls."""

    def __init__(self, message: str, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts


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
    raw = _safe_b64_decode(image_b64)[:16]
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


def _safe_task_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value or "")
    return safe.strip("._")[:120] or uuid.uuid4().hex[:8]


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
        self.memory = SkepticalMemoryManager(
            self.config.memory_db_path, max_rules=self.config.memory_max_rules
        )
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

    def _call_llm_with_retry(self, request_kwargs: dict[str, Any]) -> tuple[Any, int]:
        max_attempts = max(1, self.config.llm_retry_attempts)
        attempts = 0
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            attempts = attempt
            try:
                return self.client.chat.completions.create(**request_kwargs), attempts
            except Exception as exc:
                last_exc = exc
                if attempt >= max_attempts or self._is_non_retryable_llm_error(exc):
                    break
                sleep_for = min(
                    self.config.llm_retry_max_seconds,
                    self.config.llm_retry_min_seconds * (2 ** (attempt - 1)),
                )
                logger.warning(
                    "LLM call failed on attempt %s/%s: %s. Retrying in %.1fs",
                    attempt,
                    max_attempts,
                    exc,
                    sleep_for,
                )
                time.sleep(max(sleep_for, 0))
        message = f"{type(last_exc).__name__}: {last_exc}" if last_exc else "unknown error"
        raise LLMCallFailure(message, attempts)

    def _is_non_retryable_llm_error(self, exc: Exception) -> bool:
        text = f"{type(exc).__name__}: {exc}".lower()
        markers = (
            "authentication",
            "permissiondenied",
            "permission denied",
            "unauthorized",
            "forbidden",
            "invalid api key",
            "api key",
            "401",
            "403",
            "context length",
            "contextwindow",
            "maximum context",
            "unsupported",
            "not found",
            "404",
        )
        return any(marker in text for marker in markers)

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
        task_id = _safe_task_id(case.task_id or str(uuid.uuid4())[:8])
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
            messages = traj.to_messages(recent_steps=self.config.context_recent_steps)
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
                response, attempts = self._call_llm_with_retry(request_kwargs)
                api_calls += attempts
            except LLMCallFailure as exc:
                api_calls += exc.attempts
                failure_reason = f"llm_call_failed: {type(exc).__name__}: {exc}"
                traj.write(Role.TOOL, f"[HARNESS ERROR] {failure_reason}", step_id=step)
                break
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
            final_answer = ""
            exit_status = "limits_exceeded"

        if failure_reason and exit_status == "unknown":
            exit_status = "failed"
        elif exit_status == "unknown":
            exit_status = "submitted" if final_answer else "empty"

        applied_rules = list(memory_rules)
        reflection_dict = None
        memory_write = None

        if self.config.reflection_enabled and self._should_reflection_retry(
            final_answer, failure_reason, exit_status, case.answer
        ):
            for retry_no in range(1, self._case_reflection_retry_budget() + 1):
                reflection = self.compiler.compile_failed_trace(
                    traj.read_all(), case.instruction
                )
                reflection_dict = reflection.as_dict()
                reflection_dict["retry_attempt"] = retry_no
                traj.write_event("reflection", reflection_dict, step_id=steps_done)

                retry_rule = reflection.clin_rule
                if reflection.memory_worthy:
                    memory_write = self.memory.auto_dream_deduplication(
                        retry_rule,
                        self._memory_keywords(case, reflection),
                        advisor=self.compiler.advise_memory_update
                        if self.config.memory_model_enabled
                        else None,
                        task_instruction=case.instruction,
                    )
                    retry_rule = str(memory_write.get("rule") or retry_rule)
                    traj.write_event("memory_write", memory_write, step_id=steps_done)
                if retry_rule:
                    applied_rules.append(retry_rule)

                retry_result = self._run_reflection_retry(
                    case=case,
                    traj=traj,
                    reflection=reflection,
                    start_step=steps_done,
                    retry_no=retry_no,
                )
                steps_done = int(retry_result["steps_done"])
                api_calls += int(retry_result["api_calls"])
                total_tokens_accum += int(retry_result["total_tokens"])
                if retry_result["answer"]:
                    final_answer = str(retry_result["answer"])
                if retry_result["failure_reason"]:
                    failure_reason = self._append_reason(
                        failure_reason, str(retry_result["failure_reason"])
                    )
                exit_status = str(retry_result["exit_status"] or exit_status)
                if self._has_parseable_prediction(final_answer):
                    break

        success = self._judge_success(final_answer, case.answer)
        if case.answer and applied_rules:
            self.memory.apply_expel_reward(applied_rules, success)

        if (
            reflection_dict is None
            and self.config.reflection_enabled
            and (failure_reason or (case.answer and not success))
        ):
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
            "pred": self._extract_model_pred(final_answer),
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
            "applied_rules": applied_rules,
        }

    def _should_reflection_retry(
        self, final_answer: str, failure_reason: str, exit_status: str, gold_answer: str = ""
    ) -> bool:
        if self._case_reflection_retry_budget() <= 0:
            return False
        if gold_answer and self._has_parseable_prediction(final_answer):
            return not self._judge_success(final_answer, gold_answer)
        if self._has_parseable_prediction(final_answer) and not failure_reason:
            return False
        if exit_status in {"submitted", "self_reflection_submitted"} and self._has_parseable_prediction(final_answer):
            return False
        return True

    def _case_reflection_retry_budget(self) -> int:
        min_retries = max(0, int(self.config.min_model_attempts) - 1)
        configured = max(0, int(self.config.case_reflection_attempts))
        return max(min_retries, configured)

    def _run_reflection_retry(
        self,
        case: TaskCase,
        traj: Trajectory,
        reflection: Reflection,
        start_step: int,
        retry_no: int,
    ) -> dict[str, Any]:
        """Retry the same case with a temporary CLIN rule before falling back."""
        max_steps = max(0, self.config.case_reflection_max_steps)
        gate = ToolGateMiddleware(
            loop_limit=self.config.gate_loop_limit,
            similarity_threshold=self.config.gate_similarity_threshold,
            critical_limit=self.config.gate_critical_limit,
        )
        retry_prompt = (
            "[SELF_REFLECTION_RETRY]\n"
            f"Attempt {retry_no} is a recovery attempt for the same task.\n"
            f"Failure type: {reflection.failure_type}\n"
            f"Root cause: {reflection.root_cause}\n"
            f"Temporary guideline: {reflection.clin_rule}\n"
            "Use the guideline as a skeptical hint. Avoid repeating blocked or unhelpful tool calls. "
            "If enough evidence is already available, answer immediately. "
            "Return exactly one concise final response wrapped as <answer>...</answer>."
        )
        user_step = start_step + 1
        traj.write_event(
            "self_reflection_retry_start",
            {"retry_attempt": retry_no, "max_steps": max_steps, "clin_rule": reflection.clin_rule},
            step_id=start_step,
        )
        traj.write(Role.USER, retry_prompt, step_id=user_step)

        final_answer = ""
        failure_reason = ""
        exit_status = "self_reflection_empty"
        api_calls = 0
        total_tokens_accum = 0
        steps_done = user_step

        for offset in range(1, max_steps + 1):
            step_id = user_step + offset
            steps_done = step_id
            if self.config.mock_llm:
                final_answer = self._mock_answer(case)
                exit_status = "self_reflection_submitted"
                traj.write(
                    Role.ASSISTANT,
                    final_answer,
                    step_id=step_id,
                    extra={"reasoning_content": "MOCK_LLM self-reflection retry path."},
                )
                break

            request_kwargs = {
                "model": self.config.model_name,
                "messages": traj.to_messages(recent_steps=self.config.context_recent_steps),
                "max_tokens": self.config.max_tokens,
                "temperature": max(0.2, min(self.config.temperature, 0.8)),
                "extra_body": {"enable_thinking": True},
            }
            if not self.config.disable_tools:
                request_kwargs["tools"] = self.env.schemas
                request_kwargs["tool_choice"] = "auto"

            try:
                if self.client is None:
                    raise RuntimeError("OpenAI client is unavailable")
                response, attempts = self._call_llm_with_retry(request_kwargs)
                api_calls += attempts
            except LLMCallFailure as exc:
                api_calls += exc.attempts
                failure_reason = f"self_reflection_llm_failed: {type(exc).__name__}: {exc}"
                traj.write(Role.TOOL, f"[HARNESS ERROR] {failure_reason}", step_id=step_id)
                exit_status = "self_reflection_failed"
                break
            except Exception as exc:
                failure_reason = f"self_reflection_llm_failed: {type(exc).__name__}: {exc}"
                traj.write(Role.TOOL, f"[HARNESS ERROR] {failure_reason}", step_id=step_id)
                exit_status = "self_reflection_failed"
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
            traj.write(Role.ASSISTANT, content, step_id=step_id, extra=extra or None)

            if not tool_calls:
                if content:
                    final_answer = content
                    exit_status = "self_reflection_submitted"
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
                    failure_reason = f"self_reflection_gate_blocked: {exc}"
                    tool_result = _json_dumps({"ok": False, "error": failure_reason})
                    blocked = True
                except Exception as exc:
                    failure_reason = f"self_reflection_tool_failed: {type(exc).__name__}: {exc}"
                    tool_result = _json_dumps({"ok": False, "error": failure_reason})
                    blocked = True

                traj.write(
                    Role.TOOL,
                    tool_result,
                    step_id=step_id,
                    tool_call_id=tc_id,
                    extra={"fn_name": fn_name, "fn_args": fn_args},
                )
                if blocked:
                    exit_status = "self_reflection_gate_blocked"
                    break
            if blocked:
                break
        else:
            if not final_answer:
                failure_reason = f"self_reflection_max_steps_reached:{max_steps}"
                exit_status = "self_reflection_limits_exceeded"

        traj.write_event(
            "self_reflection_retry_status",
            {
                "retry_attempt": retry_no,
                "exit_status": exit_status,
                "pred": self._extract_model_pred(final_answer),
                "failure_reason": failure_reason,
                "api_calls": api_calls,
                "total_tokens": total_tokens_accum,
            },
            step_id=steps_done,
        )
        return {
            "answer": final_answer,
            "steps_done": steps_done,
            "failure_reason": failure_reason,
            "exit_status": exit_status,
            "api_calls": api_calls,
            "total_tokens": total_tokens_accum,
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

    def _extract_model_pred(self, answer: str) -> str:
        return self.extract_pred(answer) if self._has_parseable_prediction(answer) else ""

    def _has_parseable_prediction(self, answer: str) -> bool:
        stripped = (answer or "").strip()
        return bool(self.extract_pred(answer)) and not stripped.upper().startswith("[HARNESS]")

    def _append_reason(self, current: str, extra: str) -> str:
        if not extra:
            return current
        if current:
            return f"{current}; {extra}"
        return extra

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
        resume: bool = False,
        continue_on_error: Optional[bool] = None,
    ) -> list[dict[str, Any]]:
        results = []
        output_path = output_path or str(Path(self.config.result_dir) / "predictions.jsonl")
        continue_on_error = self.config.batch_continue_on_error if continue_on_error is None else continue_on_error
        completed_indices = self._read_completed_indices(output_path) if resume else set()
        status_counts: dict[str, int] = {}
        if Path(output_path).exists() and not resume:
            Path(output_path).unlink()
        for case in self.load_next_case():
            if case.index < start:
                continue
            if case.index in completed_indices:
                logger.info("case %s skipped because output already exists", case.task_id)
                continue
            if limit is not None and len(results) >= limit:
                break
            try:
                result = self.run_case(case, trajectory_dir=trajectory_dir)
            except Exception as exc:
                if not continue_on_error:
                    raise
                logger.error("case %s failed with uncaught exception: %s", case.task_id, exc, exc_info=True)
                result = {
                    "task_id": case.task_id or f"case_{case.index}",
                    "answer": "",
                    "pred": "",
                    "steps": 0,
                    "trajectory_path": "",
                    "summary": {},
                    "success": False,
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "exit_status": f"uncaught_{type(exc).__name__}",
                    "api_calls": 0,
                    "total_tokens": 0,
                    "reflection": None,
                    "memory_write": None,
                    "applied_rules": [],
                    "traceback": traceback.format_exc(),
                }
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
            status = str(result.get("exit_status") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            self._write_batch_status(output_path, status_counts, results)
            logger.info(
                "case %s done steps=%s pred=%s",
                case.task_id,
                result["steps"],
                result["pred"][:120],
            )
        return results

    def _write_batch_status(
        self, output_path: str, status_counts: dict[str, int], results: list[dict[str, Any]]
    ) -> None:
        status_path = Path(output_path).with_suffix(Path(output_path).suffix + ".status.json")
        summary = {
            "total_processed_this_run": len(results),
            "exit_status_counts": status_counts,
            "total_api_calls": sum(int(item.get("api_calls", 0)) for item in results),
            "total_tokens": sum(int(item.get("total_tokens", 0)) for item in results),
            "updated_at": time.time(),
        }
        status_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_completed_indices(self, output_path: str) -> set[int]:
        path = Path(output_path)
        if not path.exists():
            return set()
        completed = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row.get("index"), int):
                completed.add(row["index"])
        return completed


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

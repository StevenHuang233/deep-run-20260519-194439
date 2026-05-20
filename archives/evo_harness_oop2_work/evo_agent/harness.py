"""OOP ReAct harness with memory, physical gate, and reflection hooks."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import logging
import re
import threading
import time
import traceback
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from typing import Any, Generator, Optional

from .compiler import CognitiveCompiler
from .config import HarnessConfig, ensure_output_dirs, normalize_openai_base_url
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


def _is_http_url(value: str) -> bool:
    return bool(re.match(r"^https?://", (value or "").strip(), flags=re.I))


def _canonical_url(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    try:
        parts = urlsplit(value)
    except ValueError:
        return value.rstrip("/")
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((scheme, netloc, path, "", ""))


def _url_domain(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    try:
        return urlsplit(value).netloc.lower()
    except ValueError:
        return ""


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
        self.env = ToolEnvironment(
            disable_browser_tools=self.config.disable_browser_tools,
            retry_attempts=self.config.tool_retry_attempts,
            retry_min_seconds=self.config.tool_retry_min_seconds,
            retry_max_seconds=self.config.tool_retry_max_seconds,
            search_text_max_top_k=self.config.search_text_max_top_k,
            search_text_max_chars=self.config.search_text_max_chars,
            search_tool_max_concurrency=self.config.search_tool_max_concurrency,
            browser_tool_max_concurrency=self.config.browser_tool_max_concurrency,
            search_text_default_fetch=self.config.search_text_default_fetch,
            search_image_default_fetch=self.config.search_image_default_fetch,
            search_text_broad_query_fetch=self.config.search_text_broad_query_fetch,
        )
        self.client = None if self.config.mock_llm else self._create_openai_client()
        self._memory_lock = threading.RLock()
        self.trajectory_log: list[dict[str, Any]] = []

    def _create_openai_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package is required for real LLM calls. "
                "Install requirements.txt or set MOCK_LLM=1 for local smoke tests."
            ) from exc
        return OpenAI(base_url=normalize_openai_base_url(self.config.llm_base_url), api_key="EMPTY")

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
                image_url = self._extract_csv_image_url(row)
                image_path = ""
                image_b64 = None
                if image_value:
                    if _is_http_url(image_value):
                        image_url = image_url or image_value
                    else:
                        image_b64 = image_value
                        image_path = self._materialize_base64_image(idx, image_value)
                yield TaskCase(
                    index=idx,
                    instruction=instruction,
                    answer=row.get("answer") or "",
                    task_id=f"benchmark_{idx:03d}",
                    image=image_path,
                    image_b64=image_b64,
                    image_url=image_url,
                    metadata={
                        "source": str(path),
                        "raw_image_present": bool(image_value),
                        "submission_image": image_value,
                        "image_url": image_url,
                    },
                )

    def _extract_csv_image_url(self, row: dict[str, Any]) -> str:
        for key in (
            "image_url",
            "image_urls",
            "url",
            "image_source_url",
            "source_url",
            "web_url",
        ):
            value = (row.get(key) or "").strip()
            if not value:
                continue
            match = re.search(r"https?://[^\s,;]+", value, flags=re.I)
            if match:
                return match.group(0)
        return ""

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
                    image_b64 = self._read_local_image_b64(image_path)
                    yield TaskCase(
                        index=idx,
                        instruction=row.get("question", ""),
                        answer=row.get("answer", ""),
                        task_id=f"simplevqa_{row.get('data_id', idx)}",
                        image=image_path,
                        image_b64=image_b64,
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
                        task_id=f"2wiki_{row.get('_id') or row.get('id') or idx}",
                        metadata=row,
                    )

    def _format_2wiki_context(self, context: Any, max_chars: int = 12000) -> str:
        if isinstance(context, str):
            try:
                context = json.loads(context)
            except json.JSONDecodeError:
                return context[:max_chars]

        if isinstance(context, list):
            blocks = []
            for item in context:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                title, sent_list = item[0], item[1]
                if isinstance(sent_list, list):
                    blocks.append(f"{title}: {' '.join(str(sent) for sent in sent_list)}")
                else:
                    blocks.append(f"{title}: {sent_list}")
            return "\n".join(blocks)[:max_chars]

        if not isinstance(context, dict):
            return str(context)[:max_chars] if context else ""

        titles = context.get("title") or []
        sentences = context.get("sentences") or []
        blocks = []
        for title, sent_list in zip(titles, sentences):
            blocks.append(f"{title}: {' '.join(sent_list)}")
        text = "\n".join(blocks)
        return text[:max_chars]

    def _read_local_image_b64(self, image_path: str) -> str | None:
        if not image_path:
            return None
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            return None
        return base64.b64encode(path.read_bytes()).decode("ascii")

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
        traj = Trajectory(task_id, output_dir=trajectory_dir, reset=True)
        gate = ToolGateMiddleware(
            loop_limit=self.config.gate_loop_limit,
            similarity_threshold=self.config.gate_similarity_threshold,
            critical_limit=self.config.gate_critical_limit,
        )

        task_kind = self.planner.task_kind(
            case.instruction, bool(case.image or case.image_url or case.image_b64)
        )
        with self._memory_lock:
            retrieved_memories = self.memory.retrieve_memories(
                case.instruction, task_type=task_kind, top_k=3
            )
            memory_rules = self.memory.format_memories_for_prompt(retrieved_memories)
            applied_memory_ids = [
                str(mem.get("memory_id") or mem.get("id"))
                for mem in retrieved_memories
                if mem.get("memory_id") or mem.get("id")
            ]
            if applied_memory_ids:
                self.memory.log_usage(
                    applied_memory_ids,
                    episode_id=task_id,
                    task_type=task_kind,
                    question=case.instruction,
                    used_position="planner",
                )
        system_prompt = self.inject_historical_guidelines(SYSTEM_PROMPT, memory_rules)
        traj.write(Role.SYSTEM, system_prompt, step_id=0)
        traj.write(Role.USER, self._build_user_content(case), step_id=0)
        traj.write_event(
            "memory_retrieval",
            {
                "rules": memory_rules,
                "memory_ids": applied_memory_ids,
                "task_kind": task_kind,
                "retrieved": [
                    {
                        "memory_id": mem.get("memory_id"),
                        "memory_type": mem.get("memory_type"),
                        "score": mem.get("_score"),
                        "title": mem.get("title"),
                    }
                    for mem in retrieved_memories
                ],
            },
            step_id=0,
        )

        final_answer = ""
        failure_reason = ""
        steps_done = 0
        api_calls = 0
        total_tokens_accum = 0
        exit_status = "unknown"
        tool_state = self._new_tool_state()

        for step in range(1, max_steps + 1):
            steps_done = step
            force_answer_step = step == max_steps or (
                failure_reason
                and any(
                    marker in failure_reason
                    for marker in (
                        "tool_budget_exhausted_force_answer",
                        "gate_blocked",
                        "max_steps_reached",
                    )
                )
            )
            if force_answer_step:
                self._write_force_answer_prompt(traj, step)
            messages = traj.to_messages(
                recent_steps=self.config.context_recent_steps,
                include_images=(step == 1),
            )
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
            if not self.config.disable_tools and not force_answer_step:
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
            pseudo_tool_calls = []
            if not tool_calls and not self.config.disable_tools:
                pseudo_tool_calls = self._extract_pseudo_tool_calls(content, reasoning_content)
                if pseudo_tool_calls:
                    tool_calls = pseudo_tool_calls

            extra: dict[str, Any] = {}
            if tool_calls:
                raw_tool_calls = list(tool_calls)
                extra["tool_calls"] = [self._dump_tool_call(tc) for tc in raw_tool_calls]
                tool_calls = raw_tool_calls[: max(1, int(self.config.max_tool_calls_per_step))]
                if len(tool_calls) < len(raw_tool_calls):
                    extra["tool_calls_truncated_to"] = len(tool_calls)
            if pseudo_tool_calls:
                extra["pseudo_tool_call_recovered"] = True
            if reasoning_content:
                extra["reasoning_content"] = reasoning_content
            if total_tokens:
                extra["total_tokens"] = total_tokens
            traj.write(Role.ASSISTANT, content, step_id=step, extra=extra or None)

            if not tool_calls:
                if content:
                    if self._extract_pseudo_tool_calls(content, reasoning_content):
                        failure_reason = self._append_reason(
                            failure_reason,
                            "pseudo_tool_call_without_function_call",
                        )
                        continue
                    final_answer = content
                    exit_status = "submitted"
                    break
                continue

            blocked = False
            for tc in tool_calls:
                fn_name, fn_args, tc_id = self._parse_tool_call(tc)
                try:
                    precheck_error = self._precheck_tool_call(fn_name, fn_args, tool_state)
                    if precheck_error:
                        gate.track_error(fn_name, precheck_error)
                        raw_result = self._tool_guidance_result(
                            fn_name, precheck_error, fn_args, tool_state
                        )
                        error = precheck_error
                    else:
                        gate.inspect_call(fn_name, fn_args)
                        raw_result, error = self._dispatch_tool_with_recovery(
                            fn_name, fn_args, case, tool_state
                        )
                        self._record_tool_outcome(fn_name, fn_args, error, tool_state)
                        if error:
                            gate.track_error(fn_name, error)
                    if error and self._should_force_answer_after_tool_error(fn_name, tool_state):
                        raw_result = self._add_force_answer_guidance(raw_result)
                        failure_reason = self._append_reason(
                            failure_reason,
                            "tool_budget_exhausted_force_answer",
                        )
                    tool_result = self.env.serialize_result(raw_result)
                except ToolGateBlocked as exc:
                    failure_reason = f"gate_blocked: {exc}"
                    tool_result = _json_dumps(
                        {
                            "ok": False,
                            "error": failure_reason,
                            "harness_guidance": (
                                "Tool loop or repeated failures were blocked. "
                                "Stop calling tools and return the best concise <answer> from existing observations."
                            ),
                        }
                    )
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
        answer_repair = None

        if self.config.reflection_enabled and self._should_reflection_retry(
            final_answer, failure_reason, exit_status, case.answer
        ):
            compiled_reflection: Reflection | None = None
            compiled_retry_rule = ""
            for retry_no in range(1, self._case_reflection_retry_budget() + 1):
                if compiled_reflection is None:
                    compiled_reflection = self.compiler.compile_failed_trace(
                        traj.read_all(), case.instruction
                    )
                    compiled_retry_rule = compiled_reflection.clin_rule
                reflection = compiled_reflection
                reflection_dict = reflection.as_dict()
                reflection_dict["retry_attempt"] = retry_no
                traj.write_event("reflection", reflection_dict, step_id=steps_done)

                retry_rule = compiled_retry_rule or reflection.clin_rule
                if retry_no == 1 and reflection.memory_worthy:
                    with self._memory_lock:
                        memory_write = self.memory.auto_dream_deduplication(
                            retry_rule,
                            self._memory_keywords(case, reflection),
                            advisor=self.compiler.advise_memory_update
                            if self.config.memory_model_enabled
                            else None,
                            task_instruction=case.instruction,
                            task_type=task_kind,
                            episode_id=task_id,
                        )
                    retry_rule = str(memory_write.get("rule") or retry_rule)
                    traj.write_event("memory_write", memory_write, step_id=steps_done)
                    if memory_write.get("memory_id"):
                        applied_memory_ids.append(str(memory_write["memory_id"]))
                    compiled_retry_rule = str(memory_write.get("rule") or retry_rule)
                    retry_rule = compiled_retry_rule
                if retry_rule:
                    applied_rules.append(retry_rule)

                retry_result = self._run_reflection_retry(
                    case=case,
                    traj=traj,
                    reflection=reflection,
                    retry_rule=retry_rule,
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

        repair_attempts = max(0, int(self.config.answer_repair_attempts))
        if repair_attempts and self._needs_answer_repair(final_answer):
            repair_history: list[dict[str, Any]] = []
            for repair_no in range(1, repair_attempts + 1):
                answer_repair = self._run_short_answer_repair(
                    case=case,
                    traj=traj,
                    previous_answer=final_answer,
                    start_step=steps_done,
                    repair_no=repair_no,
                )
                repair_history.append(answer_repair)
                steps_done = int(answer_repair["steps_done"])
                api_calls += int(answer_repair["api_calls"])
                total_tokens_accum += int(answer_repair["total_tokens"])
                repaired_answer = str(answer_repair.get("answer") or "")
                if repaired_answer:
                    final_answer = repaired_answer
                if self._has_parseable_prediction(final_answer):
                    exit_status = "answer_repaired"
                    break
            if repair_history:
                answer_repair = {
                    "attempts": len(repair_history),
                    "last": repair_history[-1],
                    "history": repair_history,
                }
            if not self._has_parseable_prediction(final_answer):
                failure_reason = self._append_reason(
                    failure_reason,
                    str(
                        (repair_history[-1].get("failure_reason") if repair_history else "")
                        or "answer_repair_unresolved"
                    ),
                )

        forced_answer = None
        if self.config.forced_answer_enabled and self._needs_answer_repair(final_answer):
            forced_answer = self._run_forced_evidence_answer(
                case=case,
                traj=traj,
                previous_answer=final_answer,
                start_step=steps_done,
                failure_reason=failure_reason,
            )
            steps_done = int(forced_answer["steps_done"])
            api_calls += int(forced_answer["api_calls"])
            total_tokens_accum += int(forced_answer["total_tokens"])
            if forced_answer.get("answer"):
                final_answer = str(forced_answer["answer"])
            if self._has_parseable_prediction(final_answer):
                exit_status = "forced_answer_submitted"
                failure_reason = self._append_reason(failure_reason, "forced_answer_used")
            else:
                failure_reason = self._append_reason(
                    failure_reason,
                    str(forced_answer.get("failure_reason") or "forced_answer_unresolved"),
                )

        pred_text = self._extract_model_pred(final_answer)
        has_pred = bool(pred_text)
        eval_status = self._eval_status(final_answer, case.answer)
        success = eval_status == "correct"
        memory_outcome = self._memory_outcome(eval_status, has_pred)
        if applied_memory_ids and memory_outcome:
            with self._memory_lock:
                self.memory.update_after_episode(
                    applied_memory_ids,
                    episode_id=task_id,
                    outcome=memory_outcome,
                    judged_helpful=success,
                    comment="Automatically updated from harness episode outcome.",
                )

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
                with self._memory_lock:
                    memory_write = self.memory.auto_dream_deduplication(
                        reflection.clin_rule,
                        self._memory_keywords(case, reflection),
                        advisor=self.compiler.advise_memory_update
                        if self.config.memory_model_enabled
                        else None,
                        task_instruction=case.instruction,
                        task_type=task_kind,
                        episode_id=task_id,
                    )
                traj.write_event("memory_write", memory_write, step_id=steps_done)

        traj.write_event(
            "run_status",
            {
                "exit_status": exit_status,
                "success": success,
                "has_pred": has_pred,
                "eval_status": eval_status,
                "failure_reason": failure_reason,
                "api_calls": api_calls,
                "total_tokens": total_tokens_accum,
                "answer_repair": answer_repair,
                "forced_answer": forced_answer,
            },
            step_id=steps_done,
        )

        return {
            "task_id": task_id,
            "answer": final_answer,
            "pred": pred_text,
            "steps": steps_done,
            "trajectory_path": str(traj.path),
            "summary": traj.summary(),
            "success": success,
            "has_pred": has_pred,
            "eval_status": eval_status,
            "failure_reason": failure_reason,
            "exit_status": exit_status,
            "api_calls": api_calls,
            "total_tokens": total_tokens_accum,
            "reflection": reflection_dict,
            "memory_write": memory_write,
            "answer_repair": answer_repair,
            "forced_answer": forced_answer,
            "applied_rules": applied_rules,
            "applied_memory_ids": applied_memory_ids,
        }

    def _should_reflection_retry(
        self, final_answer: str, failure_reason: str, exit_status: str, gold_answer: str = ""
    ) -> bool:
        if self._case_reflection_retry_budget() <= 0:
            return False
        if self._is_explanatory_or_long_prediction(final_answer) and not self._is_non_answer_prediction(final_answer):
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
        retry_rule: str,
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
            f"Temporary guideline: {retry_rule or reflection.clin_rule}\n"
            "Use the guideline as a skeptical hint. Avoid repeating blocked or unhelpful tool calls. "
            "If enough evidence is already available, answer immediately. "
            "Return exactly one concise final response wrapped as <answer>...</answer>. "
            "Do not output refusal, uncertainty, tool-failure, or timeout explanations."
        )
        allow_tools = retry_no == 1
        user_step = start_step + 1
        traj.write_event(
            "self_reflection_retry_start",
            {
                "retry_attempt": retry_no,
                "max_steps": max_steps,
                "tools_enabled": allow_tools and not self.config.disable_tools,
                "clin_rule": retry_rule or reflection.clin_rule,
            },
            step_id=start_step,
        )
        traj.write(Role.USER, retry_prompt, step_id=user_step)

        final_answer = ""
        failure_reason = ""
        exit_status = "self_reflection_empty"
        api_calls = 0
        total_tokens_accum = 0
        steps_done = user_step
        tool_state = self._new_tool_state()

        for offset in range(1, max_steps + 1):
            step_id = user_step + offset
            steps_done = step_id
            force_answer_step = offset == max_steps or (
                failure_reason
                and any(
                    marker in failure_reason
                    for marker in (
                        "tool_budget_exhausted_force_answer",
                        "self_reflection_gate_blocked",
                        "self_reflection_tool_failed",
                    )
                )
            )
            if force_answer_step:
                self._write_force_answer_prompt(traj, step_id)
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
                "messages": traj.to_messages(
                    recent_steps=self.config.context_recent_steps,
                    include_images=False,
                ),
                "max_tokens": self.config.max_tokens,
                "temperature": max(0.2, min(self.config.temperature, 0.8)),
                "extra_body": {"enable_thinking": True},
            }
            if not self.config.disable_tools and allow_tools and not force_answer_step:
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
            tool_calls = (
                None
                if self.config.disable_tools or not allow_tools
                else getattr(msg, "tool_calls", None)
            )
            pseudo_tool_calls = []
            if not tool_calls and not self.config.disable_tools and allow_tools:
                pseudo_tool_calls = self._extract_pseudo_tool_calls(content, reasoning_content)
                if pseudo_tool_calls:
                    tool_calls = pseudo_tool_calls

            extra: dict[str, Any] = {}
            if tool_calls:
                raw_tool_calls = list(tool_calls)
                extra["tool_calls"] = [self._dump_tool_call(tc) for tc in raw_tool_calls]
                tool_calls = raw_tool_calls[: max(1, int(self.config.max_tool_calls_per_step))]
                if len(tool_calls) < len(raw_tool_calls):
                    extra["tool_calls_truncated_to"] = len(tool_calls)
            if pseudo_tool_calls:
                extra["pseudo_tool_call_recovered"] = True
            if reasoning_content:
                extra["reasoning_content"] = reasoning_content
            if total_tokens:
                extra["total_tokens"] = total_tokens
            traj.write(Role.ASSISTANT, content, step_id=step_id, extra=extra or None)

            if not tool_calls:
                if content:
                    if self._extract_pseudo_tool_calls(content, reasoning_content):
                        failure_reason = self._append_reason(
                            failure_reason,
                            "self_reflection_pseudo_tool_call_without_function_call",
                        )
                        continue
                    final_answer = content
                    exit_status = "self_reflection_submitted"
                    break
                continue

            blocked = False
            for tc in tool_calls:
                fn_name, fn_args, tc_id = self._parse_tool_call(tc)
                try:
                    precheck_error = self._precheck_tool_call(fn_name, fn_args, tool_state)
                    if precheck_error:
                        gate.track_error(fn_name, precheck_error)
                        raw_result = self._tool_guidance_result(
                            fn_name, precheck_error, fn_args, tool_state
                        )
                        error = precheck_error
                    else:
                        gate.inspect_call(fn_name, fn_args)
                        raw_result, error = self._dispatch_tool_with_recovery(
                            fn_name, fn_args, case, tool_state
                        )
                        self._record_tool_outcome(fn_name, fn_args, error, tool_state)
                        if error:
                            gate.track_error(fn_name, error)
                    if error and self._should_force_answer_after_tool_error(fn_name, tool_state):
                        raw_result = self._add_force_answer_guidance(raw_result)
                    tool_result = self.env.serialize_result(raw_result)
                except ToolGateBlocked as exc:
                    failure_reason = f"self_reflection_gate_blocked: {exc}"
                    tool_result = _json_dumps(
                        {
                            "ok": False,
                            "error": failure_reason,
                            "harness_guidance": (
                                "Tool loop or repeated failures were blocked. "
                                "Stop calling tools and return the best concise <answer> from existing observations."
                            ),
                        }
                    )
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

    def _run_short_answer_repair(
        self,
        case: TaskCase,
        traj: Trajectory,
        previous_answer: str,
        start_step: int,
        repair_no: int = 1,
    ) -> dict[str, Any]:
        """Append one no-tool turn to convert empty/refusal/verbose output into a short answer."""
        user_step = start_step + 1
        assistant_step = user_step + 1
        traj.write_event(
            "answer_repair_start",
            {
                "repair_attempt": repair_no,
                "reason": self._answer_repair_reason(previous_answer),
                "previous_pred": self.extract_pred(previous_answer)[:500],
            },
            step_id=start_step,
        )
        traj.write(
            Role.USER,
            (
                "[HARNESS_SHORT_ANSWER_REPAIR]\n"
                "The previous final answer was empty, a refusal, or too verbose for the benchmark answer column. "
                "Do not call tools. Use only the task, image context, and existing Observations in this trajectory.\n"
                "Before answering, internally decompose the task into target, constraints, intermediate entities, "
                "validation conditions, and output type, then select the candidate satisfying the most constraints. "
                "First choose the single most plausible concrete candidate already implied by the evidence. "
                "If the previous answer mentioned a candidate after phrases like 'most likely' or 'is', keep only that candidate.\n"
                "Return exactly one short span in <answer>...</answer>: a person, title, date, number, organization, place, or object. "
                "Use at most 8 words unless the official name/title is longer. "
                "Do not include explanations, Markdown, 'based on', 'likely', 'unknown', 'unable', 'cannot', 'insufficient', or 'determine'. "
                "If evidence is weak, output the best concrete candidate already mentioned instead of a refusal."
            ),
            step_id=user_step,
        )

        if self.config.mock_llm:
            final_answer = self._mock_answer(case)
            traj.write(
                Role.ASSISTANT,
                final_answer,
                step_id=assistant_step,
                extra={"reasoning_content": "MOCK_LLM short-answer repair path."},
            )
            status = "answer_repaired" if self._has_parseable_prediction(final_answer) else "answer_repair_unresolved"
            traj.write_event(
                "answer_repair_status",
                {
                    "repair_attempt": repair_no,
                    "exit_status": status,
                    "pred": self._extract_model_pred(final_answer),
                    "api_calls": 0,
                    "total_tokens": 0,
                },
                step_id=assistant_step,
            )
            return {
                "answer": final_answer,
                "steps_done": assistant_step,
                "failure_reason": "" if status == "answer_repaired" else "answer_repair_unresolved",
                "api_calls": 0,
                "total_tokens": 0,
                "repair_attempt": repair_no,
            }

        request_kwargs = {
            "model": self.config.model_name,
            "messages": traj.to_messages(
                recent_steps=max(self.config.context_recent_steps, 8),
                include_images=False,
            ),
            "max_tokens": min(self.config.max_tokens, 512),
            "temperature": 0.2,
            "extra_body": {"enable_thinking": False},
        }
        api_calls = 0
        total_tokens_accum = 0
        final_answer = ""
        failure_reason = ""
        try:
            if self.client is None:
                raise RuntimeError("OpenAI client is unavailable")
            response, attempts = self._call_llm_with_retry(request_kwargs)
            api_calls += attempts
            msg = response.choices[0].message
            final_answer = getattr(msg, "content", None) or ""
            reasoning_content = getattr(msg, "reasoning_content", None) or ""
            usage = getattr(response, "usage", None)
            total_tokens = getattr(usage, "total_tokens", None) if usage else None
            if total_tokens:
                total_tokens_accum += int(total_tokens)
            extra: dict[str, Any] = {}
            if reasoning_content:
                extra["reasoning_content"] = reasoning_content
            if total_tokens:
                extra["total_tokens"] = total_tokens
            traj.write(Role.ASSISTANT, final_answer, step_id=assistant_step, extra=extra or None)
        except LLMCallFailure as exc:
            api_calls += exc.attempts
            failure_reason = f"answer_repair_llm_failed: {type(exc).__name__}: {exc}"
            traj.write(Role.TOOL, f"[HARNESS ERROR] {failure_reason}", step_id=assistant_step)
        except Exception as exc:
            failure_reason = f"answer_repair_llm_failed: {type(exc).__name__}: {exc}"
            traj.write(Role.TOOL, f"[HARNESS ERROR] {failure_reason}", step_id=assistant_step)

        if final_answer and self._needs_answer_repair(final_answer):
            failure_reason = self._append_reason(failure_reason, "answer_repair_unresolved")
        status = "answer_repaired" if final_answer and not self._needs_answer_repair(final_answer) else "answer_repair_unresolved"
        traj.write_event(
            "answer_repair_status",
            {
                "repair_attempt": repair_no,
                "exit_status": status,
                "pred": self._extract_model_pred(final_answer),
                "failure_reason": failure_reason,
                "api_calls": api_calls,
                "total_tokens": total_tokens_accum,
            },
            step_id=assistant_step,
        )
        return {
            "answer": final_answer,
            "steps_done": assistant_step,
            "failure_reason": failure_reason,
            "api_calls": api_calls,
            "total_tokens": total_tokens_accum,
            "repair_attempt": repair_no,
        }

    def _run_forced_evidence_answer(
        self,
        case: TaskCase,
        traj: Trajectory,
        previous_answer: str,
        start_step: int,
        failure_reason: str,
    ) -> dict[str, Any]:
        """Last-resort no-tool answer extraction from task text, search observations, and prior reasoning."""
        user_step = start_step + 1
        assistant_step = user_step + 1
        evidence = self._build_forced_answer_evidence(traj)
        traj.write_event(
            "forced_answer_start",
            {
                "reason": self._answer_repair_reason(previous_answer),
                "previous_pred": self.extract_pred(previous_answer)[:500],
                "failure_reason": failure_reason[:500],
                "evidence_chars": len(evidence),
            },
            step_id=start_step,
        )
        traj.write(
            Role.USER,
            (
                "[HARNESS_FORCED_EVIDENCE_ANSWER]\n"
                "All normal answer attempts failed or produced an invalid benchmark answer. "
                "You must now give one concrete best-effort answer. Do not call tools.\n"
                "Use the task, prior reasoning, search/browser Observations, recovered candidates, and memory/reflection hints below. "
                "Prefer a named entity, title, date, number, organization, place, team, person, or object that appears in the evidence. "
                "If evidence is weak, choose the least-wrong concrete candidate already mentioned; never return empty or refusal text.\n"
                "Output exactly one XML answer tag and nothing else: <answer>short concrete answer</answer>. "
                "Do not include unable, cannot, unknown, insufficient, not enough, determine, placeholder brackets, Markdown, explanation, or uncertainty words.\n\n"
                f"Task:\n{case.instruction}\n\n"
                f"Failure reason:\n{failure_reason or 'none'}\n\n"
                f"Evidence and candidates:\n{evidence}"
            ),
            step_id=user_step,
        )

        if self.config.mock_llm:
            final_answer = self._mock_answer(case)
            status = "forced_answer_submitted" if self._has_parseable_prediction(final_answer) else "forced_answer_unresolved"
            traj.write(
                Role.ASSISTANT,
                final_answer,
                step_id=assistant_step,
                extra={"reasoning_content": "MOCK_LLM forced-answer path."},
            )
            traj.write_event(
                "forced_answer_status",
                {"exit_status": status, "pred": self._extract_model_pred(final_answer), "api_calls": 0, "total_tokens": 0},
                step_id=assistant_step,
            )
            return {
                "answer": final_answer,
                "steps_done": assistant_step,
                "failure_reason": "" if status == "forced_answer_submitted" else "forced_answer_unresolved",
                "api_calls": 0,
                "total_tokens": 0,
            }

        request_kwargs = {
            "model": self.config.model_name,
            "messages": traj.to_messages(
                recent_steps=max(self.config.context_recent_steps, 10),
                include_images=False,
            ),
            "max_tokens": min(self.config.max_tokens, 512),
            "temperature": 0.0,
            "extra_body": {"enable_thinking": False},
        }
        api_calls = 0
        total_tokens_accum = 0
        final_answer = ""
        forced_failure = ""
        try:
            if self.client is None:
                raise RuntimeError("OpenAI client is unavailable")
            response, attempts = self._call_llm_with_retry(request_kwargs)
            api_calls += attempts
            msg = response.choices[0].message
            final_answer = getattr(msg, "content", None) or ""
            reasoning_content = getattr(msg, "reasoning_content", None) or ""
            usage = getattr(response, "usage", None)
            total_tokens = getattr(usage, "total_tokens", None) if usage else None
            if total_tokens:
                total_tokens_accum += int(total_tokens)
            extra: dict[str, Any] = {}
            if reasoning_content:
                extra["reasoning_content"] = reasoning_content
            if total_tokens:
                extra["total_tokens"] = total_tokens
            traj.write(Role.ASSISTANT, final_answer, step_id=assistant_step, extra=extra or None)
        except LLMCallFailure as exc:
            api_calls += exc.attempts
            forced_failure = f"forced_answer_llm_failed: {type(exc).__name__}: {exc}"
            traj.write(Role.TOOL, f"[HARNESS ERROR] {forced_failure}", step_id=assistant_step)
        except Exception as exc:
            forced_failure = f"forced_answer_llm_failed: {type(exc).__name__}: {exc}"
            traj.write(Role.TOOL, f"[HARNESS ERROR] {forced_failure}", step_id=assistant_step)

        if not self._has_parseable_prediction(final_answer):
            heuristic_answer = self._heuristic_forced_answer(evidence, case.instruction)
            if heuristic_answer:
                final_answer = f"<answer>{heuristic_answer}</answer>"
                traj.write_event(
                    "forced_answer_heuristic",
                    {"pred": self._extract_model_pred(final_answer), "source": "trajectory_candidate_extraction"},
                    step_id=assistant_step,
                )

        status = "forced_answer_submitted" if self._has_parseable_prediction(final_answer) else "forced_answer_unresolved"
        if status != "forced_answer_submitted":
            forced_failure = self._append_reason(forced_failure, "forced_answer_unresolved")
        traj.write_event(
            "forced_answer_status",
            {
                "exit_status": status,
                "pred": self._extract_model_pred(final_answer),
                "failure_reason": forced_failure,
                "api_calls": api_calls,
                "total_tokens": total_tokens_accum,
            },
            step_id=assistant_step,
        )
        return {
            "answer": final_answer,
            "steps_done": assistant_step,
            "failure_reason": forced_failure,
            "api_calls": api_calls,
            "total_tokens": total_tokens_accum,
        }

    def _build_forced_answer_evidence(self, traj: Trajectory) -> str:
        max_chars = max(1000, int(self.config.forced_answer_evidence_chars))
        snippets: list[str] = []
        rows = traj.read_all()
        for row in rows:
            role = row.get("role")
            content = row.get("content")
            if row.get("event_type") in {"reflection", "memory_retrieval"}:
                if self.config.forced_answer_use_reflection:
                    snippets.append(f"[{row.get('event_type')}] {self._compact_text(content, 700)}")
            elif role == "tool":
                snippets.append(f"[tool:{row.get('fn_name') or 'unknown'}] {self._compact_text(content, 1400)}")
            elif role == "assistant":
                answer_text = self.extract_pred(str(content or ""))
                reasoning = str(row.get("reasoning_content") or "")
                if answer_text and not self._needs_answer_repair(f"<answer>{answer_text}</answer>"):
                    snippets.append(f"[assistant_answer_candidate] {self._compact_text(answer_text, 300)}")
                if reasoning:
                    snippets.append(f"[assistant_reasoning] {self._compact_text(reasoning, 1100)}")
        joined = "\n".join(snippets)
        if len(joined) <= max_chars:
            return joined
        return joined[-max_chars:]

    def _compact_text(self, value: Any, max_chars: int) -> str:
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False)
        value = re.sub(r"\s+", " ", value).strip()
        return value[:max_chars]

    def _heuristic_forced_answer(self, evidence: str, instruction: str) -> str:
        candidates: list[str] = []
        for match in re.finditer(r"<answer>(.*?)</answer>", evidence, flags=re.S | re.I):
            candidates.append(match.group(1))
        for line in evidence.splitlines():
            if "[assistant_answer_candidate]" in line:
                candidates.append(line.split("]", 1)[-1])
        for prefix in (
            r"(?:answer is|answer:|candidate is|likely answer is|provide:)\s*([A-Z][A-Za-z0-9&'().,\- ]{2,80})",
            r"(?:答案是|候选答案是|最可能是)\s*([\u4e00-\u9fffA-Za-z0-9&'().,\- ]{2,80})",
        ):
            for match in re.finditer(prefix, evidence, flags=re.I):
                candidates.append(match.group(1))
        for quoted in re.findall(r"[\"“]([^\"”]{3,80})[\"”]", evidence):
            candidates.append(quoted)
        title_like = re.findall(
            r"\b([A-Z][A-Za-z0-9'&.-]+(?:\s+[A-Z][A-Za-z0-9'&.-]+){1,8})\b",
            evidence,
        )
        candidates.extend(title_like)

        task_terms = set(re.findall(r"[a-z0-9]+", instruction.lower()))
        best = ""
        best_score = -1
        for candidate in candidates:
            cleaned = self._clean_forced_candidate(candidate)
            if not cleaned:
                continue
            if self._needs_answer_repair(f"<answer>{cleaned}</answer>"):
                continue
            words = re.findall(r"[a-z0-9]+", cleaned.lower())
            generic_penalty = sum(1 for word in words if word in task_terms)
            score = len(cleaned) - generic_penalty * 8
            if re.search(r"\d", cleaned):
                score += 5
            if 2 <= len(words) <= 8:
                score += 5
            if score > best_score:
                best = cleaned
                best_score = score
        return best

    def _clean_forced_candidate(self, candidate: str) -> str:
        candidate = re.sub(r"<.*?>", " ", str(candidate or ""))
        candidate = re.sub(r"\s+", " ", candidate).strip(" \t\r\n\"'`.,;:!?")
        candidate = re.sub(r"^\s*(the answer is|answer is|answer:|candidate is)\s+", "", candidate, flags=re.I)
        candidate = candidate.strip(" \t\r\n\"'`.,;:!?")
        if not candidate or len(candidate) > 120:
            return ""
        generic_phrases = (
            "after evidence is sufficient",
            "emit exactly one short answer",
            "wrap the answer",
            "strict answer tag",
            "answer repaired",
            "forced answer",
            "tool budget",
            "search budget",
            "not enough information",
            "unable to",
            "cannot",
            "insufficient",
            "unknown",
            "determine",
            "the run ended",
            "the agent repeated",
            "action:",
            "confidence:",
            "correct_strategy",
            "failure_type",
            "root_cause",
            "memory_worthy",
            "clin_rule",
            "missing_strict_answer",
            "incomplete_reasoning_chain",
            "retry_attempt",
            "memory_retrieval",
            "reflection",
            "episode",
            "task_kind",
            "open_search",
            "visual",
            "perform structured",
            "multi-hop validation",
            "subject validation",
            "candidate validation",
            "validation checks",
            "least-wrong concrete candidate",
            "rules",
            "memory_ids",
            "retrieved",
        )
        low = candidate.lower()
        if any(phrase in low for phrase in generic_phrases):
            return ""
        if re.fullmatch(r"[a-z_]{3,40}", candidate) and "_" in candidate:
            return ""
        if re.fullmatch(
            r"(confidence|should|may|true|false|success|failure|empty|submitted|none|null|open_search|retrieved)",
            low,
        ):
            return ""
        if re.search(r"\[(?:team name|answer|unknown|placeholder)\]", candidate, flags=re.I):
            return ""
        return candidate

    def _build_user_content(self, case: TaskCase) -> Any:
        instruction = case.instruction
        if case.image_url:
            instruction = f"{instruction}\nimage_url: {case.image_url}"
        if case.image:
            instruction = (
                f"{instruction}\nlocal_image_path_for_vision: {case.image}"
                f"\nlocal_image_path_for_search: {case.image}"
            )

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

    def _extract_pseudo_tool_calls(
        self, content: str, reasoning_content: str = ""
    ) -> list[dict[str, Any]]:
        """Recover XML-like tool calls that some models emit as plain text."""
        text = "\n".join(part for part in (content, reasoning_content) if part)
        if "<tool_call" not in text and "<function=" not in text:
            return []
        calls: list[dict[str, Any]] = []
        blocks = re.findall(r"<tool_call[^>]*>(.*?)</tool_call>", text, flags=re.S | re.I)
        if not blocks:
            blocks = [text]
        for block in blocks:
            match = re.search(r"<function=([A-Za-z_][A-Za-z0-9_]*)[^>]*>", block, flags=re.I)
            if not match:
                continue
            name = match.group(1)
            args: dict[str, Any] = {}
            for key, raw_value in re.findall(
                r"<parameter=([A-Za-z_][A-Za-z0-9_]*)[^>]*>(.*?)</parameter>",
                block,
                flags=re.S | re.I,
            ):
                args[key] = self._coerce_pseudo_arg(raw_value)
            if not args:
                json_match = re.search(r"\{.*\}", block, flags=re.S)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(0))
                        if isinstance(parsed, dict):
                            args = parsed
                    except json.JSONDecodeError:
                        args = {}
            calls.append(
                {
                    "id": f"pseudo_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }
            )
        return calls

    def _coerce_pseudo_arg(self, value: str) -> Any:
        value = re.sub(r"\s+", " ", (value or "").strip())
        lower = value.lower()
        if lower in {"true", "false"}:
            return lower == "true"
        if re.fullmatch(r"-?\d+", value):
            try:
                return int(value)
            except ValueError:
                return value
        if re.fullmatch(r"-?\d+\.\d+", value):
            try:
                return float(value)
            except ValueError:
                return value
        return value

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

    def _new_tool_state(self) -> dict[str, Any]:
        return {
            "search_calls": 0,
            "search_keys": {},
            "search_seen_refs": {},
            "search_seen_domains": {},
            "last_search_domains": [],
            "stale_searches": 0,
            "browser_failed_urls": {},
            "browser_last_errors": {},
        }

    def _tool_call_key(self, fn_name: str, fn_args: dict[str, Any]) -> str:
        if fn_name == "search_text":
            query = re.sub(r"\s+", " ", str(fn_args.get("query") or "").strip().lower())
            return f"search_text:{query}"
        if fn_name == "search_image":
            image = str(fn_args.get("image_url") or fn_args.get("image") or "").strip()
            return f"search_image:{_canonical_url(image) if _is_http_url(image) else image}"
        try:
            return f"{fn_name}:{json.dumps(fn_args, ensure_ascii=False, sort_keys=True)}"
        except TypeError:
            return f"{fn_name}:{fn_args}"

    def _precheck_tool_call(
        self, fn_name: str, fn_args: dict[str, Any], tool_state: dict[str, Any]
    ) -> str:
        if fn_name.startswith("search_"):
            search_calls = int(tool_state.get("search_calls", 0))
            if search_calls >= max(1, self.config.max_search_calls_per_case):
                return (
                    f"search_budget_exhausted: already used {search_calls} search calls in this case. "
                    "Stop searching and return the best concise <answer> from existing observations."
                )
            if int(tool_state.get("stale_searches", 0)) >= max(1, self.config.search_stale_result_limit):
                return (
                    "stale_search_loop_blocked: recent searches returned no new references. "
                    "Change strategy substantially or return the best concise <answer> from existing observations."
                )
            key = self._tool_call_key(fn_name, fn_args)
            if int(tool_state.get("search_keys", {}).get(key, 0)) >= 1:
                return (
                    "repeated_search_blocked: this search query/image was already attempted. "
                    "Use existing observations, change strategy, or answer now."
                )
            avoid_clause = self._search_avoidance_hint(fn_name, tool_state)
            if avoid_clause and fn_name == "search_text":
                query = str(fn_args.get("query") or "").lower()
                if "-site:" not in query and "site:" not in query:
                    return (
                        f"stale_domain_guard: recent searches already used {avoid_clause}. "
                        "Revise the query with -site: exclusions, a quoted rare phrase, a different entity, "
                        "or answer from current evidence."
                    )

        if fn_name == "browser_navigate":
            canonical = _canonical_url(str(fn_args.get("url") or ""))
            if canonical and int(tool_state.get("browser_failed_urls", {}).get(canonical, 0)) >= max(
                1, self.config.browser_url_failure_limit
            ):
                last_error = tool_state.get("browser_last_errors", {}).get(canonical, "previous browser failure")
                return (
                    f"browser_url_blocked: same URL already failed via browser ({last_error[:220]}). "
                    "Do not retry this URL; use search_text, another source URL, or answer from existing observations."
                )
        return ""


    def _search_avoidance_hint(self, fn_name: str, tool_state: dict[str, Any]) -> str:
        if fn_name != "search_text":
            return ""
        domains = [domain for domain in tool_state.get("last_search_domains", []) if domain]
        if len(domains) < 2:
            return ""
        unique_domains = []
        for domain in domains:
            if domain not in unique_domains:
                unique_domains.append(domain)
        if len(unique_domains) >= 3:
            return ", ".join(f"-site:{domain}" for domain in unique_domains[:3])
        return ""

    def _record_tool_outcome(
        self,
        fn_name: str,
        fn_args: dict[str, Any],
        error: str,
        tool_state: dict[str, Any],
    ) -> None:
        if fn_name.startswith("search_"):
            tool_state["search_calls"] = int(tool_state.get("search_calls", 0)) + 1
            key = self._tool_call_key(fn_name, fn_args)
            search_keys = tool_state.setdefault("search_keys", {})
            search_keys[key] = int(search_keys.get(key, 0)) + 1
            if fn_name == "search_text" and not error:
                domains = list(tool_state.get("last_result_domains", []))
                if domains:
                    tool_state["last_search_domains"] = domains

        if fn_name == "browser_navigate" and error and self._is_browser_url_failure(error):
            canonical = _canonical_url(str(fn_args.get("url") or ""))
            if canonical:
                failed_urls = tool_state.setdefault("browser_failed_urls", {})
                failed_urls[canonical] = int(failed_urls.get(canonical, 0)) + 1
                tool_state.setdefault("browser_last_errors", {})[canonical] = error
        elif fn_name == "browser_parallel" and error and self._is_browser_url_failure(error):
            failed_urls = tool_state.setdefault("browser_failed_urls", {})
            last_errors = tool_state.setdefault("browser_last_errors", {})
            for url in fn_args.get("urls") or []:
                canonical = _canonical_url(str(url))
                if canonical:
                    failed_urls[canonical] = int(failed_urls.get(canonical, 0)) + 1
                    last_errors[canonical] = error

    def _is_browser_url_failure(self, error: str) -> bool:
        text = (error or "").lower()
        return any(
            marker in text
            for marker in (
                "http 500",
                "500",
                "502",
                "503",
                "504",
                "401",
                "403",
                "timeout",
                "timed out",
                "session",
                "connection",
                "navigate failed",
                "proxy-error",
            )
        )

    def _tool_guidance_result(
        self,
        fn_name: str,
        error: str,
        fn_args: dict[str, Any],
        tool_state: dict[str, Any],
    ) -> dict[str, Any]:
        guidance = "Stop this tool path and answer from existing evidence if possible."
        if fn_name.startswith("browser_"):
            guidance = (
                "Browser failed for this URL before. Do not retry the same URL; "
                "use search_text with title/domain keywords, try a different source, or answer from observations."
            )
        elif fn_name.startswith("search_"):
            guidance = (
                "Search budget or duplicate-search guard fired. Stop searching; "
                "use gathered snippets/visual evidence and return the most likely short answer."
            )
        return {
            "ok": False,
            "error": error,
            "fn_name": fn_name,
            "fn_args": fn_args,
            "tool_state": {
                "search_calls": tool_state.get("search_calls", 0),
                "stale_searches": tool_state.get("stale_searches", 0),
                "browser_failed_urls": tool_state.get("browser_failed_urls", {}),
            },
            "harness_guidance": guidance,
        }

    def _should_force_answer_after_tool_error(
        self, fn_name: str, tool_state: dict[str, Any]
    ) -> bool:
        return fn_name.startswith("search_") and int(tool_state.get("search_calls", 0)) >= max(
            1, self.config.max_search_calls_per_case
        )

    def _add_force_answer_guidance(self, raw_result: Any) -> dict[str, Any]:
        return {
            "ok": False,
            "error": self.env.result_error(raw_result) or "tool_budget_exhausted",
            "last_result": raw_result,
            "harness_guidance": (
                "Search/tool budget is exhausted. Do not call more tools; "
                "return the best concise <answer> using current observations."
            ),
        }

    def _dispatch_tool_with_recovery(
        self,
        fn_name: str,
        fn_args: dict[str, Any],
        case: TaskCase,
        tool_state: dict[str, Any] | None = None,
    ) -> tuple[Any, str]:
        if fn_name == "search_image":
            fn_args = self._normalize_search_image_args(fn_args, case)
        raw_result = self.env.dispatch(fn_name, fn_args)
        error = self.env.result_error(raw_result)
        if fn_name.startswith("search_") and not error and tool_state is not None:
            raw_result, error = self._enforce_search_novelty(raw_result, tool_state)
        if (
            fn_name == "search_image"
            and error
            and (case.image_url or case.image)
        ):
            recovery_attempts = self._search_image_recovery_args(fn_args, case)
            recovery_results = []
            for retry_args in recovery_attempts:
                retry_result = self.env.dispatch(fn_name, retry_args)
                retry_error = self.env.result_error(retry_result)
                recovery_results.append(
                    {
                        "args": retry_args,
                        "error": retry_error,
                        "result": retry_result,
                    }
                )
                if not retry_error:
                    if tool_state is not None:
                        filtered_result, novelty_error = self._enforce_search_novelty(
                            retry_result, tool_state
                        )
                        if novelty_error:
                            retry_error = novelty_error
                            retry_result = filtered_result
                            recovery_results[-1]["error"] = retry_error
                            recovery_results[-1]["result"] = retry_result
                            continue
                        retry_result = filtered_result
                        recovery_results[-1]["result"] = retry_result
                    return {
                        "ok": True,
                        "recovered_by": "search_image_available_image",
                        "original_args": fn_args,
                        "original_error": error,
                        "recovery_results": recovery_results,
                    }, ""
            if recovery_results:
                return {
                    "ok": False,
                    "recovered_by": "search_image_available_image",
                    "original_args": fn_args,
                    "original_error": error,
                    "recovery_results": recovery_results,
                }, str(recovery_results[-1].get("error") or error)
        return raw_result, error

    def _enforce_search_novelty(
        self, raw_result: Any, tool_state: dict[str, Any]
    ) -> tuple[Any, str]:
        entries = self._search_result_entries(raw_result)
        if not entries:
            return raw_result, ""

        seen_refs = tool_state.setdefault("search_seen_refs", {})
        seen_domains = tool_state.setdefault("search_seen_domains", {})
        new_entries = []
        duplicate_count = 0
        result_domains: list[str] = []
        for entry in entries:
            ref = self._search_result_ref(entry)
            if ref and ref in seen_refs:
                duplicate_count += 1
                continue
            new_entries.append(entry)
            if ref:
                seen_refs[ref] = int(seen_refs.get(ref, 0)) + 1
            domain = self._search_result_domain(entry)
            if domain:
                result_domains.append(domain)
                seen_domains[domain] = int(seen_domains.get(domain, 0)) + 1

        if result_domains:
            tool_state["last_result_domains"] = result_domains

        if not new_entries:
            tool_state["stale_searches"] = int(tool_state.get("stale_searches", 0)) + 1
            return {
                "ok": False,
                "error": "stale_search_results: all returned URLs/titles were already seen in this case",
                "duplicate_results": duplicate_count,
                "seen_reference_count": len(seen_refs),
                "harness_guidance": (
                    "Do not repeat equivalent search terms. Use missing attributes, exclude seen domains, "
                    "try a different source/language, or stop searching and answer from existing evidence."
                ),
            }, "stale_search_results"

        tool_state["stale_searches"] = 0
        filtered = self._replace_search_result_entries(raw_result, new_entries)
        if isinstance(filtered, dict):
            filtered.setdefault("harness_novelty", {})
            filtered["harness_novelty"].update(
                {
                    "new_results": len(new_entries),
                    "duplicate_results_filtered": duplicate_count,
                    "seen_reference_count": len(seen_refs),
                }
            )
        return filtered, ""

    def _search_result_entries(self, raw_result: Any) -> list[dict[str, Any]]:
        if isinstance(raw_result, list):
            return [item for item in raw_result if isinstance(item, dict)]
        if isinstance(raw_result, dict) and isinstance(raw_result.get("results"), list):
            return [item for item in raw_result["results"] if isinstance(item, dict)]
        return []

    def _replace_search_result_entries(
        self, raw_result: Any, entries: list[dict[str, Any]]
    ) -> Any:
        if isinstance(raw_result, list):
            return entries
        if isinstance(raw_result, dict) and isinstance(raw_result.get("results"), list):
            return {**raw_result, "results": entries}
        return raw_result

    def _search_result_ref(self, entry: dict[str, Any]) -> str:
        url = str(entry.get("url") or entry.get("link") or "").strip()
        if url:
            return f"url:{_canonical_url(url) if _is_http_url(url) else url.lower()}"
        title = re.sub(r"\s+", " ", str(entry.get("title") or "").strip().lower())
        snippet = re.sub(r"\s+", " ", str(entry.get("snippet") or "").strip().lower())
        if title:
            return f"title:{title}"
        if snippet:
            return f"snippet:{snippet[:160]}"
        return ""

    def _search_result_domain(self, entry: dict[str, Any]) -> str:
        url = str(entry.get("url") or entry.get("link") or "").strip()
        return _url_domain(url) if url else ""

    def _normalize_search_image_args(
        self, fn_args: dict[str, Any], case: TaskCase
    ) -> dict[str, Any]:
        normalized = dict(fn_args)
        image_url = str(normalized.get("image_url") or "").strip()
        image = str(normalized.get("image") or "").strip()
        if image_url and not _is_http_url(image_url):
            normalized.pop("image_url", None)
            normalized.setdefault("image", image_url)
            image_url = ""
        if case.image_url and (not image_url or image_url != case.image_url):
            normalized["image_url"] = case.image_url
            normalized.pop("image", None)
            return normalized
        if not image_url and not image and case.image:
            normalized["image"] = case.image
        return normalized

    def _search_image_recovery_args(
        self, fn_args: dict[str, Any], case: TaskCase
    ) -> list[dict[str, Any]]:
        base_args = {
            key: value
            for key, value in fn_args.items()
            if key in {"top_k", "fetch", "max_chars"}
        }
        attempts: list[dict[str, Any]] = []
        if case.image_url and fn_args.get("image_url") != case.image_url:
            attempts.append({**base_args, "image_url": case.image_url})
        if case.image and fn_args.get("image") != case.image:
            attempts.append({**base_args, "image": case.image})
        if case.image_url and case.image and fn_args.get("image") != case.image:
            attempts.append({**base_args, "image": case.image})
        return attempts

    def _judge_success(self, pred: str, answer: str) -> bool:
        if not answer:
            return bool(pred.strip())
        p = _normalize_answer(pred)
        a = _normalize_answer(answer)
        return bool(a and (p == a or a in p))

    def _eval_status(self, pred: str, answer: str) -> str:
        if not self._has_parseable_prediction(pred):
            return "no_prediction"
        if not (answer or "").strip():
            return "unknown"
        return "correct" if self._judge_success(pred, answer) else "incorrect"

    def _memory_outcome(self, eval_status: str, has_pred: bool) -> str | None:
        if eval_status == "correct":
            return "success"
        if eval_status in {"incorrect", "no_prediction"}:
            return "failure"
        if not has_pred:
            return "failure"
        return None

    def extract_pred(self, answer: str) -> str:
        return _normalize_answer(answer) or answer.strip()

    def _extract_model_pred(self, answer: str) -> str:
        return self.extract_pred(answer) if self._has_parseable_prediction(answer) else ""

    def _has_parseable_prediction(self, answer: str) -> bool:
        stripped = (answer or "").strip()
        if self._extract_pseudo_tool_calls(stripped):
            return False
        if self._is_non_answer_prediction(answer):
            return False
        if self._is_explanatory_or_long_prediction(answer):
            return False
        return bool(self.extract_pred(answer)) and not stripped.upper().startswith("[HARNESS]")

    def _needs_answer_repair(self, answer: str) -> bool:
        if not (answer or "").strip():
            return True
        if not self._has_parseable_prediction(answer):
            return True
        return self._is_explanatory_or_long_prediction(answer)

    def _answer_repair_reason(self, answer: str) -> str:
        if not (answer or "").strip():
            return "empty_answer"
        if self._is_non_answer_prediction(answer):
            return "non_answer_refusal"
        if self._is_explanatory_or_long_prediction(answer):
            return "verbose_or_explanatory_answer"
        return "unknown"

    def _is_non_answer_prediction(self, answer: str) -> bool:
        pred = self.extract_pred(answer).lower()
        if not pred:
            return False
        non_answer_patterns = (
            r"^\s*https?://",
            r"^\s*www\.",
            r"\bunable\s+to\b",
            r"\bunable\b.*\b(identify|determine|verify|find|answer)\b",
            r"\bcannot\b.*\b(identify|determine|verify|find|answer)\b",
            r"\bcan\s+not\b.*\b(identify|determine|verify|find|answer)\b",
            r"\bcan't\b.*\b(identify|determine|verify|find|answer)\b",
            r"\bnot\s+able\s+to\b",
            r"\bno\s+definitive\b",
            r"\bnot\s+possible\s+to\b",
            r"\bwith\s+certainty\b",
            r"^\s*unknown\b",
            r"\binformation\s+insufficient\b",
            r"\binsufficient\s+(evidence|search|results|verification|information)\b",
            r"\bnot\s+provided\b",
            r"\bnot\s+explicitly\s+stated\b",
            r"\bnot\s+enough\s+(evidence|context|search results)\b",
        )
        if any(re.search(pattern, pred) for pattern in non_answer_patterns):
            return True
        non_answer_markers = (
            "unable to answer",
            "cannot answer",
            "can't answer",
            "can not answer",
            "cannot determine",
            "can't determine",
            "not enough information",
            "insufficient information",
            "insufficient evidence",
            "information insufficient",
            "available information",
            "current information",
            "available search results",
            "current search results",
            "given constraints",
            "no definitive",
            "definitive match",
            "definitively identify",
            "search timed out",
            "tool timed out",
            "tool failed",
            "tools failed",
            "timeout",
            "无法回答",
            "不能回答",
            "无法确定",
            "无法判断",
            "无法验证",
            "无法得出",
            "信息不足",
            "证据不足",
            "搜索超时",
            "工具失败",
            "工具超时",
            "无法访问",
        )
        return any(marker in pred for marker in non_answer_markers)

    def _is_explanatory_or_long_prediction(self, answer: str) -> bool:
        pred = self.extract_pred(answer).strip()
        if not pred:
            return False
        low = pred.lower()
        words = re.findall(r"[\w'-]+", pred, flags=re.U)
        if len(pred) > 140 or len(words) > 14:
            return True
        explanatory_markers = (
            "based on the search results",
            "based on the clues",
            "based on available",
            "the answer is likely",
            "the most likely",
            "most likely **",
            "though definitive",
            "requires additional verification",
            "without more detailed",
            "without definitive",
            "i cannot",
            "i can't",
            "i am unable",
            "not provided",
            "not explicitly stated",
            "information insufficient",
            "available information",
            "current information",
            "no definitive",
            "definitive match",
            "definitively identify",
        )
        return any(marker in low for marker in explanatory_markers)

    def _write_force_answer_prompt(self, traj: Trajectory, step_id: int) -> None:
        marker = f"force_answer_prompt:{step_id}"
        for row in traj.read_all():
            if row.get("event_type") == marker:
                return
        traj.write_event(
            marker,
            {"reason": "last_or_recovery_step_forces_final_answer"},
            step_id=step_id,
        )
        traj.write(
            Role.USER,
            (
                "[HARNESS_FORCE_ANSWER]\n"
                "This is the final answer turn for this case. Do not call tools. "
                "Use the strongest existing Observation, task text, visual evidence, and common knowledge. "
                "Return exactly one concise best-effort answer in <answer>...</answer>. "
                "Use at most 8 words unless the official name/title is longer. "
                "Never say unable/cannot/无法/信息不足/search timeout/tool failed/unknown/insufficient/determine. "
                "Never include explanations, Markdown, 'based on', 'likely', 'unknown', or uncertainty wording. "
                "If evidence is weak, output the best concrete entity already mentioned instead of a refusal."
            ),
            step_id=step_id,
        )

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
        return sorted(set(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", text.lower())))[:80]

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
        workers: int = 1,
    ) -> list[dict[str, Any]]:
        results = []
        output_path = output_path or str(Path(self.config.result_dir) / "predictions.jsonl")
        continue_on_error = self.config.batch_continue_on_error if continue_on_error is None else continue_on_error
        completed_indices = self._read_completed_indices(
            output_path, valid_only=self.config.resume_valid_only
        ) if resume else set()
        status_counts: dict[str, int] = {}
        if Path(output_path).exists() and not resume:
            Path(output_path).unlink()
        cases: list[TaskCase] = []
        for case in self.load_next_case():
            if case.index < start:
                continue
            if case.index in completed_indices:
                logger.info("case %s skipped because output already exists", case.task_id)
                continue
            if limit is not None and len(cases) >= limit:
                break
            cases.append(case)

        def process_case(case: TaskCase) -> dict[str, Any]:
            try:
                return self.run_case(case, trajectory_dir=trajectory_dir)
            except Exception as exc:
                if not continue_on_error:
                    raise
                logger.error("case %s failed with uncaught exception: %s", case.task_id, exc, exc_info=True)
                return {
                    "task_id": case.task_id or f"case_{case.index}",
                    "answer": "",
                    "pred": "",
                    "steps": 0,
                    "trajectory_path": "",
                    "summary": {},
                    "success": False,
                    "has_pred": False,
                    "eval_status": "no_prediction",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "exit_status": f"uncaught_{type(exc).__name__}",
                    "api_calls": 0,
                    "total_tokens": 0,
                    "reflection": None,
                    "memory_write": None,
                    "applied_rules": [],
                    "traceback": traceback.format_exc(),
                }

        def record_result(case: TaskCase, result: dict[str, Any]) -> None:
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

        workers = max(1, int(workers or 1))
        if workers == 1:
            for case in cases:
                record_result(case, process_case(case))
            return results

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_case = {executor.submit(process_case, case): case for case in cases}
            for future in as_completed(future_to_case):
                case = future_to_case[future]
                record_result(case, future.result())
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

    def _read_completed_indices(self, output_path: str, valid_only: bool = False) -> set[int]:
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
                if valid_only and not self._has_parseable_prediction(str(row.get("pred") or "")):
                    continue
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

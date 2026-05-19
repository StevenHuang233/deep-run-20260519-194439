"""LLM-as-a-judge scoring for prediction JSONL files."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import normalize_openai_base_url


JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for short-answer VQA.
Compare the model prediction with the reference answer for the given question.
Return strict JSON only: {"score": 0 or 1, "verdict": "correct" or "incorrect", "reason": "..."}.
Do not include markdown, explanations outside JSON, or hidden reasoning.
Score 1 only when the prediction is semantically equivalent to the reference answer.
Accept aliases, translations, formatting differences, units, and concise extra wording if the final answer is unambiguous.
Score 0 for empty predictions, wrong entities, wrong dates/numbers, overly broad answers, or contradictory extra content."""


def _env_int(name: str, default: str) -> int:
    return int(os.getenv(name, default))


def _env_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("judge response JSON must be an object")
    return data


def _clean_prediction(text: str) -> str:
    text = text or ""
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.S | re.I)
    if match:
        text = match.group(1)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class JudgeConfig:
    base_url: str = field(
        default_factory=lambda: os.getenv(
            "JUDGE_LLM_BASE_URL",
            os.getenv("REFLECTION_LLM_BASE_URL", ""),
        )
    )
    model_name: str = field(
        default_factory=lambda: os.getenv(
            "JUDGE_MODEL_NAME",
            os.getenv("REFLECTION_MODEL_NAME", "Qwen3-32B"),
        )
    )
    api_key: str = field(default_factory=lambda: os.getenv("JUDGE_API_KEY", "EMPTY"))
    max_tokens: int = field(default_factory=lambda: _env_int("JUDGE_MAX_TOKENS", "256"))
    temperature: float = field(default_factory=lambda: _env_float("JUDGE_TEMPERATURE", "0"))
    retry_attempts: int = field(default_factory=lambda: _env_int("JUDGE_RETRY_ATTEMPTS", "3"))
    retry_min_seconds: float = field(default_factory=lambda: _env_float("JUDGE_RETRY_MIN_SECONDS", "1"))
    retry_max_seconds: float = field(default_factory=lambda: _env_float("JUDGE_RETRY_MAX_SECONDS", "8"))

    @property
    def openai_base_url(self) -> str:
        return normalize_openai_base_url(self.base_url)


class LLMJudge:
    """Scores short-answer predictions through an OpenAI-compatible model."""

    def __init__(self, config: JudgeConfig | None = None, client: Any | None = None) -> None:
        self.config = config or JudgeConfig()
        self.client = client or self._create_client()

    def _create_client(self) -> Any:
        if not self.config.openai_base_url:
            raise ValueError("judge base URL is required; set JUDGE_LLM_BASE_URL or pass --judge-llm-url")
        from openai import OpenAI

        return OpenAI(base_url=self.config.openai_base_url, api_key=self.config.api_key)

    def score_row(self, row: dict[str, Any]) -> dict[str, Any]:
        prediction = _clean_prediction(str(row.get("pred") or ""))
        answer = str(row.get("answer") or "").strip()
        question = str(row.get("instruction") or row.get("question") or "").strip()
        if not prediction:
            return self._result(row, 0, "incorrect", "empty prediction", 0)
        if not answer:
            return self._result(row, 1, "correct", "no reference answer; non-empty prediction", 0)

        payload = {
            "question": question,
            "reference_answer": answer,
            "model_prediction": prediction,
        }
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        response, attempts = self._call_with_retry(messages)
        msg = response.choices[0].message
        content = getattr(msg, "content", None) or getattr(msg, "reasoning_content", None) or ""
        try:
            data = _extract_json(content)
            score = 1 if int(data.get("score", 0)) == 1 else 0
            verdict = str(data.get("verdict") or ("correct" if score else "incorrect")).lower()
            if verdict not in {"correct", "incorrect"}:
                verdict = "correct" if score else "incorrect"
            reason = str(data.get("reason") or "")
        except Exception as exc:
            score = self._heuristic_score(prediction, answer)
            verdict = "correct" if score else "incorrect"
            reason = f"judge_json_parse_failed:{type(exc).__name__}; fallback_exact_match={score}"
        result = self._result(row, score, verdict, reason, attempts)
        if reason.startswith("judge_json_parse_failed"):
            result["raw_judge_response"] = content[:1000]
        return result

    def _call_with_retry(self, messages: list[dict[str, str]]) -> tuple[Any, int]:
        last_exc: Exception | None = None
        max_attempts = max(1, self.config.retry_attempts)
        for attempt in range(1, max_attempts + 1):
            try:
                return (
                    self.client.chat.completions.create(
                        model=self.config.model_name,
                        messages=messages,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        extra_body={"enable_thinking": False},
                    ),
                    attempt,
                )
            except Exception as exc:
                last_exc = exc
                if attempt >= max_attempts:
                    break
                sleep_for = min(
                    self.config.retry_max_seconds,
                    self.config.retry_min_seconds * (2 ** (attempt - 1)),
                )
                time.sleep(max(sleep_for, 0))
        raise RuntimeError(f"judge LLM call failed after {max_attempts} attempts: {last_exc}")

    def _result(
        self,
        row: dict[str, Any],
        score: int,
        verdict: str,
        reason: str,
        judge_attempts: int,
    ) -> dict[str, Any]:
        return {
            "index": row.get("index"),
            "score": score,
            "verdict": verdict,
            "reason": reason,
            "instruction": row.get("instruction") or row.get("question") or "",
            "answer": row.get("answer") or "",
            "pred": _clean_prediction(str(row.get("pred") or "")),
            "image": row.get("image") or "",
            "judge_model": self.config.model_name,
            "judge_attempts": judge_attempts,
        }

    def _heuristic_score(self, prediction: str, answer: str) -> int:
        pred = re.sub(r"\s+", "", prediction).strip().lower()
        gold = re.sub(r"\s+", "", answer).strip().lower()
        return int(bool(gold) and (pred == gold or gold in pred))


def iter_prediction_rows(prediction_path: str | Path) -> Iterable[dict[str, Any]]:
    with open(prediction_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def summarize_scores(rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    scored = [row for row in rows if isinstance(row.get("score"), int)]
    correct = sum(int(row["score"]) for row in scored)
    total = len(scored)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "judge_model": model_name,
    }


def judge_predictions(
    prediction_path: str | Path,
    output_path: str | Path | None = None,
    config: JudgeConfig | None = None,
    limit: int | None = None,
    start: int = 0,
    resume: bool = False,
    client: Any | None = None,
) -> dict[str, Any]:
    prediction_path = Path(prediction_path)
    output_path = Path(output_path) if output_path else prediction_path.with_suffix(
        prediction_path.suffix + ".judge.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not resume:
        output_path.unlink()

    completed = _read_completed_indices(output_path) if resume else set()
    judge = LLMJudge(config=config, client=client)
    judged_rows: list[dict[str, Any]] = []
    processed = 0

    with open(output_path, "a", encoding="utf-8") as out:
        for row in iter_prediction_rows(prediction_path):
            index = row.get("index")
            if isinstance(index, int) and index < start:
                continue
            if isinstance(index, int) and index in completed:
                continue
            if limit is not None and processed >= limit:
                break
            judged = judge.score_row(row)
            judged_rows.append(judged)
            out.write(json.dumps(judged, ensure_ascii=False) + "\n")
            out.flush()
            processed += 1

    all_rows = list(iter_prediction_rows(output_path))
    summary = summarize_scores(all_rows, judge.config.model_name)
    summary.update(
        {
            "prediction_path": str(prediction_path),
            "judge_output_path": str(output_path),
            "processed_this_run": processed,
        }
    )
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _read_completed_indices(output_path: Path) -> set[int]:
    if not output_path.exists():
        return set()
    completed: set[int] = set()
    for row in iter_prediction_rows(output_path):
        index = row.get("index")
        if isinstance(index, int):
            completed.add(index)
    return completed

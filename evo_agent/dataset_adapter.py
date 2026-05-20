"""Adapters for local answered evaluation datasets.

The adapters intentionally split inference tasks from gold answers. The
generated task JSONL contains no answer-like fields, while the gold sidecar is
only consumed by judge/export steps after inference has finished.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET_ROOT = Path("/inspire/qb-ilm2/project/26summer-camp-01/26210500/datasets")


@dataclass(frozen=True)
class PreparedDataset:
    name: str
    task_path: Path
    gold_path: Path
    image_dir: Path | None
    total: int


_ANSWER_KEYS = {
    "answer",
    "answers",
    "gold",
    "gold_answer",
    "label",
    "reference",
    "reference_answer",
}


def dataset_source_path(name: str, dataset_root: str | Path = DEFAULT_DATASET_ROOT) -> Path:
    root = Path(dataset_root)
    key = normalize_dataset_name(name)
    if key == "2wiki":
        return root / "2wiki.jsonl"
    if key == "simplevqa":
        return root / "simpleVQA" / "SimpleVQA.jsonl"
    raise ValueError(f"unsupported dataset: {name!r}; expected 2wiki or simplevqa")


def normalize_dataset_name(name: str) -> str:
    normalized = (name or "").strip().lower().replace("_", "").replace("-", "")
    if normalized in {"2wiki", "2wikimultihopqa"}:
        return "2wiki"
    if normalized in {"simplevqa", "simplevqa2"}:
        return "simplevqa"
    return normalized


def prepare_answered_dataset(
    name: str,
    *,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    task_output: str | Path,
    gold_output: str | Path,
    limit: int | None = None,
    start: int = 0,
) -> PreparedDataset:
    dataset_name = normalize_dataset_name(name)
    source_path = dataset_source_path(dataset_name, dataset_root)
    task_path = Path(task_output)
    gold_path = Path(gold_output)
    task_path.parent.mkdir(parents=True, exist_ok=True)
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir = source_path.parent if dataset_name == "simplevqa" else None

    total = 0
    with open(task_path, "w", encoding="utf-8") as task_f, open(gold_path, "w", encoding="utf-8") as gold_f:
        for prepared_index, (source_index, row) in enumerate(
            _iter_selected_jsonl(source_path, start=start, limit=limit)
        ):
            task_row, gold_row = _adapt_row(dataset_name, row, prepared_index, source_index)
            _assert_no_answer_leak(task_row)
            task_f.write(json.dumps(task_row, ensure_ascii=False) + "\n")
            gold_f.write(json.dumps(gold_row, ensure_ascii=False) + "\n")
            total += 1

    return PreparedDataset(
        name=dataset_name,
        task_path=task_path,
        gold_path=gold_path,
        image_dir=image_dir,
        total=total,
    )


def attach_gold_answers(
    prediction_path: str | Path,
    gold_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Create a judge input JSONL by joining predictions with gold sidecar."""

    prediction_path = Path(prediction_path)
    gold_by_index = _load_gold_by_index(Path(gold_path))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prediction_path, encoding="utf-8") as pred_f, open(output_path, "w", encoding="utf-8") as out_f:
        for line in pred_f:
            if not line.strip():
                continue
            row = json.loads(line)
            index = row.get("index")
            gold = gold_by_index.get(index)
            if gold is not None:
                row["answer"] = gold.get("answer", "")
                row["reference_answer"] = gold.get("answer", "")
                row["dataset"] = gold.get("dataset", "")
                row["source_id"] = gold.get("source_id", "")
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_path


def _iter_selected_jsonl(path: Path, *, start: int, limit: int | None) -> Iterable[tuple[int, dict[str, Any]]]:
    emitted = 0
    with open(path, encoding="utf-8") as f:
        for source_index, line in enumerate(f):
            if source_index < start or not line.strip():
                continue
            if limit is not None and emitted >= limit:
                break
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            emitted += 1
            yield source_index, row


def _adapt_row(
    dataset_name: str,
    row: dict[str, Any],
    prepared_index: int,
    source_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if dataset_name == "2wiki":
        return _adapt_2wiki(row, prepared_index, source_index)
    if dataset_name == "simplevqa":
        return _adapt_simplevqa(row, prepared_index, source_index)
    raise ValueError(f"unsupported dataset: {dataset_name}")


def _adapt_2wiki(
    row: dict[str, Any],
    prepared_index: int,
    source_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_id = str(row.get("_id") or row.get("id") or source_index)
    task_row = {
        "_id": source_id,
        "question": str(row.get("question") or ""),
        "context": row.get("context") or "",
        "dataset": "2wiki",
        "source_id": source_id,
        "source_index": source_index,
    }
    gold_row = {
        "index": prepared_index,
        "task_id": f"2wiki_{source_id}",
        "dataset": "2wiki",
        "source_id": source_id,
        "source_index": source_index,
        "answer": str(row.get("answer") or ""),
        "question": str(row.get("question") or ""),
    }
    return task_row, gold_row


def _adapt_simplevqa(
    row: dict[str, Any],
    prepared_index: int,
    source_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_id = str(row.get("data_id") or row.get("id") or source_index)
    task_row = {
        "data_id": source_id,
        "question": str(row.get("question") or ""),
        "image": str(row.get("image") or ""),
        "image_url": str(row.get("image_url") or ""),
        "image_description": str(row.get("image_description") or ""),
        "language": row.get("language") or "",
        "dataset": "simplevqa",
        "source_id": source_id,
        "source_index": source_index,
    }
    gold_row = {
        "index": prepared_index,
        "task_id": f"simplevqa_{source_id}",
        "dataset": "simplevqa",
        "source_id": source_id,
        "source_index": source_index,
        "answer": str(row.get("answer") or ""),
        "question": str(row.get("question") or ""),
        "image": str(row.get("image") or ""),
        "image_url": str(row.get("image_url") or ""),
    }
    return task_row, gold_row


def _assert_no_answer_leak(row: dict[str, Any]) -> None:
    leaked = sorted(key for key in row if key.lower() in _ANSWER_KEYS)
    if leaked:
        raise ValueError(f"inference task row contains answer fields: {leaked}")
    serialized = json.dumps(row, ensure_ascii=False).lower()
    for marker in ("gold_answer", "reference_answer"):
        if marker in serialized:
            raise ValueError(f"inference task row contains forbidden marker: {marker}")


def _load_gold_by_index(path: Path) -> dict[int, dict[str, Any]]:
    gold_by_index: dict[int, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            index = row.get("index")
            if isinstance(index, int):
                gold_by_index[index] = row
    return gold_by_index

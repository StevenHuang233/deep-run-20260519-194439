"""Submission export helpers for the summer-camp benchmark."""

from __future__ import annotations

import csv
import json
import time
import zipfile
from pathlib import Path
from typing import Any


def _read_prediction_rows(prediction_path: str | Path) -> dict[int, dict[str, Any]]:
    predictions: dict[int, dict[str, Any]] = {}
    path = Path(prediction_path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            index = row.get("index")
            if index is None:
                continue
            predictions[int(index)] = row
    return predictions


def _clean_answer(row: dict[str, Any] | None) -> str:
    if not row:
        return ""
    return str(row.get("pred") or "").strip()


def _read_trajectory(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def export_group_submission(
    prediction_path: str | Path,
    benchmark_path: str | Path,
    trajectory_dir: str | Path,
    group_id: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create group_{id}.csv, group_{id}.json, and group_{id}.zip.

    The CSV keeps the benchmark columns and fills the answer column with
    predictions by row index. The JSON aggregates per-case JSONL trajectories.
    """

    if not str(group_id).strip():
        raise ValueError("group_id is required")

    prediction_path = Path(prediction_path)
    benchmark_path = Path(benchmark_path)
    trajectory_dir = Path(trajectory_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = _read_prediction_rows(prediction_path)

    csv.field_size_limit(2_147_483_647)
    with benchmark_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Benchmark CSV has no header: {benchmark_path}")
        fieldnames = list(reader.fieldnames)
        if "answer" not in fieldnames:
            fieldnames.append("answer")
        source_rows = list(reader)

    answered = 0
    group_csv = output_dir / f"group_{group_id}.csv"
    with group_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for index, row in enumerate(source_rows):
            output_row = dict(row)
            answer = _clean_answer(predictions.get(index))
            if answer:
                answered += 1
            output_row["answer"] = answer
            writer.writerow(output_row)

    trajectory_records = []
    missing_trajectory_indices: list[int] = []
    missing_answer_indices: list[int] = []
    for index, row in enumerate(source_rows):
        task_id = f"benchmark_{index:03d}"
        trajectory_path = trajectory_dir / f"{task_id}.jsonl"
        trajectory = _read_trajectory(trajectory_path)
        if not trajectory:
            missing_trajectory_indices.append(index)
        answer = _clean_answer(predictions.get(index))
        if not answer:
            missing_answer_indices.append(index)
        trajectory_records.append(
            {
                "index": index,
                "task_id": task_id,
                "question": row.get("problem") or row.get("instruction") or "",
                "answer": answer,
                "trajectory_path": str(trajectory_path),
                "trajectory": trajectory,
            }
        )

    group_json = output_dir / f"group_{group_id}.json"
    payload = {
        "group_id": str(group_id),
        "benchmark_path": str(benchmark_path),
        "prediction_path": str(prediction_path),
        "trajectory_dir": str(trajectory_dir),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_questions": len(source_rows),
        "answered": answered,
        "missing_answer_indices": missing_answer_indices,
        "missing_trajectory_indices": missing_trajectory_indices,
        "trajectories": trajectory_records,
    }
    group_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    group_zip = output_dir / f"group_{group_id}.zip"
    with zipfile.ZipFile(group_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(group_json, arcname=group_json.name)
        zf.write(group_csv, arcname=group_csv.name)

    return {
        "group_id": str(group_id),
        "csv_path": str(group_csv),
        "json_path": str(group_json),
        "zip_path": str(group_zip),
        "total_questions": len(source_rows),
        "answered": answered,
        "missing_answer_indices": missing_answer_indices,
        "missing_trajectory_indices": missing_trajectory_indices,
    }

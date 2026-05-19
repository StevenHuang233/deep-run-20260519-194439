#!/usr/bin/env bash
set -euo pipefail

# Run from this script's directory so relative paths stay stable.
cd "$(dirname "$0")"

# Keep these overrideable from the shell:
#   GROUP_ID, TASK_FILE, OUTPUT, RUN_ID, WORKERS, LIMIT, START
export LLM_BASE_URL="${LLM_BASE_URL:-https://notebook-inspire.sii.edu.cn/ws-7c23bd1d-9bae-4238-803a-737a35480e18/project-39fbffc7-dcca-4fb4-b43a-2f69f72f7e52/user-b1acf6ce-25a4-4cb6-b428-f427f4a59686/vscode/b2aa27b1-e0f7-425d-b208-acbd7f40ef68/68f1224c-8cc9-4e87-8701-523c6e59db1f/proxy/8000/}"
export MODEL_NAME="${MODEL_NAME:-Qwen3.5-9B}"
export REFLECTION_LLM_BASE_URL="${REFLECTION_LLM_BASE_URL:-https://notebook-inspire.sii.edu.cn/ws-7c23bd1d-9bae-4238-803a-737a35480e18/project-39fbffc7-dcca-4fb4-b43a-2f69f72f7e52/user-b1acf6ce-25a4-4cb6-b428-f427f4a59686/vscode/b2aa27b1-e0f7-425d-b208-acbd7f40ef68/68f1224c-8cc9-4e87-8701-523c6e59db1f/proxy/8001/}"
export REFLECTION_MODEL_NAME="${REFLECTION_MODEL_NAME:-Qwen3-32B}"
export REFLECTION_MODEL_ENABLED="${REFLECTION_MODEL_ENABLED:-1}"
export MEMORY_MODEL_ENABLED="${MEMORY_MODEL_ENABLED:-1}"
export SEARCH_PROXY_URL="${SEARCH_PROXY_URL:-http://127.0.0.1:1227}"
export SEARCH_PROXY_TIMEOUT="${SEARCH_PROXY_TIMEOUT:-90}"
export SANDBOX_BASE_URL="${SANDBOX_BASE_URL:-https://nat2-notebook-inspire.sii.edu.cn/ws-7c23bd1d-9bae-4238-803a-737a35480e18/project-39fbffc7-dcca-4fb4-b43a-2f69f72f7e52/user-b1acf6ce-25a4-4cb6-b428-f427f4a59686/vscode/78d53c3f-d7ea-41c0-b762-489e685cd0d3/eb170cdf-4c94-4f21-ac83-332c0d5daeff/proxy/8080/}"
export MAX_STEPS="${MAX_STEPS:-10}"
export MAX_TOKENS="${MAX_TOKENS:-4096}"
export TEMPERATURE="${TEMPERATURE:-0.4}"
export CONTEXT_RECENT_STEPS="${CONTEXT_RECENT_STEPS:-8}"
export TOOL_RETRY_ATTEMPTS="${TOOL_RETRY_ATTEMPTS:-2}"
export TOOL_RETRY_MIN_SECONDS="${TOOL_RETRY_MIN_SECONDS:-0.5}"
export TOOL_RETRY_MAX_SECONDS="${TOOL_RETRY_MAX_SECONDS:-2}"
export SEARCH_TEXT_MAX_TOP_K="${SEARCH_TEXT_MAX_TOP_K:-5}"
export SEARCH_TEXT_MAX_CHARS="${SEARCH_TEXT_MAX_CHARS:-600}"
export SEARCH_TOOL_MAX_CONCURRENCY="${SEARCH_TOOL_MAX_CONCURRENCY:-2}"
export BROWSER_TOOL_MAX_CONCURRENCY="${BROWSER_TOOL_MAX_CONCURRENCY:-2}"
export SEARCH_TEXT_DEFAULT_FETCH="${SEARCH_TEXT_DEFAULT_FETCH:-0}"
export SEARCH_IMAGE_DEFAULT_FETCH="${SEARCH_IMAGE_DEFAULT_FETCH:-0}"
export SEARCH_STALE_RESULT_LIMIT="${SEARCH_STALE_RESULT_LIMIT:-6}"
export SEARCH_TEXT_BROAD_QUERY_FETCH="${SEARCH_TEXT_BROAD_QUERY_FETCH:-0}"
export MAX_SEARCH_CALLS_PER_CASE="${MAX_SEARCH_CALLS_PER_CASE:-14}"
export BROWSER_URL_FAILURE_LIMIT="${BROWSER_URL_FAILURE_LIMIT:-1}"
export CASE_REFLECTION_ATTEMPTS="${CASE_REFLECTION_ATTEMPTS:-1}"
export CASE_REFLECTION_MAX_STEPS="${CASE_REFLECTION_MAX_STEPS:-5}"
export MIN_MODEL_ATTEMPTS="${MIN_MODEL_ATTEMPTS:-2}"
export ANSWER_REPAIR_ATTEMPTS="${ANSWER_REPAIR_ATTEMPTS:-2}"
export FORCED_ANSWER_ENABLED="${FORCED_ANSWER_ENABLED:-1}"
export FORCED_ANSWER_EVIDENCE_CHARS="${FORCED_ANSWER_EVIDENCE_CHARS:-9000}"
export FORCED_ANSWER_USE_REFLECTION="${FORCED_ANSWER_USE_REFLECTION:-0}"
export MAX_TOOL_CALLS_PER_STEP="${MAX_TOOL_CALLS_PER_STEP:-1}"
export RESUME_VALID_ONLY="${RESUME_VALID_ONLY:-0}"
FRESH_MEMORY="${FRESH_MEMORY:-1}"

RUN_ID="${RUN_ID:-benchmark_$(date +%Y%m%d_%H%M%S)}"
TASK_FILE="${TASK_FILE:-/inspire/qb-ilm2/project/26summer-camp-01/public/benchmark.csv}"
OUTPUT="${OUTPUT:-/inspire/qb-ilm2/project/26summer-camp-01/26210500/benchmark_runs/${RUN_ID}/predictions.jsonl}"
TRAJ_DIR="${TRAJ_DIR:-/inspire/qb-ilm2/project/26summer-camp-01/26210500/benchmark_runs/${RUN_ID}/trajectories}"
SUBMISSION_DIR="${SUBMISSION_DIR:-/inspire/qb-ilm2/project/26summer-camp-01/26210500/benchmark_runs/${RUN_ID}/submission}"
WORKERS="${WORKERS:-10}"
LIMIT="${LIMIT:-100}"
START="${START:-0}"
mkdir -p "$(dirname "$OUTPUT")" "$TRAJ_DIR" "$SUBMISSION_DIR"

ARGS=(
  --task-file "$TASK_FILE"
  --output "$OUTPUT"
  --traj-dir "$TRAJ_DIR"
  --run-id "$RUN_ID"
  --workers "$WORKERS"
  --start "$START"
  --limit "$LIMIT"
  --resume
)

if [[ -n "${GROUP_ID:-}" ]]; then
  ARGS+=(--group-id "$GROUP_ID" --submission-dir "$SUBMISSION_DIR")
fi

if [[ "$FRESH_MEMORY" != "0" ]]; then
  ARGS+=(--fresh-memory)
fi

echo "Run ID: $RUN_ID"
echo "Task file: $TASK_FILE"
echo "Output: $OUTPUT"
echo "Trajectories: $TRAJ_DIR"
echo "Workers: $WORKERS  Limit: $LIMIT  Start: $START"
echo "Fresh memory: $FRESH_MEMORY"

python -m task_runner "${ARGS[@]}"

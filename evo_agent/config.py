"""Runtime configuration for the OOP evolutionary harness."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def normalize_openai_base_url(base_url: str) -> str:
    base_url = (base_url or "").strip().rstrip("/")
    if not base_url or base_url.endswith("/v1"):
        return base_url
    return f"{base_url}/v1"


def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: str) -> int:
    return int(os.getenv(name, default))


def _env_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


@dataclass
class HarnessConfig:
    """Environment-backed configuration with current harness-compatible names.

    Defaults are read when the config instance is created, so tests and wrappers
    can set environment variables immediately before constructing the harness.
    """

    llm_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    )
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "qwen-3.5"))
    max_steps: int = field(default_factory=lambda: _env_int("MAX_STEPS", "20"))
    max_tokens: int = field(default_factory=lambda: _env_int("MAX_TOKENS", "16000"))
    temperature: float = field(default_factory=lambda: _env_float("TEMPERATURE", "1.0"))
    top_p: float = field(default_factory=lambda: _env_float("TOP_P", "0.95"))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", "20"))
    min_p: float = field(default_factory=lambda: _env_float("MIN_P", "0.0"))
    presence_penalty: float = field(default_factory=lambda: _env_float("PRESENCE_PENALTY", "1.5"))
    repetition_penalty: float = field(default_factory=lambda: _env_float("REPETITION_PENALTY", "1.0"))
    cz_sampling_compat: bool = field(default_factory=lambda: _env_bool("CZ_SAMPLING_COMPAT", "1"))
    disable_tools: bool = field(default_factory=lambda: _env_bool("DISABLE_TOOLS", "0"))
    disable_browser_tools: bool = field(default_factory=lambda: _env_bool("DISABLE_BROWSER_TOOLS", "0"))
    mock_llm: bool = field(default_factory=lambda: _env_bool("MOCK_LLM", "0"))
    llm_retry_attempts: int = field(default_factory=lambda: _env_int("LLM_RETRY_ATTEMPTS", "3"))
    llm_retry_min_seconds: float = field(default_factory=lambda: _env_float("LLM_RETRY_MIN_SECONDS", "1"))
    llm_retry_max_seconds: float = field(default_factory=lambda: _env_float("LLM_RETRY_MAX_SECONDS", "8"))
    tool_retry_attempts: int = field(default_factory=lambda: _env_int("TOOL_RETRY_ATTEMPTS", "2"))
    tool_retry_min_seconds: float = field(default_factory=lambda: _env_float("TOOL_RETRY_MIN_SECONDS", "1"))
    tool_retry_max_seconds: float = field(default_factory=lambda: _env_float("TOOL_RETRY_MAX_SECONDS", "8"))
    search_text_max_top_k: int = field(default_factory=lambda: _env_int("SEARCH_TEXT_MAX_TOP_K", "10"))
    search_text_max_chars: int = field(default_factory=lambda: _env_int("SEARCH_TEXT_MAX_CHARS", "500"))
    search_tool_max_concurrency: int = field(default_factory=lambda: _env_int("SEARCH_TOOL_MAX_CONCURRENCY", "2"))
    browser_tool_max_concurrency: int = field(default_factory=lambda: _env_int("BROWSER_TOOL_MAX_CONCURRENCY", "2"))
    search_text_default_fetch: bool = field(default_factory=lambda: _env_bool("SEARCH_TEXT_DEFAULT_FETCH", "1"))
    search_image_default_fetch: bool = field(default_factory=lambda: _env_bool("SEARCH_IMAGE_DEFAULT_FETCH", "1"))
    search_stale_result_limit: int = field(default_factory=lambda: _env_int("SEARCH_STALE_RESULT_LIMIT", "12"))
    search_text_broad_query_fetch: bool = field(
        default_factory=lambda: _env_bool("SEARCH_TEXT_BROAD_QUERY_FETCH", "0")
    )
    max_search_calls_per_case: int = field(default_factory=lambda: _env_int("MAX_SEARCH_CALLS_PER_CASE", "32"))
    browser_url_failure_limit: int = field(default_factory=lambda: _env_int("BROWSER_URL_FAILURE_LIMIT", "2"))
    max_tool_calls_per_step: int = field(default_factory=lambda: _env_int("MAX_TOOL_CALLS_PER_STEP", "3"))
    pre_tool_answer_gate_enabled: bool = field(
        default_factory=lambda: _env_bool("PRE_TOOL_ANSWER_GATE_ENABLED", "0")
    )
    context_recent_steps: int = field(default_factory=lambda: _env_int("CONTEXT_RECENT_STEPS", "20"))
    context_max_chars: int = field(default_factory=lambda: _env_int("CONTEXT_MAX_CHARS", "120000"))
    search_novelty_guard_enabled: bool = field(
        default_factory=lambda: _env_bool("SEARCH_NOVELTY_GUARD_ENABLED", "0")
    )
    batch_continue_on_error: bool = field(default_factory=lambda: _env_bool("BATCH_CONTINUE_ON_ERROR", "1"))
    min_model_attempts: int = field(default_factory=lambda: _env_int("MIN_MODEL_ATTEMPTS", "2"))
    resume_valid_only: bool = field(default_factory=lambda: _env_bool("RESUME_VALID_ONLY", "0"))
    answer_repair_attempts: int = field(default_factory=lambda: _env_int("ANSWER_REPAIR_ATTEMPTS", "4"))
    forced_answer_enabled: bool = field(default_factory=lambda: _env_bool("FORCED_ANSWER_ENABLED", "1"))
    final_answer_max_attempts: int = field(default_factory=lambda: _env_int("FINAL_ANSWER_MAX_ATTEMPTS", "10"))
    forced_answer_evidence_chars: int = field(default_factory=lambda: _env_int("FORCED_ANSWER_EVIDENCE_CHARS", "9000"))
    forced_answer_use_reflection: bool = field(default_factory=lambda: _env_bool("FORCED_ANSWER_USE_REFLECTION", "0"))
    case_reflection_attempts: int = field(
        default_factory=lambda: _env_int(
            "CASE_REFLECTION_ATTEMPTS", os.getenv("SELF_REFLECTION_ATTEMPTS", "0")
        )
    )
    case_reflection_max_steps: int = field(
        default_factory=lambda: _env_int(
            "CASE_REFLECTION_MAX_STEPS", os.getenv("SELF_REFLECTION_MAX_STEPS", "8")
        )
    )

    trajectory_dir: str = field(
        default_factory=lambda: os.getenv("TRAJECTORY_DIR", str(PACKAGE_ROOT / "trajectories"))
    )
    memory_db_path: str = field(
        default_factory=lambda: os.getenv("MEMORY_DB_PATH", str(PACKAGE_ROOT / "memory.json"))
    )
    memory_max_rules: int = field(default_factory=lambda: _env_int("MEMORY_MAX_RULES", "64"))
    memory_retrieve_top_k: int = field(default_factory=lambda: _env_int("MEMORY_RETRIEVE_TOP_K", "0"))
    clin_memory_write_enabled: bool = field(
        default_factory=lambda: _env_bool("CLIN_MEMORY_WRITE_ENABLED", "0")
    )
    dream_max_trajectories: int = field(default_factory=lambda: _env_int("DREAM_MAX_TRAJECTORIES", "64"))
    dream_min_failures: int = field(default_factory=lambda: _env_int("DREAM_MIN_FAILURES", "1"))
    dream_report_path: str = field(
        default_factory=lambda: os.getenv("DREAM_REPORT_PATH", str(PACKAGE_ROOT / "outputs" / "dream_report.json"))
    )

    reflection_enabled: bool = field(default_factory=lambda: os.getenv("REFLECTION_ENABLED", "1") != "0")
    reflection_model_enabled: bool = field(default_factory=lambda: _env_bool("REFLECTION_MODEL_ENABLED", "0"))
    search_chain_reflection_enabled: bool = field(
        default_factory=lambda: _env_bool("SEARCH_CHAIN_REFLECTION_ENABLED", "1")
    )
    reflection_base_url: str = field(
        default_factory=lambda: os.getenv(
            "REFLECTION_LLM_BASE_URL",
            os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        )
    )
    reflection_model_name: str = field(
        default_factory=lambda: os.getenv("REFLECTION_MODEL_NAME", "qwen-3-32b")
    )
    reflection_model_max_b: float = field(
        default_factory=lambda: _env_float("REFLECTION_MODEL_MAX_B", "32")
    )
    reflection_api_key: str = field(default_factory=lambda: os.getenv("REFLECTION_API_KEY", "EMPTY"))
    memory_model_enabled: bool = field(
        default_factory=lambda: _env_bool("MEMORY_MODEL_ENABLED", os.getenv("REFLECTION_MODEL_ENABLED", "0"))
    )

    case_memory_enabled: bool = field(default_factory=lambda: _env_bool("CASE_MEMORY_ENABLED", "1"))
    case_memory_model_enabled: bool = field(
        default_factory=lambda: _env_bool(
            "CASE_MEMORY_MODEL_ENABLED",
            os.getenv("REFLECTION_MODEL_ENABLED", "0"),
        )
    )
    case_memory_base_url: str = field(
        default_factory=lambda: os.getenv(
            "CASE_MEMORY_LLM_BASE_URL",
            os.getenv(
                "REFLECTION_LLM_BASE_URL",
                os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
            ),
        )
    )
    case_memory_model_name: str = field(
        default_factory=lambda: os.getenv(
            "CASE_MEMORY_MODEL_NAME",
            os.getenv("REFLECTION_MODEL_NAME", "qwen-3-32b"),
        )
    )
    case_memory_api_key: str = field(default_factory=lambda: os.getenv("CASE_MEMORY_API_KEY", "EMPTY"))
    candidate_review_base_url: str = field(
        default_factory=lambda: os.getenv(
            "CANDIDATE_REVIEW_LLM_BASE_URL",
            os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        )
    )
    candidate_review_model_name: str = field(
        default_factory=lambda: os.getenv(
            "CANDIDATE_REVIEW_MODEL_NAME",
            os.getenv("MODEL_NAME", "qwen-3.5"),
        )
    )
    candidate_review_api_key: str = field(
        default_factory=lambda: os.getenv("CANDIDATE_REVIEW_API_KEY", "EMPTY")
    )
    case_memory_log_path: str = field(
        default_factory=lambda: os.getenv(
            "CASE_MEMORY_LOG_PATH",
            str(PACKAGE_ROOT / "case_memory_records.jsonl"),
        )
    )
    strategy_memory_path: str = field(
        default_factory=lambda: os.getenv(
            "STRATEGY_MEMORY_PATH",
            str(PACKAGE_ROOT / "strategy_memory.jsonl"),
        )
    )
    strategy_memory_top_k: int = field(default_factory=lambda: _env_int("STRATEGY_MEMORY_TOP_K", "3"))
    strategy_memory_max_blocks: int = field(default_factory=lambda: _env_int("STRATEGY_MEMORY_MAX_BLOCKS", "96"))
    case_memory_max_records_prompt: int = field(
        default_factory=lambda: _env_int("CASE_MEMORY_MAX_RECORDS_PROMPT", "12")
    )
    case_memory_late_search_threshold: int = field(
        default_factory=lambda: _env_int("CASE_MEMORY_LATE_SEARCH_THRESHOLD", "3")
    )

    gate_loop_limit: int = field(default_factory=lambda: _env_int("GATE_LOOP_LIMIT", "3"))
    gate_critical_limit: int = field(default_factory=lambda: _env_int("GATE_CRITICAL_LIMIT", "6"))
    gate_similarity_threshold: float = field(
        default_factory=lambda: _env_float("GATE_SIMILARITY_THRESHOLD", "0.85")
    )

    result_dir: str = field(default_factory=lambda: os.getenv("RESULT_DIR", str(PACKAGE_ROOT / "outputs")))


def ensure_output_dirs(config: HarnessConfig) -> None:
    Path(config.trajectory_dir).mkdir(parents=True, exist_ok=True)
    Path(config.result_dir).mkdir(parents=True, exist_ok=True)
    Path(config.memory_db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(config.dream_report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(config.case_memory_log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(config.strategy_memory_path).parent.mkdir(parents=True, exist_ok=True)

"""Runtime configuration for the OOP evolutionary harness."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


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
    disable_tools: bool = field(default_factory=lambda: _env_bool("DISABLE_TOOLS", "0"))
    mock_llm: bool = field(default_factory=lambda: _env_bool("MOCK_LLM", "0"))
    llm_retry_attempts: int = field(default_factory=lambda: _env_int("LLM_RETRY_ATTEMPTS", "3"))
    llm_retry_min_seconds: float = field(default_factory=lambda: _env_float("LLM_RETRY_MIN_SECONDS", "1"))
    llm_retry_max_seconds: float = field(default_factory=lambda: _env_float("LLM_RETRY_MAX_SECONDS", "8"))
    context_recent_steps: int = field(default_factory=lambda: _env_int("CONTEXT_RECENT_STEPS", "8"))
    batch_continue_on_error: bool = field(default_factory=lambda: _env_bool("BATCH_CONTINUE_ON_ERROR", "1"))
    min_model_attempts: int = field(default_factory=lambda: _env_int("MIN_MODEL_ATTEMPTS", "5"))
    case_reflection_attempts: int = field(
        default_factory=lambda: _env_int(
            "CASE_REFLECTION_ATTEMPTS", os.getenv("SELF_REFLECTION_ATTEMPTS", "4")
        )
    )
    case_reflection_max_steps: int = field(
        default_factory=lambda: _env_int(
            "CASE_REFLECTION_MAX_STEPS", os.getenv("SELF_REFLECTION_MAX_STEPS", "6")
        )
    )

    trajectory_dir: str = field(
        default_factory=lambda: os.getenv("TRAJECTORY_DIR", str(PACKAGE_ROOT / "trajectories"))
    )
    memory_db_path: str = field(
        default_factory=lambda: os.getenv("MEMORY_DB_PATH", str(PACKAGE_ROOT / "memory.json"))
    )
    memory_max_rules: int = field(default_factory=lambda: _env_int("MEMORY_MAX_RULES", "64"))
    dream_max_trajectories: int = field(default_factory=lambda: _env_int("DREAM_MAX_TRAJECTORIES", "64"))
    dream_min_failures: int = field(default_factory=lambda: _env_int("DREAM_MIN_FAILURES", "1"))
    dream_report_path: str = field(
        default_factory=lambda: os.getenv("DREAM_REPORT_PATH", str(PACKAGE_ROOT / "outputs" / "dream_report.json"))
    )

    reflection_enabled: bool = field(default_factory=lambda: os.getenv("REFLECTION_ENABLED", "1") != "0")
    reflection_model_enabled: bool = field(default_factory=lambda: _env_bool("REFLECTION_MODEL_ENABLED", "0"))
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

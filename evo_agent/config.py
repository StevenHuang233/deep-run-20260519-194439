"""Runtime configuration for the OOP evolutionary harness."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class HarnessConfig:
    """Environment-backed configuration with current harness-compatible names."""

    llm_base_url: str = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    model_name: str = os.getenv("MODEL_NAME", "qwen-3.5")
    max_steps: int = int(os.getenv("MAX_STEPS", "20"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "16000"))
    temperature: float = float(os.getenv("TEMPERATURE", "1.0"))
    disable_tools: bool = os.getenv("DISABLE_TOOLS", "0") == "1"
    mock_llm: bool = os.getenv("MOCK_LLM", "0") == "1"

    trajectory_dir: str = os.getenv(
        "TRAJECTORY_DIR", str(PACKAGE_ROOT / "trajectories")
    )
    memory_db_path: str = os.getenv(
        "MEMORY_DB_PATH", str(PACKAGE_ROOT / "memory.json")
    )

    reflection_enabled: bool = os.getenv("REFLECTION_ENABLED", "1") != "0"
    reflection_model_enabled: bool = os.getenv("REFLECTION_MODEL_ENABLED", "0") == "1"
    reflection_base_url: str = os.getenv(
        "REFLECTION_LLM_BASE_URL", os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    )
    reflection_model_name: str = os.getenv("REFLECTION_MODEL_NAME", "qwen-3-32b")
    reflection_model_max_b: float = float(os.getenv("REFLECTION_MODEL_MAX_B", "32"))
    reflection_api_key: str = os.getenv("REFLECTION_API_KEY", "EMPTY")
    memory_model_enabled: bool = os.getenv("MEMORY_MODEL_ENABLED", os.getenv("REFLECTION_MODEL_ENABLED", "0")) == "1"

    gate_loop_limit: int = int(os.getenv("GATE_LOOP_LIMIT", "3"))
    gate_critical_limit: int = int(os.getenv("GATE_CRITICAL_LIMIT", "6"))
    gate_similarity_threshold: float = float(
        os.getenv("GATE_SIMILARITY_THRESHOLD", "0.85")
    )

    result_dir: str = os.getenv("RESULT_DIR", str(PACKAGE_ROOT / "outputs"))


def ensure_output_dirs(config: HarnessConfig) -> None:
    Path(config.trajectory_dir).mkdir(parents=True, exist_ok=True)
    Path(config.result_dir).mkdir(parents=True, exist_ok=True)
    Path(config.memory_db_path).parent.mkdir(parents=True, exist_ok=True)

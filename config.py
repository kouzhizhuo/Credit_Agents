"""CreditAgent 运行时配置。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple


def _first_non_empty(*values: Optional[str], default: Optional[str] = None) -> Optional[str]:
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return default


def _default_benchmark_file() -> str:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    local_candidates = (
        os.path.join(
            project_root, "creditagent", "data", "credit_risk_assessment_benchmark.jsonl"
        ),
        os.path.join(os.path.dirname(__file__), "data", "data_sample.jsonl"),
    )
    for local_data in local_candidates:
        if os.path.isfile(local_data):
            return local_data
    return _first_non_empty(
        os.getenv("CREDITAGENT_INPUT_FILE", ""),
        os.getenv("INPUT_FILE", ""),
        default="",
    ) or ""


def _shared_model_name() -> str:
    return _first_non_empty(
        os.getenv("CREDITAGENT_MODEL", ""),
        os.getenv("MODEL", ""),
        default="",
    ) or ""


def _shared_base_url() -> str:
    return _first_non_empty(
        os.getenv("CREDITAGENT_BASE_URL", ""),
        os.getenv("BASE_URL", ""),
        default="",
    ) or ""


def _shared_api_key() -> str:
    return _first_non_empty(
        os.getenv("CREDITAGENT_API_KEY", ""),
        os.getenv("API_KEY", ""),
        default="EMPTY",
    ) or "EMPTY"


# ---------------------------- Layer 配置 ---------------------------- #
@dataclass
class ScenarioConfig:
    """任务/场景级配置 —— 数据路径、目标列、用户/时间列、输出目录等。"""

    target_col: str = "target"
    user_col: str = "user_id"
    time_col: str = "task_create_date"
    output_dir: str = "./outputs"
    default_input_file: str = field(
        default_factory=_default_benchmark_file
    )


@dataclass
class Layer1Config:
    """Layer 1 特征治理管道参数。"""

    target_features: int = 100
    missing_threshold: float = 0.95
    outlier_method: str = "iqr"
    prune_ratio: float = 0.1
    cv_folds: int = 5
    lambda_weight: float = 0.7             # S(f) = λ·Gain + (1-λ)·Rel_llm
    dict_file: Optional[str] = None        # 特征字典 xlsx (可选)
    output_filename: str = "selected_100_features.csv"
    enable_llm_final_judge: bool = True
    final_judge_batch_size: int = 25
    final_judge_drop_threshold: float = 3.0


@dataclass
class Layer2Config:
    """Layer 2 六维度评估参数。"""

    parallel: bool = False
    temperature: float = 0.0
    timeout: int = 60
    enable_tools: bool = True
    max_tool_iterations: int = 2
    enabled_tools: Tuple[str, ...] = field(
        default_factory=lambda: ("web_search", "document_reader", "knowledge_base")
    )
    knowledge_base_dir: str = "./docs"
    web_search_max_results: int = 5


@dataclass
class Layer3Config:
    """Layer 3 融合 & 决策阈值。"""

    threshold: float = 50.0
    margin: float = 10.0
    temperature: float = 0.2
    max_tokens: int = 4096

    @classmethod
    def from_env(cls) -> "Layer3Config":
        threshold = float(os.getenv("CREDITAGENT_DECISION_THRESHOLD", "50.0"))
        margin = float(os.getenv("CREDITAGENT_DECISION_MARGIN", "10.0"))
        return cls(threshold=threshold, margin=margin)


# ---------------------------- LLM 配置 ---------------------------- #
@dataclass
class LayerLLMConfig:
    """单层 LLM 客户端配置。"""

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout: int = 60

    @property
    def enabled(self) -> bool:
        return bool(self.model) and bool(self.api_key) and self.api_key != "EMPTY"


@dataclass
class LLMConfig:
    """三层 LLM 客户端的整体配置。"""

    layer1: LayerLLMConfig = field(default_factory=LayerLLMConfig)
    layer2: LayerLLMConfig = field(default_factory=LayerLLMConfig)
    layer3: LayerLLMConfig = field(default_factory=LayerLLMConfig)

    @classmethod
    def from_env(cls) -> "LLMConfig":
        shared_model = _shared_model_name()
        shared_base_url = _shared_base_url()
        shared_api_key = _shared_api_key()
        return cls(
            layer1=LayerLLMConfig(
                api_key=_first_non_empty(
                    os.getenv("CREDITAGENT_LAYER1_API_KEY", ""),
                    os.getenv("LAYER1_API_KEY", ""),
                    shared_api_key,
                ),
                base_url=_first_non_empty(
                    os.getenv("CREDITAGENT_LAYER1_BASE_URL", ""),
                    os.getenv("LAYER1_BASE_URL", ""),
                    shared_base_url,
                ),
                model=_first_non_empty(
                    os.getenv("CREDITAGENT_LAYER1_MODEL", ""),
                    os.getenv("LAYER1_MODEL", ""),
                    shared_model,
                ),
                temperature=0.2,
                max_tokens=1024,
            ),
            layer2=LayerLLMConfig(
                api_key=_first_non_empty(
                    os.getenv("CREDITAGENT_LAYER2_API_KEY", ""),
                    os.getenv("LAYER2_API_KEY", ""),
                    shared_api_key,
                ),
                base_url=_first_non_empty(
                    os.getenv("CREDITAGENT_LAYER2_BASE_URL", ""),
                    os.getenv("LAYER2_BASE_URL", ""),
                    shared_base_url,
                ),
                model=_first_non_empty(
                    os.getenv("CREDITAGENT_LAYER2_MODEL", ""),
                    os.getenv("LAYER2_MODEL", ""),
                    shared_model,
                ),
                temperature=0.0,
                max_tokens=2048,
            ),
            layer3=LayerLLMConfig(
                api_key=_first_non_empty(
                    os.getenv("CREDITAGENT_LAYER3_API_KEY", ""),
                    os.getenv("LAYER3_API_KEY", ""),
                    shared_api_key,
                ),
                base_url=_first_non_empty(
                    os.getenv("CREDITAGENT_LAYER3_BASE_URL", ""),
                    os.getenv("LAYER3_BASE_URL", ""),
                    shared_base_url,
                ),
                model=_first_non_empty(
                    os.getenv("CREDITAGENT_LAYER3_MODEL", ""),
                    os.getenv("LAYER3_MODEL", ""),
                    shared_model,
                ),
                temperature=0.2,
                max_tokens=4096,
            ),
        )


# ---------------------------- 顶层容器 ---------------------------- #
@dataclass
class RuntimeConfig:
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    layer1: Layer1Config = field(default_factory=Layer1Config)
    layer2: Layer2Config = field(default_factory=Layer2Config)
    layer3: Layer3Config = field(default_factory=Layer3Config.from_env)
    llm: LLMConfig = field(default_factory=LLMConfig.from_env)
    model_tag: str = field(
        default_factory=lambda: _first_non_empty(
            os.getenv("CREDITAGENT_MODEL_TAG", ""),
            os.getenv("MODEL_TAG", ""),
            default="default",
        )
        or "default"
    )


__all__ = [
    "ScenarioConfig",
    "Layer1Config",
    "Layer2Config",
    "Layer3Config",
    "LayerLLMConfig",
    "LLMConfig",
    "RuntimeConfig",
]

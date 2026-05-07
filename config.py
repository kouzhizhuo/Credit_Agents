"""CreditAgent runtime configuration.

All sub-configs are expressed as dataclasses; ``RuntimeConfig`` is the single
public container.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple


def _try_import_project_config() -> Optional[Any]:
    """尝试从项目根目录导入旧 ``config.Config``，失败返回 None。"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        from config import Config  # type: ignore
        return Config
    except Exception:
        return None


PROJECT_CONFIG = _try_import_project_config()


# ---------------------------- Layer 配置 ---------------------------- #
@dataclass
class ScenarioConfig:
    """任务/场景级配置 —— 数据路径、目标列、用户/时间列、输出目录等。"""

    target_col: str = "target"
    user_col: str = "user_id"
    time_col: str = "task_create_date"
    output_dir: str = "./outputs"
    default_input_file: str = field(
        default_factory=lambda: getattr(
            PROJECT_CONFIG, "DEFAULT_INPUT_FILE", ""
        ) if PROJECT_CONFIG else ""
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
    # LLM 最终判断 (新)
    enable_llm_final_judge: bool = True
    final_judge_batch_size: int = 25
    # LLM 最终判定分数（0-10）小于该阈值则剔除
    final_judge_drop_threshold: float = 3.0


@dataclass
class Layer2Config:
    """Layer 2 六维度评估参数。"""

    parallel: bool = False
    temperature: float = 0.0
    timeout: int = 60
    # 工具增强 (新)
    enable_tools: bool = True
    max_tool_iterations: int = 2           # 每个维度最多触发几轮工具调用
    enabled_tools: Tuple[str, ...] = field(
        default_factory=lambda: ("web_search", "document_reader", "knowledge_base")
    )
    knowledge_base_dir: str = "./docs"     # KnowledgeBaseTool 检索根目录
    web_search_max_results: int = 5


@dataclass
class Layer3Config:
    """Layer 3 融合 & 决策阈值。"""

    threshold: float = 50.0
    margin: float = 10.0
    temperature: float = 0.2
    max_tokens: int = 4096

    @classmethod
    def from_project_config(cls) -> "Layer3Config":
        if PROJECT_CONFIG is None:
            return cls()
        return cls(
            threshold=float(getattr(PROJECT_CONFIG, "DECISION_THRESHOLD", 50.0)),
            margin=float(getattr(PROJECT_CONFIG, "DECISION_MARGIN", 10.0)),
        )


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
    def from_project_config(cls) -> "LLMConfig":
        if PROJECT_CONFIG is None:
            return cls()
        return cls(
            layer1=LayerLLMConfig(
                api_key=getattr(PROJECT_CONFIG, "LAYER1_API_KEY", None),
                base_url=getattr(PROJECT_CONFIG, "LAYER1_BASE_URL", None),
                model=getattr(PROJECT_CONFIG, "LAYER1_MODEL", None),
                temperature=0.2,
                max_tokens=1024,
            ),
            layer2=LayerLLMConfig(
                api_key=getattr(PROJECT_CONFIG, "LAYER2_API_KEY", None),
                base_url=getattr(PROJECT_CONFIG, "LAYER2_BASE_URL", None),
                model=getattr(PROJECT_CONFIG, "LAYER2_MODEL", None),
                temperature=0.0,
                max_tokens=2048,
            ),
            layer3=LayerLLMConfig(
                api_key=getattr(PROJECT_CONFIG, "LAYER3_API_KEY", None),
                base_url=getattr(PROJECT_CONFIG, "LAYER3_BASE_URL", None),
                model=getattr(PROJECT_CONFIG, "LAYER3_MODEL", None),
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
    layer3: Layer3Config = field(default_factory=Layer3Config.from_project_config)
    llm: LLMConfig = field(default_factory=LLMConfig.from_project_config)
    model_tag: str = field(
        default_factory=lambda: getattr(
            PROJECT_CONFIG, "DEFAULT_MODEL_NAME", "creditagent"
        ) if PROJECT_CONFIG else "creditagent"
    )


__all__ = [
    "ScenarioConfig",
    "Layer1Config",
    "Layer2Config",
    "Layer3Config",
    "LayerLLMConfig",
    "LLMConfig",
    "RuntimeConfig",
    "PROJECT_CONFIG",
]

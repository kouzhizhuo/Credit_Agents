"""Layer2Pipeline: 六维度专业评估。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..agents.layer2 import (
    DEFAULT_DIM_SPECS,
    DimensionAgent,
    build_six_dimension_agents,
    run_six_dimension_scoring,
)
from ..config import Layer2Config, LayerLLMConfig
from ..core.blackboard import SharedBlackboard
from ..core.llm_client import LLMClient, build_llm_client_from_layer_cfg
from ..tools.registry import ToolRegistry, build_default_tool_registry


@dataclass
class Layer2Result:
    scores: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    blackboard_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "六大维度评分": dict(self.scores),
            "六大维度明细": dict(self.details),
            "共享黑板": dict(self.blackboard_snapshot),
        }


class Layer2Pipeline:
    """Layer 2 端到端流水线：指令 → 六维度评分。"""

    def __init__(
        self,
        *,
        layer2_cfg: Layer2Config,
        llm_cfg: Optional[LayerLLMConfig] = None,
        llm: Optional[LLMClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
        verbose: bool = False,
    ):
        self.cfg = layer2_cfg
        self.verbose = verbose
        if llm is not None:
            self.llm = llm
        elif llm_cfg is not None:
            self.llm = build_llm_client_from_layer_cfg(llm_cfg)
        else:
            self.llm = LLMClient()
        if tool_registry is not None:
            self.tool_registry = tool_registry
        elif layer2_cfg.enable_tools:
            self.tool_registry = build_default_tool_registry(
                enabled=layer2_cfg.enabled_tools,
                knowledge_base_dir=layer2_cfg.knowledge_base_dir,
                web_search_max_results=layer2_cfg.web_search_max_results,
            )
        else:
            self.tool_registry = None

    # ---- API ----
    def build_agents(
        self, blackboard: Optional[SharedBlackboard] = None
    ) -> Dict[str, DimensionAgent]:
        return build_six_dimension_agents(
            self.llm,
            specs=DEFAULT_DIM_SPECS,
            blackboard=blackboard,
            tool_registry=self.tool_registry,
            max_tool_iterations=self.cfg.max_tool_iterations,
            verbose=self.verbose,
        )

    def run(
        self,
        dim_texts: Dict[str, str],
        *,
        blackboard: Optional[SharedBlackboard] = None,
    ) -> Layer2Result:
        bb = blackboard or SharedBlackboard()
        agents = self.build_agents(blackboard=bb)
        res = run_six_dimension_scoring(
            agents, dim_texts=dim_texts, blackboard=bb, parallel=self.cfg.parallel
        )
        return Layer2Result(
            scores=res.get("六大维度评分", {}),
            details=res.get("六大维度明细", {}),
            blackboard_snapshot=res.get("共享黑板", {}),
        )


__all__ = ["Layer2Pipeline", "Layer2Result"]

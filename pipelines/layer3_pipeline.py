"""Layer3Pipeline: 决策融合 + 策略分析。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..agents.layer3 import ScoreDecisionAgent, StrategyAnalysisAgent
from ..config import Layer3Config, LayerLLMConfig
from ..core.llm_client import LLMClient, build_llm_client_from_layer_cfg


@dataclass
class Layer3Result:
    final_risk_score: Optional[float] = None
    final_decision: str = "未执行Layer 3"
    score_decision: Dict[str, Any] = field(default_factory=dict)
    strategy_analysis: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_risk_score": self.final_risk_score,
            "final_decision": self.final_decision,
            "score_decision": dict(self.score_decision),
            "strategy_analysis": dict(self.strategy_analysis),
        }


class Layer3Pipeline:
    """Layer 3 端到端流水线。"""

    def __init__(
        self,
        *,
        layer3_cfg: Layer3Config,
        llm_cfg: Optional[LayerLLMConfig] = None,
        llm: Optional[LLMClient] = None,
    ):
        self.cfg = layer3_cfg
        if llm is not None:
            self.llm = llm
        elif llm_cfg is not None:
            self.llm = build_llm_client_from_layer_cfg(llm_cfg)
        else:
            self.llm = LLMClient()

    # ---- API ----
    def run(
        self,
        six_dim_scores: Dict[str, Any],
        six_dim_details: Dict[str, Any],
    ) -> Layer3Result:
        score_agent = ScoreDecisionAgent(
            llm=self.llm,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        score_result = score_agent.compute_risk_score(
            six_dim_scores, six_dim_details
        )
        final_risk_score = float(score_result.get("final_risk_score", 50.0))

        strategy_agent = StrategyAnalysisAgent(
            llm=self.llm,
            threshold=self.cfg.threshold,
            margin=self.cfg.margin,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        strategy_result = strategy_agent.analyze_strategy(
            final_risk_score, six_dim_details, score_result
        )
        return Layer3Result(
            final_risk_score=final_risk_score,
            final_decision=str(strategy_result.get("final_decision", "未知")),
            score_decision=score_result,
            strategy_analysis=strategy_result,
        )


__all__ = ["Layer3Pipeline", "Layer3Result"]

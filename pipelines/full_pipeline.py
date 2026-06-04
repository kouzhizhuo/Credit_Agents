"""FullPipeline: Layer1 → Layer2 → Layer3 的端到端编排。

Layer1 输入是表格数据 (pandas.DataFrame)；Layer2/Layer3 的输入是单样本
``instruction`` 字符串（内含六维度分段，解析规则见 ``utils.extract_dims_from_instruction``）。
因此 Full Pipeline 分开提供 ``run_layer1`` / ``run_layer23`` / ``run_per_sample``。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import RuntimeConfig
from ..core.blackboard import SharedBlackboard
from .layer1_pipeline import Layer1Pipeline, Layer1Result
from .layer2_pipeline import Layer2Pipeline, Layer2Result
from .layer3_pipeline import Layer3Pipeline, Layer3Result


def _extract_dims_from_instruction(instruction_text: str) -> Dict[str, str]:
    """延迟导入 ``utils.extract_dims_from_instruction``，降低模块耦合。"""
    try:
        from utils import extract_dims_from_instruction  # type: ignore

        return extract_dims_from_instruction(instruction_text)
    except Exception:
        return {}


def _dims_to_dim_texts(dims: Dict[str, str]) -> Dict[str, str]:
    get = lambda k: (dims.get(k, "") or "无相关数据").strip()
    return {
        "申请合理性": get("dim6"),
        "借款人基本信息": "\n".join([get("dim1_1"), get("dim1_2")]).strip(),
        "偿还能力": "\n".join([get("dim2_1"), get("dim2_2")]).strip(),
        "偿还意愿": "\n".join([get("dim3_1"), get("dim3_2")]).strip(),
        "信用历史": "\n".join([get("dim4_1"), get("dim4_2")]).strip(),
        "欺诈风险": get("dim5"),
    }


class FullPipeline:
    """Layer1 + Layer2 + Layer3 一体化容器。"""

    def __init__(self, cfg: RuntimeConfig, *, verbose: bool = False):
        self.cfg = cfg
        self.verbose = verbose
        self.layer1 = Layer1Pipeline(
            scenario=cfg.scenario,
            layer1_cfg=cfg.layer1,
            llm_cfg=cfg.llm.layer1,
            verbose=verbose,
        )
        self.layer2 = Layer2Pipeline(
            layer2_cfg=cfg.layer2,
            llm_cfg=cfg.llm.layer2,
            verbose=verbose,
        )
        self.layer3 = Layer3Pipeline(
            layer3_cfg=cfg.layer3,
            llm_cfg=cfg.llm.layer3,
        )

    # ---- Layer 1 ----
    def run_layer1(self, df: pd.DataFrame) -> Layer1Result:
        return self.layer1.run(df)

    # ---- Layer 2/3 串联 ----
    def run_layer23(
        self,
        dim_texts: Dict[str, str],
        blackboard: Optional[SharedBlackboard] = None,
    ) -> Dict[str, Any]:
        l2 = self.layer2.run(dim_texts, blackboard=blackboard)
        l3 = self.layer3.run(l2.scores, l2.details)
        return {
            "layer2_result": l2.to_dict(),
            "layer3_score_decision": l3.score_decision,
            "layer3_strategy_analysis": l3.strategy_analysis,
            "final_risk_score": l3.final_risk_score,
            "final_decision": l3.final_decision,
        }

    # ---- 单样本 Layer 2+3 ----
    def run_per_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        instruction = sample.get("instruction", "")
        dims = _extract_dims_from_instruction(instruction)
        dim_texts = _dims_to_dim_texts(dims)
        result: Dict[str, Any] = {
            "instruction": instruction,
            "ground_truth": sample.get("simple_result", sample.get("ground_truth", "")),
        }
        result.update(self.run_layer23(dim_texts))
        return result

    # ---- 批量样本 ----
    def run_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for s in samples:
            try:
                out.append(self.run_per_sample(s))
            except Exception as e:
                print(f"[FullPipeline][ERROR] 样本失败: {e}")
        return out


__all__ = ["FullPipeline"]

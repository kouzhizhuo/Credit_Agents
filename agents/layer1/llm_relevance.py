"""Layer 1 · LLMRelevanceAgent

对每个特征给出 0-10 的业务语义相关性评分 (Rel_llm)，归一化到 [0, 1]，
用于混合评分 ``S(f_i) = λ · Gain(f_i) + (1 - λ) · Rel_llm(f_i)`` (Eq. 9)。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ...core.llm_client import LLMClient
from ..base import BaseAgent
from .schema import FeatureMetadata


LLM_FEATURE_RELEVANCE_PROMPT = """你是一位信贷风控专家。请从信贷审批、违约预测与业务可解释性角度，对下列特征的相关度打分。

评分规则：0-10 分，分数越高表示该特征对信用风险评估越重要或越具业务解释价值（含统计上低频但专家认为重要的"黑天鹅"类指标）。

特征列表（每行：特征名 + 简短描述）：
{feature_list}

请仅输出一个 JSON 对象，不要其他文字。键为特征名（与上面列表完全一致），值为 0-10 的整数或浮点数。
```json
{{ "特征名1": 8, "特征名2": 3, ... }}
```
"""


class LLMRelevanceAgent(BaseAgent):
    """Layer 1 LLM 语义相关性评分 Agent。"""

    name = "layer1.llm_relevance"

    def __init__(self, llm: LLMClient, batch_size: int = 25, verbose: bool = True):
        self.llm = llm
        self.batch_size = batch_size
        self.verbose = verbose

    # ---- API ----
    def run(
        self,
        feature_names: List[str],
        metadata_list: Optional[List[FeatureMetadata]] = None,
    ) -> Dict[str, float]:
        return self.score_features(feature_names, metadata_list)

    def score_features(
        self,
        feature_names: List[str],
        metadata_list: Optional[List[FeatureMetadata]] = None,
    ) -> Dict[str, float]:
        if not self.llm or not self.llm.enabled:
            # 无 LLM → 所有特征中性分
            return {name: 0.5 for name in feature_names}

        meta_dict = {m.name: m for m in (metadata_list or [])}
        result: Dict[str, float] = {}
        total_batches = (len(feature_names) + self.batch_size - 1) // self.batch_size
        self._log(
            f"        LLM 语义相关性评分: {len(feature_names)} 个特征 × {total_batches} 批"
        )

        for bi, start in enumerate(range(0, len(feature_names), self.batch_size), 1):
            batch = feature_names[start : start + self.batch_size]
            lines = []
            for name in batch:
                desc = (meta_dict.get(name).description if meta_dict.get(name) else "") or name
                lines.append(f"- {name}: {desc}")
            prompt = LLM_FEATURE_RELEVANCE_PROMPT.format(feature_list="\n".join(lines))
            obj = self.llm.chat_json(prompt, temperature=0.2, max_tokens=1024)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k not in batch:
                        continue
                    try:
                        result[k] = max(0.0, min(1.0, float(v) / 10.0))
                    except (TypeError, ValueError):
                        result[k] = 0.5
                self._log(
                    f"        批次 {bi}/{total_batches}: 完成, 命中 {len(result)}"
                )
            else:
                self._log(
                    f"        [WARNING] 批次 {bi}/{total_batches} 解析失败, 中性分兜底"
                )
                for name in batch:
                    result.setdefault(name, 0.5)

        for name in feature_names:
            result.setdefault(name, 0.5)
        return result

    # ---- helpers ----
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


__all__ = ["LLMRelevanceAgent", "LLM_FEATURE_RELEVANCE_PROMPT"]

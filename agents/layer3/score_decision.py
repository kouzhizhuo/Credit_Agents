"""Layer 3 · ScoreDecisionAgent (The Quantifier)

将 Layer 2 六个维度的 ``(r_j, e_j)`` 融合为 ``S_final ∈ [0, 100]``：
- 采用 Constraint-Aware Fusion：欺诈维度具有"否决"权重；
- 输出同时包含 ``dimension_weights`` / ``fusion_logic`` / ``key_risk_factors`` 供审计链使用。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ...core.llm_client import LLMClient
from ...core.utils import safe_json_loads
from ..base import BaseAgent


SCORE_DECISION_PROMPT = """### 背景

你是一位资深信贷风控专家，担任 **评分决策Agent（量化器）** 的角色。
你的任务是将 Layer 2 六个专业维度的评估结果融合为最终的加权风险分数。

### Layer 2 评估结果

**六大维度评分：**
```json
{scores_text}
```

**六大维度详细评估：**
```json
{details_text}
```

### 任务要求

1. **计算加权风险分数**: S_final ∈ [0, 100]，分数越高表示风险越低/信用越好。
2. **Constraint-Aware Fusion**: 欺诈风险具有否决权重 (评分≤3应显著降低最终分数)。
3. **输出严格 JSON**:
   - `final_risk_score`: 最终风险分数 (0-100 浮点数)
   - `dimension_weights`: 每个维度的权重 (0-1, 字典)
   - `fusion_logic`: 融合逻辑说明 (<=200 字符)
   - `key_risk_factors`: 关键风险因子列表 (<=3 项)

```json
{{
  "final_risk_score": 75.5,
  "dimension_weights": {{
    "申请合理性": 0.15,
    "借款人基本信息": 0.15,
    "偿还能力": 0.20,
    "偿还意愿": 0.15,
    "信用历史": 0.20,
    "欺诈风险": 0.15
  }},
  "fusion_logic": "...",
  "key_risk_factors": ["..."]
}}
```
"""


@dataclass
class ScoreDecisionAgent(BaseAgent):
    llm: LLMClient
    max_retries: int = 3
    temperature: float = 0.3
    max_tokens: int = 4096
    name: str = "layer3.score_decision"

    # ---- API ----
    def run(
        self,
        six_dim_scores: Dict[str, Any],
        six_dim_details: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self.compute_risk_score(six_dim_scores, six_dim_details)

    def compute_risk_score(
        self,
        six_dim_scores: Dict[str, Any],
        six_dim_details: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self.llm or not self.llm.enabled:
            return self._fallback_fusion(six_dim_scores)

        prompt = SCORE_DECISION_PROMPT.format(
            scores_text=json.dumps(six_dim_scores, ensure_ascii=False, indent=2),
            details_text=json.dumps(six_dim_details, ensure_ascii=False, indent=2),
        )
        for _ in range(max(1, self.max_retries)):
            obj = self.llm.chat_json(
                prompt, temperature=self.temperature, max_tokens=self.max_tokens
            )
            if isinstance(obj, dict) and "final_risk_score" in obj:
                return obj
            print("[ScoreDecisionAgent][WARNING] 提取 JSON 失败，重试 ...")
        # 重试仍失败 → 规则兜底
        return self._fallback_fusion(six_dim_scores)

    # ---- helpers ----
    @staticmethod
    def _fallback_fusion(six_dim_scores: Dict[str, Any]) -> Dict[str, Any]:
        """LLM 不可用时的保底融合：线性加权 + 欺诈 veto。"""
        default_weights = {
            "申请合理性": 0.15,
            "借款人基本信息": 0.15,
            "偿还能力": 0.20,
            "偿还意愿": 0.15,
            "信用历史": 0.20,
            "欺诈风险": 0.15,
        }
        total = 0.0
        for name, w in default_weights.items():
            raw = six_dim_scores.get(name)
            try:
                r = float(raw) if raw is not None else 5.0
            except (TypeError, ValueError):
                r = 5.0
            total += w * (r / 10.0)
        final_score = round(total * 100, 2)
        # 欺诈 Veto: 若 ≤ 3，整体 × 0.6
        fraud = six_dim_scores.get("欺诈风险")
        try:
            if fraud is not None and float(fraud) <= 3:
                final_score = round(final_score * 0.6, 2)
        except (TypeError, ValueError):
            pass
        return {
            "final_risk_score": final_score,
            "dimension_weights": default_weights,
            "fusion_logic": "LLM 未启用，使用默认线性加权 + 欺诈否决兜底。",
            "key_risk_factors": [],
        }


__all__ = ["ScoreDecisionAgent", "SCORE_DECISION_PROMPT"]

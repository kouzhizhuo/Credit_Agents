"""Layer 3 · StrategyAnalysisAgent (The Policy Guard)

- Credit Strategy Matrix + Hard-Stop 判定；
- 当 ``S_final`` 距离阈值 ``τ`` 不超过 ``margin`` 时触发 Deep Review；
- 输出最终 decision: 通过 / 拒绝 / 深度审查。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ...core.llm_client import LLMClient
from ...core.utils import safe_json_loads
from ..base import BaseAgent


STRATEGY_PROMPT = """### 背景

你是一位资深信贷风控专家，担任 **策略分析Agent（Policy Guard）** 的角色。
你的任务是应用机构的 Credit Strategy Matrix，对边界情况进行深度审查。

### 当前评估状态

**最终风险分数 S_final**: {final_risk_score:.2f}/100
**决策阈值 τ**: {threshold:.2f}
**边界区域**: [{lower:.2f}, {upper:.2f}]

**Layer 2 详细评估结果：**
```json
{details_text}
```

**Score Decision Agent 融合结果：**
```json
{score_result_text}
```

### 任务要求

1. 应用 Credit Strategy Matrix，评估申请是否符合机构信贷策略。
2. 如果 S_final 在边界区域内，需要触发 Deep Review，并指出要重新审查的维度。
3. 检查 Hard-Stop（欺诈极高、重大诉讼等），触发则必须拒绝。
4. 输出严格 JSON：

```json
{{
  "strategy_assessment": "...",
  "risk_level": "中风险",
  "dimensions_to_review": ["..."],
  "review_reasons": ["..."],
  "hard_stop_triggered": false,
  "hard_stop_reason": "",
  "recommendation": "深度审查"
}}
```
"""


@dataclass
class StrategyAnalysisAgent(BaseAgent):
    llm: LLMClient
    threshold: float = 50.0
    margin: float = 10.0
    max_retries: int = 3
    temperature: float = 0.2
    max_tokens: int = 4096
    name: str = "layer3.strategy_analysis"

    # ---- API ----
    def run(
        self,
        final_risk_score: float,
        six_dim_details: Dict[str, Any],
        score_decision_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self.analyze_strategy(
            final_risk_score, six_dim_details, score_decision_result
        )

    def analyze_strategy(
        self,
        final_risk_score: float,
        six_dim_details: Dict[str, Any],
        score_decision_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self.llm or not self.llm.enabled:
            result = self._fallback(final_risk_score, six_dim_details)
            result["needs_deep_review"] = self._needs_deep_review(final_risk_score)
            result["final_decision"] = self._final_decision(
                final_risk_score, result, six_dim_details
            )
            return result

        prompt = STRATEGY_PROMPT.format(
            final_risk_score=final_risk_score,
            threshold=self.threshold,
            lower=self.threshold - self.margin,
            upper=self.threshold + self.margin,
            details_text=json.dumps(six_dim_details, ensure_ascii=False, indent=2),
            score_result_text=json.dumps(
                score_decision_result, ensure_ascii=False, indent=2
            ),
        )
        result: Optional[Dict[str, Any]] = None
        for _ in range(max(1, self.max_retries)):
            obj = self.llm.chat_json(
                prompt, temperature=self.temperature, max_tokens=self.max_tokens
            )
            if isinstance(obj, dict) and "recommendation" in obj:
                result = obj
                break
            print("[StrategyAnalysisAgent][WARNING] 提取 JSON 失败，重试 ...")

        if result is None:
            result = self._fallback(final_risk_score, six_dim_details)

        result["needs_deep_review"] = self._needs_deep_review(final_risk_score)
        result["final_decision"] = self._final_decision(
            final_risk_score, result, six_dim_details
        )
        return result

    # ---- helpers ----
    def _needs_deep_review(self, s: float) -> bool:
        return (self.threshold - self.margin) <= s <= (self.threshold + self.margin)

    def _final_decision(
        self,
        s: float,
        strategy_result: Dict[str, Any],
        six_dim_details: Dict[str, Any],
    ) -> str:
        if strategy_result.get("hard_stop_triggered", False):
            return "拒绝"
        fraud = six_dim_details.get("欺诈风险") or {}
        try:
            if float(fraud.get("评分", 5)) <= 2:
                return "拒绝"
        except (TypeError, ValueError):
            pass
        if s >= self.threshold + self.margin:
            return "通过"
        if s <= self.threshold - self.margin:
            return "拒绝"
        rec = strategy_result.get("recommendation", "深度审查")
        return rec if rec in {"通过", "拒绝"} else "深度审查"

    def _fallback(
        self, final_risk_score: float, six_dim_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "strategy_assessment": "LLM 未启用，使用规则回退。",
            "risk_level": (
                "低风险"
                if final_risk_score >= self.threshold + self.margin
                else "高风险"
                if final_risk_score <= self.threshold - self.margin
                else "中风险"
            ),
            "dimensions_to_review": [],
            "review_reasons": [],
            "hard_stop_triggered": False,
            "hard_stop_reason": "",
            "recommendation": "深度审查"
            if self._needs_deep_review(final_risk_score)
            else ("通过" if final_risk_score >= self.threshold else "拒绝"),
        }


__all__ = ["StrategyAnalysisAgent", "STRATEGY_PROMPT"]

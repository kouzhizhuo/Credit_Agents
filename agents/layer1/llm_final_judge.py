"""Layer 1 · LLMFinalJudgeAgent (新)

在细粒度选择之后、最终导出之前，调用 LLM 对入围的候选特征做**业务层最终判断**：
- 每个特征给出 ``verdict ∈ {keep, warn, drop}`` + 分数 (0-10) + 简短理由；
- ``drop`` 且分数低于阈值 → 从最终结果剔除；
- ``warn`` → 保留但在审计链里标注；
- ``keep`` → 正常保留。

对应技术报告 Eq. 9 的 *LLM-derived contextual relevance* 在**最终决策口**的落地，
确保"黑天鹅"/合规敏感/潜在标签泄漏等仅靠统计无法发现的问题由专家语义网兜住。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...core.llm_client import LLMClient
from ..base import BaseAgent
from .schema import FeatureMetadata


LLM_FINAL_JUDGE_SYSTEM = (
    "你是一位资深信贷风控专家，负责对候选特征做最终业务层审查。"
    " 你需要判断每个特征是否真的适合进入下游多智能体推理与评分模型，"
    " 重点关注：业务相关性、可解释性、合规风险、潜在标签泄漏、异常稀疏度等。"
    " 严格只输出一个 JSON 对象，不要任何解释性文字。"
)

LLM_FINAL_JUDGE_PROMPT = """### 任务

以下是经过统计筛选 (Pearson + XGBoost + Rel_llm 混合分) 得到的候选特征。
请你针对每个特征给出最终业务判断。

### 判断字段定义（每个特征都要给）
- ``verdict``: "keep" | "warn" | "drop"
  * keep: 业务含义清晰、与信贷风险高度相关、可审计
  * warn: 可保留但存在轻度担忧 (如稀疏、含义不够直观、与目标弱相关)
  * drop: 建议剔除 (如 ID/操作字段/数据泄漏/无意义派生)
- ``score``: 0-10 的整数 (越高越推荐保留)
- ``reason``: <= 40 字的简要理由

### 候选特征 (name: description)
{feature_list}

### 输出严格 JSON (键=特征名, 与上面列表完全一致)
```json
{{
  "特征名1": {{"verdict": "keep", "score": 8, "reason": "..."}},
  "特征名2": {{"verdict": "drop", "score": 1, "reason": "..."}}
}}
```
"""


@dataclass
class FinalJudgeResult:
    """LLM 最终判断结果。"""

    kept_features: List[str] = field(default_factory=list)
    dropped_features: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    per_feature: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kept_features": list(self.kept_features),
            "dropped_features": list(self.dropped_features),
            "warnings": list(self.warnings),
            "per_feature": dict(self.per_feature),
        }


class LLMFinalJudgeAgent(BaseAgent):
    """Layer 1 LLM 最终业务判断 Agent。"""

    name = "layer1.llm_final_judge"

    def __init__(
        self,
        llm: LLMClient,
        batch_size: int = 25,
        drop_threshold: float = 3.0,
        verbose: bool = True,
    ):
        self.llm = llm
        self.batch_size = batch_size
        self.drop_threshold = drop_threshold
        self.verbose = verbose

    # ---- API ----
    def run(
        self,
        feature_names: List[str],
        metadata_list: Optional[List[FeatureMetadata]] = None,
    ) -> FinalJudgeResult:
        return self.judge(feature_names, metadata_list)

    def judge(
        self,
        feature_names: List[str],
        metadata_list: Optional[List[FeatureMetadata]] = None,
    ) -> FinalJudgeResult:
        result = FinalJudgeResult()
        if not feature_names:
            return result
        if not self.llm or not self.llm.enabled:
            # 无 LLM 时默认全部保留
            result.kept_features = list(feature_names)
            result.warnings.append("LLM 未启用，跳过最终业务判断，默认全部保留")
            self._log("[LLMFinalJudge] LLM 未启用，全部保留")
            return result

        meta_dict = {m.name: m for m in (metadata_list or [])}
        total_batches = (len(feature_names) + self.batch_size - 1) // self.batch_size
        self._log(
            f"[LLMFinalJudge] 启动: {len(feature_names)} 个特征, 分 {total_batches} 批, "
            f"drop_threshold={self.drop_threshold}"
        )

        for bi, start in enumerate(range(0, len(feature_names), self.batch_size), 1):
            batch = feature_names[start : start + self.batch_size]
            lines = []
            for name in batch:
                desc = (meta_dict.get(name).description if meta_dict.get(name) else "") or name
                lines.append(f"- {name}: {desc}")
            prompt = LLM_FINAL_JUDGE_PROMPT.format(feature_list="\n".join(lines))
            obj = self.llm.chat_json(
                prompt, system=LLM_FINAL_JUDGE_SYSTEM, temperature=0.2, max_tokens=2048
            )
            if not isinstance(obj, dict):
                self._log(f"  [WARNING] 批次 {bi}/{total_batches} 解析失败，整批保留")
                for name in batch:
                    result.per_feature[name] = {
                        "verdict": "keep",
                        "score": 5,
                        "reason": "LLM 解析失败，默认保留",
                    }
                continue
            self._apply_batch(batch, obj, result)
            self._log(f"  批次 {bi}/{total_batches} 完成")

        # 根据 per_feature 分组汇总
        for name in feature_names:
            info = result.per_feature.get(name)
            if not info:
                info = {"verdict": "keep", "score": 5, "reason": "缺失 LLM 判断，默认保留"}
                result.per_feature[name] = info

            verdict = str(info.get("verdict", "keep")).lower()
            try:
                score = float(info.get("score", 5))
            except (TypeError, ValueError):
                score = 5.0

            if verdict == "drop" and score <= self.drop_threshold:
                result.dropped_features.append(name)
            elif verdict == "warn":
                result.kept_features.append(name)
                result.warnings.append(f"{name}: {info.get('reason', '')}")
            else:
                result.kept_features.append(name)

        self._log(
            f"[LLMFinalJudge] 完成: keep={len(result.kept_features)} "
            f"| drop={len(result.dropped_features)} | warn={len(result.warnings)}"
        )
        return result

    # ---- helpers ----
    @staticmethod
    def _apply_batch(
        batch: List[str],
        obj: Dict[str, Any],
        result: FinalJudgeResult,
    ) -> None:
        for name in batch:
            entry = obj.get(name)
            if not isinstance(entry, dict):
                # 兼容 LLM 只返回分数而非对象
                try:
                    score = float(entry)
                    entry = {
                        "verdict": "keep" if score >= 5 else "drop",
                        "score": score,
                        "reason": "只返回了分数",
                    }
                except (TypeError, ValueError):
                    entry = {"verdict": "keep", "score": 5, "reason": "缺失判断"}
            # 规范化
            verdict = str(entry.get("verdict", "keep")).lower()
            if verdict not in ("keep", "warn", "drop"):
                verdict = "keep"
            try:
                score = float(entry.get("score", 5))
            except (TypeError, ValueError):
                score = 5.0
            reason = str(entry.get("reason", "")).strip()[:80]
            result.per_feature[name] = {
                "verdict": verdict,
                "score": score,
                "reason": reason,
            }

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


__all__ = ["LLMFinalJudgeAgent", "FinalJudgeResult", "LLM_FINAL_JUDGE_PROMPT"]

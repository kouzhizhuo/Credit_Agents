"""Layer 1 · FinalValidationAgent

将最终选定的特征集导出为 CSV + 汇总日志 JSON（供审计链使用）。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from ...core.utils import ensure_dir, to_json_serializable
from ..base import BaseAgent
from .llm_final_judge import FinalJudgeResult
from .schema import FeatureMetadata


class FinalValidationAgent(BaseAgent):
    """导出最终选中的特征 + 审计日志。"""

    name = "layer1.final_validation"

    def __init__(self, output_dir: str = "./outputs", verbose: bool = True):
        self.output_dir = output_dir
        self.verbose = verbose
        ensure_dir(output_dir)

    # ---- API ----
    def run(
        self,
        selected_features: List[str],
        metadata_list: List[FeatureMetadata],
        cleaning_log: Dict[str, Any],
        selection_log: Dict[str, Any],
        *,
        screening_log: Optional[List[Any]] = None,
        final_judge: Optional[FinalJudgeResult] = None,
        output_filename: str = "selected_100_features.csv",
    ) -> str:
        return self.validate_and_export(
            selected_features,
            metadata_list,
            cleaning_log,
            selection_log,
            screening_log=screening_log,
            final_judge=final_judge,
            output_filename=output_filename,
        )

    def validate_and_export(
        self,
        selected_features: List[str],
        metadata_list: List[FeatureMetadata],
        cleaning_log: Dict[str, Any],
        selection_log: Dict[str, Any],
        *,
        screening_log: Optional[List[Any]] = None,
        final_judge: Optional[FinalJudgeResult] = None,
        output_filename: str = "selected_100_features.csv",
    ) -> str:
        # 1) CSV 导出
        meta = {m.name: m for m in metadata_list}
        rows: List[Dict[str, Any]] = []
        for feat in selected_features:
            m = meta.get(feat)
            if m is None:
                rows.append(
                    {
                        "feature_name": str(feat),
                        "dtype": "unknown",
                        "missing_rate": 0.0,
                        "unique_count": 0,
                        "description": "",
                        "is_numeric": True,
                        "is_categorical": False,
                    }
                )
            else:
                rows.append(
                    {
                        "feature_name": m.name,
                        "dtype": str(m.dtype),
                        "missing_rate": float(m.missing_rate or 0.0),
                        "unique_count": int(m.unique_count or 0),
                        "description": m.description or "",
                        "is_numeric": bool(m.is_numeric),
                        "is_categorical": bool(m.is_categorical),
                    }
                )
        # 把 LLM Final Judge 的结论拼进 CSV 便于专家核对
        if final_judge is not None:
            for r in rows:
                info = final_judge.per_feature.get(r["feature_name"])
                if info:
                    r["final_verdict"] = info.get("verdict", "keep")
                    r["final_score"] = info.get("score", 5)
                    r["final_reason"] = info.get("reason", "")

        csv_path: Optional[str] = None
        try:
            csv_path = os.path.join(self.output_dir, output_filename)
            pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
            self._log(f"[FinalValidation] CSV 已导出: {csv_path}")
        except Exception as e:
            print(f"[FinalValidation][ERROR] CSV 导出失败: {e}")

        # 2) JSON 日志
        log_path = os.path.join(self.output_dir, "feature_selection_log.json")
        payload = to_json_serializable(
            {
                "selected_features": [str(f) for f in selected_features],
                "feature_count": int(len(selected_features)),
                "cleaning_log": cleaning_log,
                "screening_log": screening_log or [],
                "selection_log": selection_log,
                "final_judge": final_judge.to_dict() if final_judge else None,
                "feature_summary": rows,
            }
        )
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self._log(f"[FinalValidation] JSON 日志已保存: {log_path}")
        except Exception as e:
            print(f"[FinalValidation][ERROR] JSON 写入失败: {e}")

        return csv_path or log_path

    # ---- helpers ----
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


__all__ = ["FinalValidationAgent"]

"""Layer 1 · FineGrainedSelectionAgent

迭代式细粒度特征选择：
- Pearson 相关性剔除最低 ~10%；
- XGBoost 重要性 Gain + LLM 语义相关性 Rel_llm → 混合分 S(f)=λ·Gain+(1-λ)·Rel_llm；
- 每轮用 5 折交叉验证记录性能。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold, cross_val_score

from ...core.llm_client import LLMClient
from ..base import BaseAgent
from .llm_relevance import LLMRelevanceAgent
from .schema import FeatureMetadata


class FineGrainedSelectionAgent(BaseAgent):
    """细粒度选择 Agent。"""

    name = "layer1.fine_selection"

    def __init__(
        self,
        target_col: str,
        target_features: int = 100,
        prune_ratio: float = 0.1,
        cv_folds: int = 5,
        lambda_weight: float = 0.7,
        llm: Optional[LLMClient] = None,
        verbose: bool = True,
    ):
        self.target_col = target_col
        self.target_features = target_features
        self.prune_ratio = prune_ratio
        self.cv_folds = cv_folds
        self.lambda_weight = lambda_weight
        self.llm = llm
        self.verbose = verbose
        self.iteration_logs: List[Dict[str, Any]] = []

    # ---- API ----
    def run(
        self,
        df: pd.DataFrame,
        feature_list: List[str],
        metadata_list: List[FeatureMetadata],
    ) -> Tuple[List[str], Dict[str, Any]]:
        return self.select(df, feature_list, metadata_list)

    # ---- core ----
    def select(
        self,
        df: pd.DataFrame,
        feature_list: List[str],
        metadata_list: List[FeatureMetadata],
    ) -> Tuple[List[str], Dict[str, Any]]:
        current = [f for f in feature_list if f != self.target_col]
        initial_count = len(current)
        X = df[current].copy()
        y = df[self.target_col].copy()
        if y.dtype == "object" or y.dtype.name == "category":
            y = pd.Categorical(y).codes

        self._log("[细粒度选择] 启动")
        self._log(
            f"  输入 {initial_count} → 目标 {self.target_features}, "
            f"prune_ratio={self.prune_ratio}, λ={self.lambda_weight}, cv={self.cv_folds}, "
            f"LLM={'是' if (self.llm and self.llm.enabled) else '否'}"
        )

        iteration = 0
        while len(current) > self.target_features:
            iteration += 1
            n_before = len(current)
            self._log(f"\n[迭代 {iteration}] 当前 {n_before}")

            # 1) Pearson
            corr_drop = self._remove_redundant_features(
                X[current], y, int(n_before * self.prune_ratio)
            )
            after_corr = [f for f in current if f not in corr_drop]
            if len(after_corr) < self.target_features:
                self._log("  相关性剔除后不足目标数，提前截断")
                after_corr = current[: self.target_features]
                break

            # 2) 混合分剔除
            imp_drop = self._remove_low_importance_features(
                X[after_corr], y, int(len(after_corr) * self.prune_ratio),
                [m for m in metadata_list if m.name in after_corr],
            )
            current = [f for f in after_corr if f not in imp_drop]
            n_after = len(current)

            # 3) CV
            cv_score = self._evaluate_features(X[current], y) if current else 0.0
            self.iteration_logs.append(
                {
                    "iteration": iteration,
                    "features_before": n_before,
                    "features_after_corr": len(after_corr),
                    "features_after": n_after,
                    "removed_by_correlation": len(corr_drop),
                    "removed_by_importance": len(imp_drop),
                    "cv_score": float(cv_score) if not pd.isna(cv_score) else 0.0,
                    "removed_corr_all": [str(f) for f in corr_drop],
                    "removed_importance_all": [str(f) for f in imp_drop],
                }
            )
            self._log(
                f"  特征数: {n_before} → {len(after_corr)} → {n_after}; "
                f"CV={cv_score:.4f}"
            )

        # 末端截断
        if len(current) > self.target_features:
            hybrid = self._get_hybrid_scores(
                X[current], y, current,
                [m for m in metadata_list if m.name in current],
            )
            current = [
                f
                for f, _ in sorted(hybrid.items(), key=lambda x: x[1], reverse=True)[
                    : self.target_features
                ]
            ]

        cv_final = self._evaluate_features(X[current], y) if current else 0.0
        self._log(f"[细粒度选择] 完成, 最终 {len(current)} 个, CV={cv_final:.4f}")

        return current, {
            "iterations": int(iteration),
            "final_feature_count": int(len(current)),
            "initial_feature_count": int(initial_count),
            "final_cv_score": float(cv_final) if not pd.isna(cv_final) else 0.0,
            "iteration_logs": self.iteration_logs,
        }

    # ---- steps ----
    def _remove_redundant_features(
        self, X: pd.DataFrame, y: pd.Series, num_to_remove: int
    ) -> List[str]:
        if num_to_remove <= 0 or len(X.columns) <= num_to_remove:
            return []
        corr: Dict[str, float] = {}
        for col in X.columns:
            try:
                if pd.api.types.is_numeric_dtype(X[col]):
                    r, _ = pearsonr(X[col].fillna(0), y)
                    corr[col] = float(abs(r)) if not np.isnan(r) else 0.0
                else:
                    enc = pd.get_dummies(X[col], prefix=col)
                    vals = []
                    for c in enc.columns:
                        r, _ = pearsonr(enc[c].fillna(0), y)
                        if not np.isnan(r):
                            vals.append(abs(r))
                    corr[col] = float(max(vals) if vals else 0.0)
            except Exception:
                corr[col] = 0.0
        return [f for f, _ in sorted(corr.items(), key=lambda x: x[1])[:num_to_remove]]

    def _remove_low_importance_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_to_remove: int,
        metadata_list: Optional[List[FeatureMetadata]] = None,
    ) -> List[str]:
        if num_to_remove <= 0 or len(X.columns) <= num_to_remove:
            return []
        hybrid = self._get_hybrid_scores(X, y, list(X.columns), metadata_list)
        return [
            f for f, _ in sorted(hybrid.items(), key=lambda x: x[1])[:num_to_remove]
        ]

    def _get_hybrid_scores(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str],
        metadata_list: Optional[List[FeatureMetadata]] = None,
    ) -> Dict[str, float]:
        gain_raw = self._compute_feature_importance(X[feature_names], y)
        if not gain_raw:
            return {f: 0.5 for f in feature_names}
        lo, hi = min(gain_raw.values()), max(gain_raw.values())
        span = (hi - lo) or 1.0
        gain_norm = {f: (gain_raw.get(f, 0) - lo) / span for f in feature_names}

        if self.llm and self.llm.enabled:
            try:
                rel = LLMRelevanceAgent(self.llm, verbose=self.verbose).score_features(
                    feature_names, metadata_list
                )
                return {
                    f: self.lambda_weight * gain_norm.get(f, 0)
                    + (1 - self.lambda_weight) * rel.get(f, 0.5)
                    for f in feature_names
                }
            except Exception as e:
                self._log(f"[WARNING] Rel_llm 失败, 仅用 Gain: {e}")
        return gain_norm

    def _compute_feature_importance(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        import xgboost as xgb

        X_proc = X.copy()
        for col in X_proc.columns:
            if not pd.api.types.is_numeric_dtype(X_proc[col]):
                X_proc[col] = pd.Categorical(X_proc[col]).codes
        try:
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric="logloss",
                use_label_encoder=False,
            )
            model.fit(X_proc.fillna(0), y)
            return dict(zip(X_proc.columns, model.feature_importances_))
        except Exception as e:
            self._log(f"  [WARNING] XGBoost 训练失败: {e}，使用均匀重要性")
            return {c: 1.0 for c in X.columns}

    def _evaluate_features(self, X: pd.DataFrame, y: pd.Series) -> float:
        import xgboost as xgb

        if X.shape[1] == 0:
            return 0.0
        X_proc = X.copy()
        for col in X_proc.columns:
            if not pd.api.types.is_numeric_dtype(X_proc[col]):
                X_proc[col] = pd.Categorical(X_proc[col]).codes
        try:
            model = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                eval_metric="logloss",
                use_label_encoder=False,
            )
            scoring = "roc_auc" if len(np.unique(y)) == 2 else "f1_macro"
            scores = cross_val_score(
                model,
                X_proc.fillna(0),
                y,
                cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42),
                scoring=scoring,
            )
            return float(scores.mean())
        except Exception as e:
            self._log(f"  [WARNING] 交叉验证失败: {e}")
            return 0.0

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


__all__ = ["FineGrainedSelectionAgent"]

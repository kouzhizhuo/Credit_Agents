"""Layer 1 · DataCleaningAgent

处理缺失值、异常值、类型转换。对应技术报告 Layer 1 (Eq. 1) 前置清洗阶段。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..base import BaseAgent
from .schema import CleaningLog


class DataCleaningAgent(BaseAgent):
    """数据清理 Agent。

    Parameters
    ----------
    missing_threshold:
        单列缺失率阈值，超过则整列剔除。
    outlier_method:
        ``"iqr"`` 或 ``"zscore"``，仅对数值列裁剪。
    """

    name = "layer1.data_cleaning"

    def __init__(
        self,
        missing_threshold: float = 0.95,
        outlier_method: str = "iqr",
        verbose: bool = True,
    ):
        self.missing_threshold = missing_threshold
        self.outlier_method = outlier_method
        self.verbose = verbose
        self.cleaning_log = CleaningLog()

    # ---- API ----
    def run(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        return self.clean(df, target_col=target_col)

    def clean(
        self, df: pd.DataFrame, target_col: Optional[str] = None
    ) -> pd.DataFrame:
        self._log(f"[INFO] 数据清理开始: 原始形状 {df.shape}")
        df_clean = df.copy()

        # 1) 高缺失率列
        missing_rates = df_clean.isnull().sum() / len(df_clean)
        high_missing_cols = missing_rates[
            missing_rates > self.missing_threshold
        ].index.tolist()
        if target_col and target_col in high_missing_cols:
            high_missing_cols.remove(target_col)
        df_clean = df_clean.drop(columns=high_missing_cols)
        self.cleaning_log.removed_columns.extend(high_missing_cols)
        self.cleaning_log.missing_values_filled = len(high_missing_cols)
        if high_missing_cols:
            self._log(f"  移除 {len(high_missing_cols)} 个高缺失率列")

        # 2) 数值列缺失值 → 中位数
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        if target_col and target_col in numeric_cols:
            numeric_cols = numeric_cols.drop(target_col)
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].median(), inplace=True)

        # 3) 分类列缺失值 → 'MISSING'
        categorical_cols = df_clean.select_dtypes(
            include=["object", "category"]
        ).columns
        if target_col and target_col in categorical_cols:
            categorical_cols = categorical_cols.drop(target_col)
        for col in categorical_cols:
            if df_clean[col].isnull().any():
                df_clean[col].fillna("MISSING", inplace=True)

        # 4) 异常值裁剪
        outlier_total = 0
        for col in numeric_cols:
            series = df_clean[col]
            if self.outlier_method == "iqr":
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                cnt = int(((series < lo) | (series > hi)).sum())
                if cnt > 0:
                    df_clean[col] = series.clip(lower=lo, upper=hi)
                    outlier_total += cnt
            elif self.outlier_method == "zscore":
                mean, std = series.mean(), series.std()
                if std and not np.isnan(std):
                    z = np.abs((series - mean) / std)
                    cnt = int((z > 3).sum())
                    if cnt > 0:
                        df_clean[col] = series.clip(
                            lower=series.quantile(0.01),
                            upper=series.quantile(0.99),
                        )
                        outlier_total += cnt
        self.cleaning_log.outliers_removed += outlier_total
        if outlier_total:
            self._log(f"  处理了 {outlier_total} 个异常值")

        # 5) 数值字符串 → 数值
        for col in df_clean.columns:
            if col == target_col or df_clean[col].dtype != "object":
                continue
            try:
                numeric_series = pd.to_numeric(df_clean[col], errors="coerce")
                if numeric_series.notna().sum() / len(df_clean) > 0.8:
                    df_clean[col] = numeric_series
                    self.cleaning_log.type_conversions.append(
                        f"{col}: object -> numeric"
                    )
            except Exception:
                pass

        self._log(
            f"[INFO] 数据清理完成: 清理后形状 {df_clean.shape} "
            f"(移除 {len(high_missing_cols)} 列)"
        )
        return df_clean

    # ---- log ----
    def get_log(self) -> Dict[str, Any]:
        return {
            "missing_values_filled": int(self.cleaning_log.missing_values_filled),
            "outliers_removed": int(self.cleaning_log.outliers_removed),
            "type_conversions": list(self.cleaning_log.type_conversions),
            "format_fixes": list(self.cleaning_log.format_fixes),
            "removed_columns": list(self.cleaning_log.removed_columns),
        }

    # ---- helpers ----
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


__all__ = ["DataCleaningAgent"]

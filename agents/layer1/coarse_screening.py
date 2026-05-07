"""Layer 1 · CoarseScreeningAgent

剔除 ID/哈希、常量、重复列等无信息列。
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

import pandas as pd

from ..base import BaseAgent
from .schema import FeatureMetadata


class CoarseScreeningAgent(BaseAgent):
    """粗筛选 Agent。"""

    name = "layer1.coarse_screening"

    ID_PATTERNS = (
        r"^id$", r"^idx$", r"^index$", r"^row_id$",
        r".*_id$", r".*_hash$", r".*_uuid$",
        r"^编号$", r"^索引$", r".*编号$",
    )

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.removed_features: List[Tuple[str, str]] = []

    # ---- API ----
    def run(
        self,
        df: pd.DataFrame,
        metadata_list: List[FeatureMetadata],
        target_col: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        return self.screen(df, metadata_list, target_col=target_col)

    def screen(
        self,
        df: pd.DataFrame,
        metadata_list: List[FeatureMetadata],
        target_col: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        self._log(f"[INFO] 粗筛选开始: 输入特征数 {len(metadata_list)}")
        remove: set = set()

        for meta in metadata_list:
            if meta.name == target_col:
                continue
            # ID / Hash
            if any(re.match(p, meta.name, re.IGNORECASE) for p in self.ID_PATTERNS):
                remove.add(meta.name)
                self.removed_features.append((meta.name, "ID/哈希字段"))
                continue
            # 常量列
            if meta.unique_count <= 1:
                remove.add(meta.name)
                self.removed_features.append((meta.name, "常量列"))
                continue
            # 近乎唯一列
            if len(df) > 0 and meta.unique_count / len(df) > 0.99:
                remove.add(meta.name)
                self.removed_features.append((meta.name, "高唯一值列（疑似哈希）"))

        # 重复列
        for dup in self._find_duplicate_columns(df, remove, target_col):
            remove.add(dup)
            self.removed_features.append((dup, "重复列"))

        remaining = [c for c in df.columns if c not in remove]
        if target_col and target_col not in remaining:
            remaining.append(target_col)
        df_screened = df[remaining].copy()
        self._log(
            f"[INFO] 粗筛选完成: 移除 {len(remove)} 个, 保留 {len(remaining)} 个"
        )
        return df_screened, remaining

    def get_removed_features(self) -> List[Tuple[str, str]]:
        return [(str(f), str(r)) for f, r in self.removed_features]

    # ---- helpers ----
    @staticmethod
    def _find_duplicate_columns(
        df: pd.DataFrame, exclude: set, target_col: Optional[str]
    ) -> List[str]:
        dup: List[str] = []
        checked: set = set()
        cols = list(df.columns)
        for i, c1 in enumerate(cols):
            if c1 in exclude or c1 == target_col or c1 in checked:
                continue
            for c2 in cols[i + 1 :]:
                if c2 in exclude or c2 == target_col or c2 in checked:
                    continue
                if df[c1].equals(df[c2]):
                    dup.append(c2)
                    checked.add(c2)
        return dup

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


__all__ = ["CoarseScreeningAgent"]

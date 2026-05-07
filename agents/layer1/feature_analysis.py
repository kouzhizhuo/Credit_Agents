"""Layer 1 · FeatureAnalysisAgent

为每个特征生成元数据 + 业务描述：字典优先 → LLM → 启发式兜底。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...core.llm_client import LLMClient
from ..base import BaseAgent
from .schema import FeatureMetadata


class FeatureAnalysisAgent(BaseAgent):
    """特征分析 Agent。

    Parameters
    ----------
    llm: 可选 ``LLMClient``。无 LLM 时走字典/启发式。
    dict_file: 特征字典 xlsx 路径（默认尝试 ``./dxm_dict.xlsx``）。
    """

    name = "layer1.feature_analysis"

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        dict_file: Optional[str] = None,
        verbose: bool = True,
    ):
        self.llm = llm
        self.dict_file = dict_file
        self.verbose = verbose
        self.feature_dict: Dict[str, str] = {}
        self._load_feature_dict()

    # ---- API ----
    def run(
        self, df: pd.DataFrame, target_col: Optional[str] = None
    ) -> List[FeatureMetadata]:
        return self.analyze(df, target_col=target_col)

    def analyze(
        self, df: pd.DataFrame, target_col: Optional[str] = None
    ) -> List[FeatureMetadata]:
        feature_cols = [c for c in df.columns if c != target_col]
        total = len(feature_cols)
        self._log(f"[INFO] 开始分析 {total} 个特征 ...")

        out: List[FeatureMetadata] = []
        dict_hits = llm_calls = heuristic_falls = 0
        print_interval = max(1, min(20, total // 10))

        for idx, col in enumerate(feature_cols, 1):
            if idx % print_interval == 0 or idx in (1, total):
                self._log(
                    f"  [进度] {idx}/{total} "
                    f"- 字典:{dict_hits} LLM:{llm_calls} 启发式:{heuristic_falls} - 当前: {col}"
                )
            dtype = str(df[col].dtype)
            missing_rate = float(df[col].isnull().sum() / len(df))
            unique_count = int(df[col].nunique())
            example_values = self._to_py_examples(df[col].dropna().head(5).tolist())
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            is_categorical = not is_numeric and unique_count < len(df) * 0.5

            description, source = self._generate_description(col, df[col], is_numeric)
            if source == "dict":
                dict_hits += 1
            elif source == "llm":
                llm_calls += 1
            else:
                heuristic_falls += 1

            out.append(
                FeatureMetadata(
                    name=col,
                    dtype=dtype,
                    missing_rate=missing_rate,
                    unique_count=unique_count,
                    example_values=example_values,
                    description=description,
                    is_numeric=is_numeric,
                    is_categorical=is_categorical,
                )
            )

        self._log(
            f"[INFO] 特征分析完成: 字典 {dict_hits} | LLM {llm_calls} | 启发式 {heuristic_falls}"
        )
        return out

    # ---- dict loader ----
    def _load_feature_dict(self) -> None:
        path = self.dict_file or os.path.join(os.getcwd(), "dxm_dict.xlsx")
        self.dict_file = path
        if not os.path.isfile(path):
            self._log(f"[INFO] 特征字典文件不存在: {path}，将使用 LLM 或启发式方法")
            return
        try:
            import openpyxl  # noqa
            df_dict = pd.read_excel(path)
        except ImportError:
            self._log("[WARNING] 需要安装 openpyxl 来读取 xlsx 文件")
            return
        except Exception as e:
            self._log(f"[WARNING] 加载特征字典失败: {e}")
            return

        feat_col = desc_col = None
        for col in df_dict.columns:
            lo = str(col).lower()
            if any(k in lo for k in ("字段", "field", "特征", "feature", "name", "名称", "列名", "column")):
                feat_col = feat_col or col
            elif any(k in lo for k in ("描述", "desc", "说明", "解释", "definition", "定义", "含义", "meaning")):
                desc_col = desc_col or col
        if feat_col is None:
            feat_col = df_dict.columns[0]
        if desc_col is None and len(df_dict.columns) > 1:
            desc_col = df_dict.columns[1]
        if feat_col is None or desc_col is None:
            self._log("[WARNING] 无法识别特征字典的列结构")
            return
        for _, row in df_dict.iterrows():
            name = str(row[feat_col]).strip()
            desc = str(row[desc_col]).strip() if pd.notna(row[desc_col]) else ""
            if name and desc and desc.lower() not in ("nan", "none", ""):
                self.feature_dict[name] = desc
        self._log(f"[INFO] 已加载特征字典: {len(self.feature_dict)} 条")

    # ---- description ----
    def _generate_description(
        self, col: str, series: pd.Series, is_numeric: bool
    ) -> Tuple[str, str]:
        if col in self.feature_dict:
            return self.feature_dict[col], "dict"
        if self.llm and self.llm.enabled:
            try:
                stats = {
                    "mean": float(series.mean()) if is_numeric else None,
                    "std": float(series.std()) if is_numeric else None,
                    "min": float(series.min()) if is_numeric else None,
                    "max": float(series.max()) if is_numeric else None,
                    "unique_count": int(series.nunique()),
                    "sample_values": self._to_py_examples(
                        series.dropna().head(3).tolist()
                    ),
                }
                prompt = (
                    "请为以下特征生成一句简洁的业务描述（不超过 50 字），用于信贷风控特征分析。\n\n"
                    f"特征名：{col}\n数据类型：{'数值型' if is_numeric else '分类型'}\n"
                    f"统计信息：{json.dumps(stats, ensure_ascii=False)}\n\n"
                    "只输出这一句描述，不要其他内容。"
                )
                desc = self.llm.chat(prompt, temperature=0.3, max_tokens=100)
                if desc:
                    return desc.strip(), "llm"
            except Exception as e:
                print(f"[WARNING] LLM 描述生成失败 ({col}): {e}")

        lower = col.lower()
        if "age" in lower or "年龄" in col:
            return "年龄信息", "heuristic"
        if "income" in lower or "收入" in col:
            return "收入信息", "heuristic"
        if "debt" in lower or "负债" in col:
            return "负债信息", "heuristic"
        if "credit" in lower or "信用" in col:
            return "信用相关信息", "heuristic"
        if "id" in lower or "编号" in col:
            return "标识符", "heuristic"
        return f"{col}特征", "heuristic"

    # ---- helpers ----
    @staticmethod
    def _to_py_examples(values: List[Any]) -> List[Any]:
        out = []
        for v in values:
            if isinstance(v, (np.integer,)):
                out.append(int(v))
            elif isinstance(v, (np.floating,)):
                f = float(v)
                out.append(None if (np.isnan(f) or np.isinf(f)) else f)
            elif isinstance(v, np.bool_):
                out.append(bool(v))
            elif pd.isna(v):
                out.append(None)
            else:
                out.append(v)
        return out

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


__all__ = ["FeatureAnalysisAgent"]

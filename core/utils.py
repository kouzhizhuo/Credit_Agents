"""core.utils: 通用工具函数（JSON 解析/序列化、目录/随机种子等）。"""
from __future__ import annotations

import json
import os
import random
import re
import string
from typing import Any, Optional


# ---------------------------- 目录 & 随机种子 ---------------------------- #
def ensure_dir(path: str) -> str:
    """确保目录存在，返回绝对路径。"""
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def set_seed(seed: int) -> None:
    """固定 Python / numpy / random 随机种子。"""
    random.seed(seed)
    try:
        import numpy as np  # noqa
        np.random.seed(seed)
    except Exception:
        pass


# ---------------------------- JSON 解析/清洗 ---------------------------- #
_SEED_THINK_TAG = "</seed:think>"


def _clean_and_validate_json(raw_json: str) -> str:
    """去掉非打印字符后验证 JSON 对象边界。"""
    allowed_chars = set(
        string.printable + "，。！？；：“”‘’（）【】《》·～、\n\t "
    )
    cleaned = "".join(c for c in raw_json if c in allowed_chars or ord(c) >= 128)
    if not (cleaned.startswith("{") and cleaned.endswith("}")):
        raise ValueError(
            f"extracted content is not a valid JSON object: {cleaned[:100]}..."
        )
    return cleaned


def extract_first_json_obj(text: str) -> str:
    """从 LLM 输出中抽取第一个 JSON 对象字符串。

    兼容:
    - ``</seed:think>`` 之后的片段 (Seed 系列模型常见)
    - ```json ... ``` 代码块
    - 裸花括号块 ``{...}``
    """
    if not text:
        raise ValueError("empty llm output")

    tag_pos = text.find(_SEED_THINK_TAG)
    search_text = text if tag_pos == -1 else text[tag_pos + len(_SEED_THINK_TAG):]

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", search_text, re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return _clean_and_validate_json(candidate)

    start = search_text.find("{")
    end = search_text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"cannot find json object: {search_text[:200]}...")

    raw_json = search_text[start : end + 1].strip()
    return _clean_and_validate_json(raw_json)


def safe_json_loads(text: str) -> Optional[Any]:
    """容错 JSON 解析：先尝试抽取对象块；失败再直接 loads；都失败返回 None。

    额外尝试修复尾逗号 ``,}`` / ``,]``。
    """
    if not text:
        return None
    try:
        raw = extract_first_json_obj(text)
    except Exception:
        raw = text.strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        fixed = re.sub(r",\s*([}\]])", r"\1", raw)
        try:
            return json.loads(fixed)
        except Exception:
            return None


# ---------------------------- JSON 序列化 ---------------------------- #
def to_json_serializable(obj: Any) -> Any:
    """递归把 numpy / pandas 类型转换成 Python 原生类型，确保 ``json.dumps`` 可用。"""
    try:
        import numpy as np
        import pandas as pd
    except Exception:  # pragma: no cover
        np = None  # type: ignore
        pd = None  # type: ignore

    if obj is None:
        return None

    if pd is not None:
        try:
            if pd.isna(obj):
                return None
        except (TypeError, ValueError):
            pass

    if np is not None:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            val = float(obj)
            if np.isnan(val) or np.isinf(val):
                return None
            return val
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return [to_json_serializable(x) for x in obj.tolist()]

    if pd is not None:
        if isinstance(obj, pd.Series):
            return [to_json_serializable(x) for x in obj.tolist()]
        if isinstance(obj, pd.DataFrame):
            return [to_json_serializable(r) for r in obj.to_dict("records")]

    if isinstance(obj, dict):
        return {str(k): to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_serializable(item) for item in obj]
    if isinstance(obj, (bytes,)):
        return obj.decode("utf-8", errors="ignore")
    if isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        import math

        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    try:
        return str(obj)
    except Exception:
        return None


__all__ = [
    "ensure_dir",
    "set_seed",
    "extract_first_json_obj",
    "safe_json_loads",
    "to_json_serializable",
]

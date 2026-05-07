"""Layer 1 中的轻量 dataclass 与常量。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class CleaningLog:
    missing_values_filled: int = 0
    outliers_removed: int = 0
    type_conversions: List[str] = field(default_factory=list)
    format_fixes: List[str] = field(default_factory=list)
    removed_columns: List[str] = field(default_factory=list)


@dataclass
class FeatureMetadata:
    name: str
    dtype: str
    missing_rate: float
    unique_count: int
    example_values: List[Any]
    description: str = ""
    is_numeric: bool = True
    is_categorical: bool = False


__all__ = ["CleaningLog", "FeatureMetadata"]

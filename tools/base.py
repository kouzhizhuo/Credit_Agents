"""tools.base: 工具基类与标准返回类型。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ToolResult:
    """工具执行结果。始终字符串化，避免把复杂对象塞进 LLM。"""

    content: str
    success: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_observation(self) -> str:
        return self.content


class Tool:
    """工具基类。子类需要实现 ``run()`` 并定义 ``name`` / ``description`` / ``schema``。"""

    name: str = "tool"
    description: str = ""
    # OpenAI Function-Calling 兼容的 JSON Schema（简化版）
    schema: Dict[str, Any] = {}

    def run(self, **kwargs: Any) -> ToolResult:  # pragma: no cover - abstract
        raise NotImplementedError

    # 面向 LLM 描述自身（用于 prompt 注入）
    def describe(self) -> str:
        required = ", ".join((self.schema or {}).get("required", []))
        fields = ", ".join((self.schema or {}).get("properties", {}).keys())
        return (
            f"- {self.name}: {self.description}\n"
            f"  参数: {{{fields}}} 必填: [{required}]"
        )


__all__ = ["Tool", "ToolResult"]

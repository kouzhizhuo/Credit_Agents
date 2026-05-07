"""tools.registry: 工具注册表 + 默认构造。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .base import Tool, ToolResult


@dataclass
class ToolRegistry:
    """按名字管理一组工具。"""

    tools: Dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        if not tool or not tool.name:
            raise ValueError("Tool must have a non-empty name")
        self.tools[tool.name] = tool

    def has(self, name: str) -> bool:
        return name in self.tools

    def get(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    def names(self) -> List[str]:
        return list(self.tools.keys())

    def describe_all(self) -> str:
        return "\n".join(t.describe() for t in self.tools.values())

    def call(self, name: str, args: Dict[str, Any]) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(
                content=f"[ToolRegistry] 未注册的工具: {name}", success=False
            )
        try:
            return tool.run(**(args or {}))
        except Exception as e:
            return ToolResult(
                content=f"[ToolRegistry] 工具 {name} 调用异常: {e}", success=False
            )


def build_default_tool_registry(
    enabled: Iterable[str] = ("web_search", "document_reader", "knowledge_base"),
    *,
    knowledge_base_dir: str = "./docs",
    web_search_max_results: int = 5,
    document_base_dir: str = ".",
) -> ToolRegistry:
    """按配置构造默认工具集。"""
    registry = ToolRegistry()
    enabled_set = set(enabled)
    if "web_search" in enabled_set:
        from .web_search import WebSearchTool

        registry.register(WebSearchTool(max_results=web_search_max_results))
    if "document_reader" in enabled_set:
        from .document_reader import DocumentReaderTool

        registry.register(DocumentReaderTool(base_dir=document_base_dir))
    if "knowledge_base" in enabled_set:
        from .knowledge_base import KnowledgeBaseTool

        registry.register(KnowledgeBaseTool(kb_dir=knowledge_base_dir))
    return registry


__all__ = ["ToolRegistry", "build_default_tool_registry"]

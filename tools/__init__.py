"""tools: Layer 2 Agent 可调用的外部工具。

统一约定: 每个 Tool 实现 ``name``/``description``/``schema``/``run(**kwargs) -> str``。
无法调用（依赖缺失/离线）时 ``run()`` 返回一段提示字符串而不是抛错，保证上层不崩溃。

当前内置:
- ``WebSearchTool``: DuckDuckGo / 自定义 HTTP 搜索 (最小依赖, 可选禁用)。
- ``DocumentReaderTool``: 读取本地 txt/md/pdf 文档的片段。
- ``KnowledgeBaseTool``: 基于 BM25 的本地文档检索（项目 docs/ 目录）。
"""
from .base import Tool, ToolResult
from .web_search import WebSearchTool
from .document_reader import DocumentReaderTool
from .knowledge_base import KnowledgeBaseTool
from .registry import ToolRegistry, build_default_tool_registry

__all__ = [
    "Tool",
    "ToolResult",
    "WebSearchTool",
    "DocumentReaderTool",
    "KnowledgeBaseTool",
    "ToolRegistry",
    "build_default_tool_registry",
]

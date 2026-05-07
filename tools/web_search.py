"""tools.web_search: 轻量网页搜索工具。

优先使用 ``duckduckgo_search`` (若已安装)；若未安装且本机无网，回退为占位结果，
保证上层 Agent 不会因为工具失败而崩溃。
"""
from __future__ import annotations

from typing import Any, List

from .base import Tool, ToolResult


class WebSearchTool(Tool):
    """通用网页搜索工具，面向 Layer 2 Agent 的"外部信息补强"。"""

    name = "web_search"
    description = "通过搜索引擎获取与查询相关的最近互联网摘要，用于补充征信/政策/新闻等外部证据。"
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词 (中英文均可)"},
            "max_results": {"type": "integer", "description": "最多返回条数, 默认 5"},
        },
        "required": ["query"],
    }

    def __init__(self, max_results: int = 5, region: str = "cn-zh"):
        self.default_max_results = max_results
        self.region = region

    def run(self, **kwargs: Any) -> ToolResult:
        query = str(kwargs.get("query", "")).strip()
        if not query:
            return ToolResult(content="缺少 query 参数", success=False)
        k = int(kwargs.get("max_results", self.default_max_results))

        items = self._ddg_search(query, k)
        if items is None:
            return ToolResult(
                content=(
                    "[WebSearch] 当前环境未安装 duckduckgo_search 或无法联网，"
                    f"查询 `{query}` 未获得结果。建议在 prompt 中结合 dimension_data 自行判断。"
                ),
                success=False,
                meta={"query": query},
            )
        if not items:
            return ToolResult(
                content=f"[WebSearch] `{query}` 没有检索到任何结果。",
                success=True,
                meta={"query": query},
            )
        lines = []
        for i, it in enumerate(items, 1):
            title = it.get("title") or it.get("heading") or ""
            snippet = it.get("body") or it.get("snippet") or ""
            href = it.get("href") or it.get("url") or ""
            lines.append(f"{i}. {title}\n   {snippet}\n   {href}")
        return ToolResult(
            content="\n".join(lines),
            success=True,
            meta={"query": query, "count": len(items)},
        )

    # ---- impl ----
    def _ddg_search(self, query: str, k: int) -> List[dict] | None:
        try:
            from duckduckgo_search import DDGS  # type: ignore
        except Exception:
            return None
        try:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=k, region=self.region))
        except Exception as e:
            print(f"[WebSearchTool] DDG 调用失败: {e}")
            return []


__all__ = ["WebSearchTool"]

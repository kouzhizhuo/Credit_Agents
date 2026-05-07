"""tools.knowledge_base: 本地 docs/ 目录的轻量语义/词频检索。

实现采用 **词频+子串匹配** 的最简兜底方案，不引入向量数据库依赖；
若项目未来想替换为 BM25 / FAISS，只需替换 ``_rank()`` 即可。
"""
from __future__ import annotations

import os
import re
from typing import Any, List, Tuple

from .base import Tool, ToolResult


def _iter_files(root: str, exts: Tuple[str, ...]) -> List[str]:
    out: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in exts:
                out.append(os.path.join(dirpath, f))
    return out


class KnowledgeBaseTool(Tool):
    name = "knowledge_base"
    description = "检索本地知识库 (项目 docs/ 目录下的 txt/md)，返回与 query 最相关的文档片段。"
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "自然语言检索词"},
            "top_k": {"type": "integer", "description": "返回片段数, 默认 3"},
            "snippet_len": {"type": "integer", "description": "每个片段字符数, 默认 400"},
        },
        "required": ["query"],
    }

    SUPPORTED_EXTS: Tuple[str, ...] = (".txt", ".md", ".log")

    def __init__(self, kb_dir: str = "./docs"):
        self.kb_dir = os.path.abspath(kb_dir)
        self._index: List[Tuple[str, str]] = []
        self._load()

    def run(self, **kwargs: Any) -> ToolResult:
        query = str(kwargs.get("query", "")).strip()
        if not query:
            return ToolResult(content="缺少 query 参数", success=False)
        if not self._index:
            return ToolResult(
                content=f"[KnowledgeBase] 目录 {self.kb_dir} 下没有可检索的 txt/md 文档",
                success=False,
            )
        k = int(kwargs.get("top_k", 3))
        snippet_len = int(kwargs.get("snippet_len", 400))
        hits = self._rank(query, top_k=k)
        if not hits:
            return ToolResult(
                content=f"[KnowledgeBase] `{query}` 未检索到匹配内容",
                success=True,
                meta={"query": query},
            )
        sections = []
        for path, score, snippet in hits:
            relpath = os.path.relpath(path, self.kb_dir)
            sections.append(
                f"[{relpath}] (score={score:.2f})\n{snippet[:snippet_len]}\n"
            )
        return ToolResult(
            content="\n---\n".join(sections),
            success=True,
            meta={"query": query, "hits": len(hits)},
        )

    # ---- impl ----
    def _load(self) -> None:
        if not os.path.isdir(self.kb_dir):
            return
        for path in _iter_files(self.kb_dir, self.SUPPORTED_EXTS):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                self._index.append((path, text))
            except Exception:
                continue

    def _rank(
        self, query: str, top_k: int
    ) -> List[Tuple[str, float, str]]:
        tokens = [t for t in re.split(r"\s+|[,，。；:：、/]", query) if t]
        scored: List[Tuple[str, float, str]] = []
        for path, text in self._index:
            score = 0.0
            text_lower = text.lower()
            for t in tokens:
                tl = t.lower()
                score += text_lower.count(tl) * (1.0 + len(tl) / 10.0)
            if score <= 0:
                continue
            # 找到第一个命中位置作为 snippet
            idx = 0
            for t in tokens:
                i = text_lower.find(t.lower())
                if i >= 0:
                    idx = max(0, i - 50)
                    break
            snippet = text[idx : idx + 600]
            scored.append((path, score, snippet))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


__all__ = ["KnowledgeBaseTool"]

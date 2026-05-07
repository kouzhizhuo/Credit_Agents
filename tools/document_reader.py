"""tools.document_reader: 本地文档阅读工具。

支持 ``.txt`` / ``.md`` / ``.pdf``。pdf 读取时优先使用 ``pypdf`` 或 ``PyPDF2``，
若两者都未安装则返回提示信息而不是抛错。
"""
from __future__ import annotations

import os
from typing import Any

from .base import Tool, ToolResult


class DocumentReaderTool(Tool):
    name = "document_reader"
    description = "按路径读取本地文档 (txt/md/pdf)，返回片段文本，用于 Agent 查阅合同/政策/征信报告等。"
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文档相对或绝对路径"},
            "offset": {"type": "integer", "description": "起始字符偏移, 默认 0"},
            "limit": {"type": "integer", "description": "最多读取的字符数, 默认 4000"},
        },
        "required": ["path"],
    }

    def __init__(self, base_dir: str = "."):
        self.base_dir = os.path.abspath(base_dir)

    def run(self, **kwargs: Any) -> ToolResult:
        path = str(kwargs.get("path", "")).strip()
        if not path:
            return ToolResult(content="缺少 path 参数", success=False)
        resolved = path if os.path.isabs(path) else os.path.join(self.base_dir, path)
        if not os.path.isfile(resolved):
            return ToolResult(content=f"[DocumentReader] 文件不存在: {resolved}", success=False)

        offset = max(0, int(kwargs.get("offset", 0)))
        limit = int(kwargs.get("limit", 4000))

        text = self._read_text(resolved)
        if text is None:
            return ToolResult(
                content=f"[DocumentReader] 不支持的文件类型或读取失败: {resolved}",
                success=False,
            )
        chunk = text[offset : offset + limit]
        return ToolResult(
            content=chunk,
            success=True,
            meta={
                "path": resolved,
                "total_len": len(text),
                "offset": offset,
                "returned_len": len(chunk),
            },
        )

    # ---- impl ----
    @staticmethod
    def _read_text(path: str) -> str | None:
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in (".txt", ".md", ".log"):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            if ext == ".pdf":
                try:
                    from pypdf import PdfReader  # type: ignore
                except Exception:
                    try:
                        from PyPDF2 import PdfReader  # type: ignore
                    except Exception:
                        return None
                reader = PdfReader(path)
                parts = []
                for page in reader.pages:
                    try:
                        parts.append(page.extract_text() or "")
                    except Exception:
                        continue
                return "\n".join(parts)
            # 其他类型默认按文本尝试
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            print(f"[DocumentReaderTool] 读取失败: {e}")
            return None


__all__ = ["DocumentReaderTool"]

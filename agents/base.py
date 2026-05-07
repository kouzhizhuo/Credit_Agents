"""Abstract base class for all CreditAgent agents."""
from __future__ import annotations

from typing import Any


class BaseAgent:
    """所有 Agent 的统一接口。``run(**kwargs)`` 为默认入口。"""

    name: str = "base"

    def run(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError


__all__ = ["BaseAgent"]

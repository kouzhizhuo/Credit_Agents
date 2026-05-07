"""core.blackboard: Layer 2 专用的共享黑板架构。

对应技术报告中的 *Shared Blackboard Architecture*：
每个 Agent 将 ``Intermediate Risk Findings`` 发布到共享状态
``S_shared``，下游 Agent 通过 ``depends_on`` 订阅上游 finding。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable


@dataclass
class SharedBlackboard:
    """跨 Agent 的共享黑板。"""

    findings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    global_context: Dict[str, Any] = field(default_factory=dict)
    tool_trace: Dict[str, list] = field(default_factory=dict)

    # ---- 读 ----
    def get_related_findings(
        self, agent_name: str, depends_on: Iterable[str]
    ) -> str:
        """以可读文本形式汇总上游 Agent 的 finding 供下游注入 prompt。"""
        if not depends_on:
            return ""
        sections = []
        for dep in depends_on:
            if dep not in self.findings:
                continue
            dep_obj = self.findings[dep]
            score = dep_obj.get("评分", "未知")
            basis = dep_obj.get("依据", "")
            evidence = dep_obj.get("关键证据", []) or []
            sections.append(
                f"{dep}维度评估结果：\n"
                f"- 评分：{score}/10\n"
                f"- 依据：{basis}\n"
                f"- 关键证据：{', '.join(evidence) if evidence else '无'}\n"
            )
        return "\n".join(sections) if sections else ""

    def get_global_context(self) -> str:
        """拼接全局上下文信息，供 ``can_access_global=True`` 的 Agent 使用。"""
        if not self.global_context:
            return ""
        return (
            "全局上下文信息："
            + json.dumps(self.global_context, ensure_ascii=False, indent=2)
        )

    def snapshot(self) -> Dict[str, Any]:
        """返回可序列化的黑板快照。"""
        return {
            "findings": {k: dict(v) for k, v in self.findings.items()},
            "global_context": dict(self.global_context),
            "tool_trace": {k: list(v) for k, v in self.tool_trace.items()},
        }

    # ---- 写 ----
    def publish(self, agent_name: str, findings: Dict[str, Any]) -> None:
        self.findings[agent_name] = findings

    def update_global(self, key: str, value: Any) -> None:
        self.global_context[key] = value

    def record_tool_call(
        self, agent_name: str, tool_name: str, args: Any, result: Any
    ) -> None:
        """记录 Agent 的工具调用轨迹，供审计链使用。"""
        self.tool_trace.setdefault(agent_name, []).append(
            {"tool": tool_name, "args": args, "result": result}
        )


__all__ = ["SharedBlackboard"]

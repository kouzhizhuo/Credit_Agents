"""Layer 2 · DimensionAgent

单个维度的专业评估 Agent，核心能力:
1. 订阅 Shared Blackboard 的上游 finding / global context；
2. 可选地通过 ``ToolRegistry`` 调用外部工具 (web_search / document_reader /
   knowledge_base) 获取补充证据；
3. 最终输出严格 JSON ``{"维度","评分","依据","关键证据"}``。

工具调用协议（Prompt-level Tool-Use，显式 JSON，不依赖模型原生 function calling）:

    第一步：Agent 看到 `dimension_data + 上游 finding + 可用工具`，可以选择返回
            ``{"action": "use_tool", "tool": "web_search", "args": {...}}``；
            也可以直接返回最终评分 JSON。
    第二步：若请求了工具，执行结果作为 "工具观察" 再注入 prompt，重复直到
            ``max_tool_iterations`` 或直接返回评分。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...core.blackboard import SharedBlackboard
from ...core.llm_client import LLMClient
from ...core.utils import safe_json_loads
from ...tools.base import Tool, ToolResult
from ...tools.registry import ToolRegistry
from ..base import BaseAgent
from .specs import DIMENSION_PROMPTS, DimensionSpec


TOOL_INSTRUCTION = """### 工具使用协议

你可以访问以下工具来补强证据。若判断当前信息已足够，请**直接**返回最终评分 JSON；
否则可以返回一个工具调用请求。

可用工具:
{tool_list}

若需要调用工具，严格返回如下 JSON (action=use_tool)：
```json
{{
  "action": "use_tool",
  "tool": "<tool_name>",
  "args": {{"...": "..."}}
}}
```

若已得到结论，直接返回评分 JSON (无需 action 字段):
```json
{{
  "维度": "...",
  "评分": 1,
  "依据": "...",
  "关键证据": ["..."]
}}
```
"""


@dataclass
class DimensionAgent(BaseAgent):
    spec: DimensionSpec
    llm: LLMClient
    blackboard: Optional[SharedBlackboard] = None
    tool_registry: Optional[ToolRegistry] = None
    max_tool_iterations: int = 2
    verbose: bool = False
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"layer2.{self.spec.name}"

    # ---- API ----
    def run(self, dimension_data: str) -> Dict[str, Any]:
        return self.evaluate(dimension_data)

    def evaluate(self, dimension_data: str) -> Dict[str, Any]:
        system_prompt, user_template = DIMENSION_PROMPTS.get(
            self.spec.name, self._default_prompts()
        )
        blackboard = self.blackboard or SharedBlackboard()
        related = (
            blackboard.get_related_findings(self.spec.name, self.spec.depends_on) or ""
        )
        related_section = (
            f"### 上游维度的 findings\n{related}" if related else ""
        )
        global_ctx = (
            blackboard.get_global_context() if self.spec.can_access_global else ""
        )
        global_section = f"### 全局上下文\n{global_ctx}" if global_ctx else ""

        observations: List[str] = []
        final_obj: Optional[Dict[str, Any]] = None

        iterations = self.max_tool_iterations if (
            self.tool_registry and self.tool_registry.tools
        ) else 0

        for step in range(iterations + 1):
            tool_section = self._render_tool_section(observations)
            user_prompt = user_template.format(
                dimension_data=dimension_data or "无相关数据",
                related_findings=related_section,
                global_context=global_section,
                tool_observations=tool_section,
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            if step < iterations and self.tool_registry and self.tool_registry.tools:
                messages[0]["content"] += "\n\n" + TOOL_INSTRUCTION.format(
                    tool_list=self.tool_registry.describe_all()
                )

            raw = self.llm.chat_messages(messages)
            obj = safe_json_loads(raw or "")
            if not isinstance(obj, dict):
                # 解析失败 → 兜底中性分，结束
                final_obj = self._neutral_result("LLM 输出解析失败")
                break

            action = str(obj.get("action", "")).lower()
            if action == "use_tool" and step < iterations:
                tool_name = str(obj.get("tool", "")).strip()
                args = obj.get("args") or {}
                self._log(f"[{self.spec.name}] 调用工具 {tool_name} args={args}")
                tool_result = self._call_tool(tool_name, args, blackboard)
                observations.append(
                    f"工具 `{tool_name}` 返回:\n{tool_result.as_observation()}"
                )
                continue

            final_obj = self._normalize_result(obj)
            break

        if final_obj is None:
            final_obj = self._neutral_result("工具迭代耗尽，未获得最终评分")

        # 记录工具轨迹到 blackboard
        if observations and blackboard is not None:
            blackboard.record_tool_call(
                self.spec.name, "dimension_agent", {"steps": len(observations)}, final_obj
            )
        return final_obj

    # ---- helpers ----
    def _call_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        blackboard: SharedBlackboard,
    ) -> ToolResult:
        if not self.tool_registry or not self.tool_registry.has(tool_name):
            return ToolResult(
                content=f"[DimensionAgent] 未注册的工具 {tool_name}", success=False
            )
        result = self.tool_registry.call(tool_name, args)
        blackboard.record_tool_call(self.spec.name, tool_name, args, {
            "content_preview": (result.content or "")[:300],
            "success": result.success,
        })
        return result

    def _render_tool_section(self, observations: List[str]) -> str:
        if not observations:
            return ""
        joined = "\n\n".join(f"[Observation {i+1}]\n{o}" for i, o in enumerate(observations))
        return f"### 已获得的工具观察\n{joined}"

    def _normalize_result(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        obj = dict(obj)
        obj.setdefault("维度", self.spec.name)
        try:
            obj["评分"] = int(obj.get("评分", 5))
        except (TypeError, ValueError):
            obj["评分"] = 5
        obj.setdefault("依据", "")
        if not isinstance(obj.get("关键证据"), list):
            obj["关键证据"] = []
        return obj

    def _neutral_result(self, reason: str) -> Dict[str, Any]:
        return {
            "维度": self.spec.name,
            "评分": 5,
            "依据": reason,
            "关键证据": [],
        }

    def _default_prompts(self) -> tuple:
        system = (
            f"你是一位资深信贷风控专家。你的任务是评估{self.spec.name}维度。"
            " 评分规则：1-10分，分数越高表示风险越低。严格只输出 JSON。"
        )
        user = (
            f"维度名称：{self.spec.name}\n"
            f"维度关注点：{self.spec.focus}\n\n"
            "该维度可用数据：\n{dimension_data}\n\n"
            "{related_findings}\n{global_context}\n{tool_observations}\n\n"
            "请输出严格 JSON:\n"
            "{{\"维度\": \"...\", \"评分\": 1, \"依据\": \"...\", \"关键证据\": [\"...\"]}}"
        )
        return system, user

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


__all__ = ["DimensionAgent"]

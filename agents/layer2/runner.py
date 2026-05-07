"""Layer 2 · Runner

串行驱动六个 DimensionAgent，按 ``can_access_global`` / ``depends_on`` 排序；
同时支持可选的工具注册表与 LangChain ChatOpenAI 兼容模式 (兼容旧调用)。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ...core.blackboard import SharedBlackboard
from ...core.llm_client import LLMClient
from ...tools.registry import ToolRegistry
from .dimension_agent import DimensionAgent
from .specs import DEFAULT_DIM_SPECS, DimensionSpec


def build_six_dimension_agents(
    llm: LLMClient,
    *,
    specs: Tuple[DimensionSpec, ...] = DEFAULT_DIM_SPECS,
    blackboard: Optional[SharedBlackboard] = None,
    tool_registry: Optional[ToolRegistry] = None,
    max_tool_iterations: int = 2,
    verbose: bool = False,
) -> Dict[str, DimensionAgent]:
    """按 specs 构造 ``{维度名: DimensionAgent}``。"""
    return {
        spec.name: DimensionAgent(
            spec=spec,
            llm=llm,
            blackboard=blackboard,
            tool_registry=tool_registry,
            max_tool_iterations=max_tool_iterations,
            verbose=verbose,
        )
        for spec in specs
    }


def _topologically_ordered(
    agents: Dict[str, DimensionAgent],
    specs: Tuple[DimensionSpec, ...] = DEFAULT_DIM_SPECS,
) -> List[str]:
    spec_map = {s.name: s for s in specs}
    order: List[str] = []
    processed: set = set()
    # 第一轮：全局可见的 Agent
    for name in agents:
        s = spec_map.get(name)
        if s and s.can_access_global:
            order.append(name)
            processed.add(name)
    # 后续：按依赖
    while len(processed) < len(agents):
        progressed = False
        for name in agents:
            if name in processed:
                continue
            s = spec_map.get(name)
            if s and all(dep in processed for dep in s.depends_on):
                order.append(name)
                processed.add(name)
                progressed = True
        if not progressed:
            for name in agents:
                if name not in processed:
                    order.append(name)
                    processed.add(name)
            break
    return order


def run_six_dimension_scoring(
    agents: Dict[str, DimensionAgent],
    *,
    dim_texts: Dict[str, str],
    blackboard: Optional[SharedBlackboard] = None,
    parallel: bool = False,  # 保留签名以兼容旧代码
) -> Dict[str, Any]:
    """执行六维度评估并聚合输出。

    注：依赖顺序必须串行执行，``parallel`` 参数仅为向后兼容保留。
    """
    bb = blackboard or SharedBlackboard()
    order = _topologically_ordered(agents)
    raw: Dict[str, Dict[str, Any]] = {}

    for name in order:
        data = dim_texts.get(name, "无相关数据") or "无相关数据"
        try:
            result = agents[name].evaluate(data)
        except Exception as e:
            print(f"[Layer2][ERROR] Agent '{name}' 失败: {e}")
            result = {
                "维度": name,
                "评分": 5,
                "依据": f"评估失败: {e}",
                "关键证据": [],
            }
        raw[name] = result
        bb.publish(name, result)

    scores: Dict[str, Any] = {}
    for dim_name, detail in raw.items():
        if isinstance(detail, dict) and "评分" in detail:
            scores[dim_name] = detail["评分"]
        else:
            scores[dim_name] = None

    return {
        "六大维度评分": scores,
        "六大维度明细": raw,
        "共享黑板": bb.snapshot(),
    }


__all__ = [
    "build_six_dimension_agents",
    "run_six_dimension_scoring",
]

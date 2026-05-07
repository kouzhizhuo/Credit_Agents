"""Layer 2: Multi-Agent Domain Reasoning —— 六维度专业评估。

对齐技术报告 §Layer 2：
- Non-Parallel Domain Specialist + Shared Blackboard Architecture
- 每个 Agent 专注一个维度 (Macro / Basic / Solvency / Willingness / History / Fraud)
- 新增: 每个 Agent 可通过工具调用（web_search / document_reader / knowledge_base）
  获取外部证据，增强"Evidence Chain e_j"的质量。
"""
from .specs import DimensionSpec, DEFAULT_DIM_SPECS, DIMENSION_PROMPTS
from .dimension_agent import DimensionAgent
from .runner import run_six_dimension_scoring, build_six_dimension_agents

__all__ = [
    "DimensionSpec",
    "DEFAULT_DIM_SPECS",
    "DIMENSION_PROMPTS",
    "DimensionAgent",
    "build_six_dimension_agents",
    "run_six_dimension_scoring",
]

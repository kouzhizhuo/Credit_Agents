"""agents: 三层 Agent 的工程化实现。

子包:
- ``layer1``: Dynamic Data Governance (Data Cleaning / Feature Analysis /
  Coarse Screening / Fine-Grained Selection / LLM Final Judge / Final Validation)。
- ``layer2``: 六维度 Domain Reasoning Agent + Blackboard + Tool Calling。
- ``layer3``: Score Decision Agent + Strategy Analysis Agent。
"""
from .base import BaseAgent

__all__ = ["BaseAgent"]

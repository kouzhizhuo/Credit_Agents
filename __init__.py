"""CreditAgent: Hierarchical Multi-Agent Systems for High-Stakes Financial Decision-Making.

工程化后的三层多智能体系统实现，对齐技术报告 (CreditAgent_Technical_Report_0421.pdf)：

- Layer 1 (Dynamic Data Governance): 自动化特征治理管道，含数据清理、特征分析、
  粗筛选、细粒度选择（Pearson + XGBoost + LLM 语义相关性）、LLM 最终业务判断 & 验证导出。
- Layer 2 (Multi-Agent Domain Reasoning): 六维度专业 Agent + Shared Blackboard +
  可插拔工具（网页搜索、文档阅读、知识库检索）增强证据获取。
- Layer 3 (Decision Fusion & Risk Gating): Score Decision Agent 与
  Strategy Analysis Agent，支持 Hard-Stop 与 Deep Review。

目录布局参考 ``alphaevolve_agent``：``config`` / ``core`` / ``agents`` / ``tools`` / ``pipelines``。
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]

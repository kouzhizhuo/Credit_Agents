"""CreditAgent: Hierarchical Multi-Agent Credit Review System.

Three-layer architecture:

- Layer 1 (Dynamic Data Governance): Automated feature governance pipeline
  with data cleaning, feature analysis, coarse screening, fine-grained
  selection (Pearson + XGBoost + LLM semantic relevance), and LLM final
  business judgment.
- Layer 2 (Multi-Agent Domain Reasoning): Six specialist agents with a
  shared blackboard and optional tool augmentation (web search, document
  reader, knowledge base).
- Layer 3 (Decision Fusion & Risk Gating): Score fusion via learned
  attention weights, GRPO-optimized threshold, and hard-stop vetoes.
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]

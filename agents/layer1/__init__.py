"""Layer 1: Dynamic Data Governance —— 自动化特征治理管道。

按技术报告 Layer 1 (Eq. 1 / Eq. 9) 组织:
- ``DataCleaningAgent``: 缺失/异常/类型清理。
- ``FeatureAnalysisAgent``: 元数据 + 业务描述（字典优先、LLM 次之、启发式兜底）。
- ``CoarseScreeningAgent``: ID/哈希/常量/重复列剔除。
- ``FineGrainedSelectionAgent``: 迭代式 Pearson + XGBoost + LLM 语义相关性筛选。
- ``LLMRelevanceAgent``: Rel_llm 评分。
- ``LLMFinalJudgeAgent``: 细粒度选择后用 LLM 做业务层最终判断，决定保留/剔除/警告。
- ``FinalValidationAgent``: 导出 CSV + 日志 JSON。
"""
from .schema import CleaningLog, FeatureMetadata
from .data_cleaning import DataCleaningAgent
from .feature_analysis import FeatureAnalysisAgent
from .coarse_screening import CoarseScreeningAgent
from .llm_relevance import LLMRelevanceAgent
from .fine_selection import FineGrainedSelectionAgent
from .llm_final_judge import LLMFinalJudgeAgent, FinalJudgeResult
from .final_validation import FinalValidationAgent

__all__ = [
    "CleaningLog",
    "FeatureMetadata",
    "DataCleaningAgent",
    "FeatureAnalysisAgent",
    "CoarseScreeningAgent",
    "LLMRelevanceAgent",
    "FineGrainedSelectionAgent",
    "LLMFinalJudgeAgent",
    "FinalJudgeResult",
    "FinalValidationAgent",
]

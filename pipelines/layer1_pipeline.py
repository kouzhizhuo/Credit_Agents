"""Layer1Pipeline: 特征治理端到端流水线。

顺序：
    DataCleaning → FeatureAnalysis → CoarseScreening → FineGrainedSelection
    → LLMFinalJudge (新) → FinalValidation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from ..agents.layer1 import (
    CoarseScreeningAgent,
    DataCleaningAgent,
    FeatureAnalysisAgent,
    FinalJudgeResult,
    FinalValidationAgent,
    FineGrainedSelectionAgent,
    LLMFinalJudgeAgent,
)
from ..config import Layer1Config, LayerLLMConfig, ScenarioConfig
from ..core.llm_client import LLMClient, build_llm_client_from_layer_cfg


@dataclass
class Layer1Result:
    selected_features: List[str] = field(default_factory=list)
    cleaning_log: Dict[str, Any] = field(default_factory=dict)
    screening_log: List[Any] = field(default_factory=list)
    selection_log: Dict[str, Any] = field(default_factory=dict)
    final_judge: Optional[FinalJudgeResult] = None
    output_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_features": list(self.selected_features),
            "cleaning_log": dict(self.cleaning_log),
            "screening_log": list(self.screening_log),
            "selection_log": dict(self.selection_log),
            "final_judge": self.final_judge.to_dict() if self.final_judge else None,
            "output_path": self.output_path,
        }


class Layer1Pipeline:
    """Layer 1 端到端流水线。"""

    def __init__(
        self,
        *,
        scenario: ScenarioConfig,
        layer1_cfg: Layer1Config,
        llm_cfg: Optional[LayerLLMConfig] = None,
        llm: Optional[LLMClient] = None,
        verbose: bool = True,
    ):
        self.scenario = scenario
        self.cfg = layer1_cfg
        self.verbose = verbose
        if llm is not None:
            self.llm = llm
        elif llm_cfg is not None:
            self.llm = build_llm_client_from_layer_cfg(llm_cfg)
        else:
            self.llm = LLMClient()

    # ---- API ----
    def run(self, df: pd.DataFrame) -> Layer1Result:
        tgt = self.scenario.target_col

        cleaner = DataCleaningAgent(
            missing_threshold=self.cfg.missing_threshold,
            outlier_method=self.cfg.outlier_method,
            verbose=self.verbose,
        )
        df_clean = cleaner.clean(df, target_col=tgt)

        analyzer = FeatureAnalysisAgent(
            llm=self.llm, dict_file=self.cfg.dict_file, verbose=self.verbose
        )
        metadata_list = analyzer.analyze(df_clean, target_col=tgt)

        screener = CoarseScreeningAgent(verbose=self.verbose)
        df_screen, remaining = screener.screen(df_clean, metadata_list, target_col=tgt)
        # 仅保留被粗筛保留下来的特征元数据
        metadata_after = [m for m in metadata_list if m.name in remaining]

        selector = FineGrainedSelectionAgent(
            target_col=tgt,
            target_features=self.cfg.target_features,
            prune_ratio=self.cfg.prune_ratio,
            cv_folds=self.cfg.cv_folds,
            lambda_weight=self.cfg.lambda_weight,
            llm=self.llm,
            verbose=self.verbose,
        )
        selected, selection_log = selector.select(
            df_screen, remaining, metadata_after
        )

        final_judge: Optional[FinalJudgeResult] = None
        if self.cfg.enable_llm_final_judge:
            judge = LLMFinalJudgeAgent(
                llm=self.llm,
                batch_size=self.cfg.final_judge_batch_size,
                drop_threshold=self.cfg.final_judge_drop_threshold,
                verbose=self.verbose,
            )
            final_judge = judge.judge(
                selected,
                [m for m in metadata_after if m.name in selected],
            )
            if final_judge.dropped_features:
                selected = [
                    f for f in selected if f not in set(final_judge.dropped_features)
                ]

        validator = FinalValidationAgent(
            output_dir=self.scenario.output_dir, verbose=self.verbose
        )
        output_path = validator.validate_and_export(
            selected,
            [m for m in metadata_after if m.name in selected],
            cleaner.get_log(),
            selection_log,
            screening_log=screener.get_removed_features(),
            final_judge=final_judge,
            output_filename=self.cfg.output_filename,
        )

        return Layer1Result(
            selected_features=selected,
            cleaning_log=cleaner.get_log(),
            screening_log=screener.get_removed_features(),
            selection_log=selection_log,
            final_judge=final_judge,
            output_path=output_path,
        )


__all__ = ["Layer1Pipeline", "Layer1Result"]

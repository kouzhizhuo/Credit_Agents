# CreditAgent

**Hierarchical Multi-Agent Credit Review System**

This repository contains the implementation and evaluation data for CreditAgent, a three-layer hierarchical credit review system that decomposes credit underwriting into evidence filtering, specialist risk analysis, and decision fusion.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CreditAgent Pipeline                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Layer 1: Dynamic Data Governance                                   │
│  ┌───────────┐   ┌──────────┐   ┌───────────┐   ┌──────────────┐  │
│  │Data Clean │──▶│Coarse    │──▶│XGBoost +  │──▶│LLM Final    │  │
│  │& Analysis │   │Screening │   │LLM Select │   │Judge        │  │
│  └───────────┘   └──────────┘   └───────────┘   └──────────────┘  │
│                                                                     │
│  Layer 2: Multi-Agent Domain Reasoning (Shared Blackboard)          │
│  ┌────────┐ ┌────────┐ ┌─────────┐ ┌─────────┐ ┌──────┐ ┌─────┐  │
│  │ Macro  │▶│ Basic  │▶│ History │▶│Solvency │▶│Will. │▶│Fraud│  │
│  └────────┘ └────────┘ └─────────┘ └─────────┘ └──────┘ └─────┘  │
│       ▲           ▲          ▲          ▲          ▲        ▲      │
│       └───────────┴──────────┴──────────┴──────────┴────────┘      │
│                       Shared Blackboard                             │
│                                                                     │
│  Layer 3: Decision Fusion & Risk Gating                             │
│  ┌────────────────────┐   ┌──────────────┐   ┌──────────────────┐  │
│  │Attention-Weighted  │──▶│GRPO Threshold│──▶│Hard-Stop Vetoes  │  │
│  │Score Fusion        │   │Optimization  │   │& Final Decision  │  │
│  └────────────────────┘   └──────────────┘   └──────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Code Structure

```
creditagent/
├── config.py                 # Runtime configuration (all layers)
├── run.py                    # CLI entry point
├── agents/
│   ├── base.py               # Abstract agent interface
│   ├── layer1/               # Feature governance agents
│   │   ├── data_cleaning.py      # Missing values, outliers, type conversion
│   │   ├── coarse_screening.py   # Remove uninformative columns
│   │   ├── feature_analysis.py   # Generate feature metadata (dict/LLM/heuristic)
│   │   ├── fine_selection.py     # Iterative selection: Pearson + XGBoost + LLM
│   │   ├── llm_relevance.py     # LLM semantic relevance scoring
│   │   ├── llm_final_judge.py   # LLM business-layer final review
│   │   └── final_validation.py  # Export audit artifacts
│   ├── layer2/               # Specialist risk agents
│   │   ├── specs.py              # Six-dimension definitions & prompts
│   │   ├── dimension_agent.py    # Single-dimension evaluation with tool use
│   │   └── runner.py            # Topological execution with blackboard
│   └── layer3/               # Decision fusion agents
│       ├── score_decision.py     # Attention-weighted score fusion
│       └── strategy_analysis.py  # Threshold + hard-stop + final decision
├── core/
│   ├── blackboard.py         # Shared Blackboard implementation
│   ├── llm_client.py         # OpenAI-compatible API wrapper
│   └── utils.py              # JSON parsing, seed, serialization
├── pipelines/
│   ├── layer1_pipeline.py    # End-to-end feature governance
│   ├── layer2_pipeline.py    # Six-agent evaluation pipeline
│   ├── layer3_pipeline.py    # Score fusion + strategy pipeline
│   └── full_pipeline.py      # Layer 1 → 2 → 3 orchestration
├── tools/                    # Optional tool augmentation for Layer 2
│   ├── web_search.py             # DuckDuckGo web search
│   ├── document_reader.py       # Local file reader (txt/pdf)
│   ├── knowledge_base.py        # BM25 local document retrieval
│   └── registry.py             # Tool registration & dispatch
├── data/
│   └── data_sample.jsonl     # Sample evaluation data (6 dimensions)
└── requirements.txt
```

## Layer 2 Agent Specifications

| Agent | Paper Name | Core Input | Focus | Risk Metric |
|-------|-----------|------------|-------|-------------|
| Macro | Macro | PESTEL factors | Systematic risk | Market sensitivity |
| Basic | Basic | KYB/KYC | Identity verification | Fraud suspicion |
| History | History | Credit bureau | Payment velocity | Default risk |
| Solvency | Solvency | Cash flow | Assets vs. debt | Liquidity buffer |
| Willingness | Willingness | Behavioral signals | Repayment intent | Sincerity score |
| Fraud | Fraud | Anomaly patterns | Adversarial behavior | Synthetic patterns |

Agents execute sequentially following a dependency graph. Each agent writes structured findings `{risk_score, evidence}` to the shared blackboard, enabling downstream agents to condition on upstream analysis.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Layer 1: Feature Governance

Select decision-relevant features from raw credit data:

```bash
python -m creditagent.run layer1 \
    --input dataset/raw_features.csv \
    --target target \
    --output-dir outputs/layer1
```

### Layer 2 + 3: Multi-Agent Evaluation & Decision

Run specialist risk assessment and decision fusion on prepared samples:

```bash
python -m creditagent.run layer2-3 \
    --input data/data_sample.jsonl \
    --output outputs/predictions.jsonl \
    --threshold 50 --margin 10
```

### Full Pipeline

```bash
python -m creditagent.run all \
    --layer1-input dataset/raw_features.csv \
    --target target \
    --samples data/data_sample.jsonl \
    --output outputs/predictions.jsonl
```

## Configuration

All parameters are configured via `config.py` dataclasses. Key settings:

- **Layer 1**: `target_features` (default 100), `lambda_weight` (hybrid score balance), `enable_llm_final_judge`
- **Layer 2**: `parallel` (blackboard requires sequential), `enable_tools`, tool selection
- **Layer 3**: `threshold` (decision boundary), `margin` (deep-review zone width)

LLM backends are configured via environment variables or the `LLMConfig` dataclass, supporting any OpenAI-compatible API endpoint.

## Data Format

Input samples for Layer 2+3 are JSONL with the following structure:

```json
{
  "instruction": "Six-dimension credit evidence text (structured fields)",
  "simple_result": "ground_truth_label"
}
```

The `instruction` field contains structured credit bureau data organized into six evaluation dimensions, which are parsed and routed to the corresponding specialist agents.

## Evaluation Metrics

- **BEC (Business Efficiency Coefficient)**: Cost-sensitive utility metric with asymmetric FN/FP costs (C_FN:C_FP = 3:1)
- **Accuracy**: Overall prediction quality
- **Stability (σ)**: Standard deviation of per-group accuracies

## License

This project is released for academic research purposes.

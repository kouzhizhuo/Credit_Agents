"""CreditAgent CLI entry point.

支持三种子命令：
- ``layer1``   : 仅运行特征治理流水线 (CSV → 选定特征 CSV + 审计 JSON)
- ``layer2-3`` : 读取 JSONL 样本 → Layer2 六维评分 + Layer3 决策融合
- ``all``      : 先 layer1 再 layer2-3 (需要分别指定 CSV 和 JSONL 输入)

使用示例::

    python -m creditagent.run layer2-3 \
        --input data/samples.jsonl \
        --output outputs/creditagent/results.jsonl \
        --threshold 50 --margin 10

    python -m creditagent.run layer1 \
        --input dataset/merged_output_data.csv \
        --target target \
        --output-dir outputs/creditagent/layer1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

import pandas as pd


def _ensure_project_on_path() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.abspath(os.path.join(here, ".."))
    if parent not in sys.path:
        sys.path.insert(0, parent)


_ensure_project_on_path()


from creditagent.config import RuntimeConfig  # noqa: E402
from creditagent.pipelines import FullPipeline  # noqa: E402


# ---------------------------- 通用 ---------------------------- #
def _build_runtime(args: argparse.Namespace) -> RuntimeConfig:
    cfg = RuntimeConfig()
    if getattr(args, "target", None):
        cfg.scenario.target_col = args.target
    if getattr(args, "output_dir", None):
        cfg.scenario.output_dir = args.output_dir
    if getattr(args, "threshold", None) is not None:
        cfg.layer3.threshold = float(args.threshold)
    if getattr(args, "margin", None) is not None:
        cfg.layer3.margin = float(args.margin)
    if getattr(args, "disable_tools", False):
        cfg.layer2.enable_tools = False
    if getattr(args, "disable_llm_judge", False):
        cfg.layer1.enable_llm_final_judge = False
    return cfg


# ---------------------------- Layer 1 ---------------------------- #
def _run_layer1(args: argparse.Namespace) -> None:
    cfg = _build_runtime(args)
    pipeline = FullPipeline(cfg, verbose=True)
    if not args.input or not os.path.isfile(args.input):
        print(f"[run] 缺少 Layer1 输入 CSV: {args.input}")
        return
    nrows = args.nrows if args.nrows and args.nrows > 0 else None
    print(f"[run] 读取 {args.input} (nrows={nrows}) ...")
    df = pd.read_csv(args.input, nrows=nrows)
    print(f"[run] shape={df.shape}, target={cfg.scenario.target_col}")
    result = pipeline.run_layer1(df)
    print(f"[run] Layer1 完成, 最终 {len(result.selected_features)} 个特征")
    print(f"[run] 输出: {result.output_path}")


# ---------------------------- Layer 2+3 ---------------------------- #
def _load_jsonl(path: str) -> list:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except Exception as e:
                print(f"[run][WARN] 跳过非法 JSONL 行: {e}")
    return out


def _normalize_sample(raw: Dict[str, Any]) -> Dict[str, Any]:
    """把原始 JSONL 行转换成 FullPipeline.run_per_sample 期望的格式。"""
    # 允许旧的 `ground_truth` / `simple_result` 兼容字段
    sample: Dict[str, Any] = {
        "instruction": raw.get("instruction", ""),
        "simple_result": raw.get("simple_result", raw.get("ground_truth", "")),
    }
    return sample


def _run_layer23(args: argparse.Namespace) -> None:
    cfg = _build_runtime(args)
    pipeline = FullPipeline(cfg, verbose=False)
    if not args.input or not os.path.isfile(args.input):
        print(f"[run] 缺少样本 JSONL: {args.input}")
        return
    samples = _load_jsonl(args.input)
    if not samples:
        print("[run] 样本为空")
        return

    output_path = args.output or os.path.join(
        cfg.scenario.output_dir, "creditagent_predictions.jsonl"
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    ok = fail = 0
    print(f"[run] Layer2+3 开始，共 {len(samples)} 条")
    with open(output_path, "w", encoding="utf-8") as fout:
        for i, raw in enumerate(samples, 1):
            try:
                res = pipeline.run_per_sample(_normalize_sample(raw))
                fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                fout.flush()
                ok += 1
            except Exception as e:
                print(f"[run][ERROR] 样本 {i} 失败: {e}")
                fail += 1
    print(f"[run] 完成: success={ok}, fail={fail}")
    print(f"[run] 输出: {output_path}")


# ---------------------------- CLI ---------------------------- #
def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--threshold", type=float, default=None, help="Layer3 决策阈值 τ")
    p.add_argument("--margin", type=float, default=None, help="Layer3 边界区间 margin")
    p.add_argument(
        "--disable-tools",
        action="store_true",
        help="关闭 Layer2 工具调用（web_search / document_reader / knowledge_base）",
    )
    p.add_argument(
        "--disable-llm-judge",
        action="store_true",
        help="关闭 Layer1 LLM 最终业务判断",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认 ./outputs）",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CreditAgent runtime (Layer1 / Layer2 / Layer3)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # layer1
    p1 = sub.add_parser("layer1", help="运行 Layer 1 特征治理流水线")
    p1.add_argument("--input", type=str, required=True, help="CSV 输入路径")
    p1.add_argument("--target", type=str, default="target", help="目标列名")
    p1.add_argument("--nrows", type=int, default=None, help="调试：仅读前 N 行")
    _add_common_args(p1)

    # layer2-3
    p2 = sub.add_parser("layer2-3", help="运行 Layer 2 + Layer 3")
    p2.add_argument("--input", type=str, required=True, help="JSONL 输入路径")
    p2.add_argument("--output", type=str, default=None, help="JSONL 输出路径")
    _add_common_args(p2)

    # all
    p3 = sub.add_parser("all", help="Layer1 + Layer2-3 (需要同时指定 CSV / JSONL)")
    p3.add_argument("--layer1-input", type=str, required=True, help="Layer1 CSV")
    p3.add_argument("--target", type=str, default="target", help="Layer1 目标列名")
    p3.add_argument("--samples", type=str, required=True, help="Layer2+3 JSONL")
    p3.add_argument("--output", type=str, default=None, help="Layer2+3 JSONL 输出")
    p3.add_argument("--nrows", type=int, default=None)
    _add_common_args(p3)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "layer1":
        _run_layer1(args)
    elif args.cmd == "layer2-3":
        _run_layer23(args)
    elif args.cmd == "all":
        # layer1
        class _A:
            pass

        a1 = _A()
        a1.input = args.layer1_input
        a1.target = args.target
        a1.nrows = args.nrows
        a1.threshold = args.threshold
        a1.margin = args.margin
        a1.disable_tools = args.disable_tools
        a1.disable_llm_judge = args.disable_llm_judge
        a1.output_dir = args.output_dir
        _run_layer1(a1)  # type: ignore[arg-type]
        # layer2+3
        a2 = _A()
        a2.input = args.samples
        a2.output = args.output
        a2.threshold = args.threshold
        a2.margin = args.margin
        a2.disable_tools = args.disable_tools
        a2.disable_llm_judge = args.disable_llm_judge
        a2.output_dir = args.output_dir
        _run_layer23(a2)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()

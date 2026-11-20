#!/usr/bin/env python3
"""
跨 Benchmark 汇总对照脚本：
将 ScienceWorld (A) 与 TextWorldExpress (B) 的 Stage 3 产出按相同统计口径并排对比，
产出跨 Bench 对照报告 (cross_benchmark_comparison.md) 与总览对比图 (Fig_cross_benchmark_overview.png)。
注意：遵循全组约定，绝不直接拿两边原始 reward 绝对值比大小，只比相关性、正负比例与标准化趋势。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import ensure_dir


def _df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(empty)_"
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = [
        "| " + " | ".join([str(row[c]) if pd.notna(row[c]) else "" for c in df.columns]) + " |"
        for _, row in df.iterrows()
    ]
    return "\n".join([header, sep, *body])


def main() -> None:
    parser = argparse.ArgumentParser(description="Side-by-side comparison across ScienceWorld and TextWorldExpress.")
    parser.add_argument("--sw-results", required=True, help="Stage 3 results dir for ScienceWorld.")
    parser.add_argument("--tw-results", required=True, help="Stage 3 results dir for TextWorldExpress.")
    parser.add_argument("--out-dir", required=True, help="Output directory for cross-benchmark summary.")
    args = parser.parse_args()

    sw_dir = Path(args.sw_results) / "tables"
    tw_dir = Path(args.tw_results) / "tables"
    out_dir = ensure_dir(args.out_dir)

    sw_sum = pd.read_csv(sw_dir / "summary_stats.csv")
    tw_sum = pd.read_csv(tw_dir / "summary_stats.csv")
    sw_e09 = pd.read_csv(sw_dir / "e09_correlations.csv")
    tw_e09 = pd.read_csv(tw_dir / "e09_correlations.csv")

    # 1. Side-by-side E09 correlations (the fairest cross-benchmark comparison!)
    merged_e09 = sw_e09.merge(
        tw_e09, on=["x", "y"], suffixes=("_SW", "_TW"), how="outer"
    )
    merged_e09["same_sign"] = np.where(
        merged_e09["spearman_pooled_SW"].notna() & merged_e09["spearman_pooled_TW"].notna(),
        np.sign(merged_e09["spearman_pooled_SW"]) == np.sign(merged_e09["spearman_pooled_TW"]),
        np.nan,
    )
    merged_e09.to_csv(out_dir / "cross_benchmark_e09.csv", index=False)

    # 2. Side-by-side distribution comparison (normalized / dimensionless quantities only)
    keep_metrics = ["I_empty_norm", "I_neutral_norm", "I_fmt_delta", "H_C_mean", "H_C_p90", "H_A_mean", "D_self_mean"]
    sw_s = sw_sum[sw_sum["metric"].isin(keep_metrics)][["metric", "mean", "p50", "pos_frac"]]
    tw_s = tw_sum[tw_sum["metric"].isin(keep_metrics)][["metric", "mean", "p50", "pos_frac"]]
    merged_dist = sw_s.merge(tw_s, on="metric", suffixes=("_SW", "_TW"), how="outer")
    merged_dist.to_csv(out_dir / "cross_benchmark_distributions.csv", index=False)

    # 3. Side-by-side overview figure
    key_pairs = [
        ("I_empty_norm", "G3_strict"),
        ("I_empty_norm", "terminal_success"),
        ("I_empty_norm", "action_valid"),
        ("H_C_mean", "G3_strict"),
        ("H_C_mean", "terminal_success"),
        ("H_C_mean", "action_valid"),
        ("H_C_p90", "G3_strict"),
        ("H_A_mean", "G3_strict"),
    ]
    plot_rows = []
    for x, y in key_pairs:
        m = merged_e09[(merged_e09["x"] == x) & (merged_e09["y"] == y)]
        if not m.empty:
            r = m.iloc[0]
            plot_rows.append(
                {
                    "pair": f"{x} vs {y}",
                    "SW": r["spearman_pooled_SW"],
                    "TW": r["spearman_pooled_TW"],
                }
            )
    pdf = pd.DataFrame(plot_rows)
    if not pdf.empty:
        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        y_pos = np.arange(len(pdf))
        h = 0.36
        ax.barh(y_pos - h / 2, pdf["SW"].fillna(0), height=h, label="ScienceWorld", color="#1E88E5")
        ax.barh(y_pos + h / 2, pdf["TW"].fillna(0), height=h, label="TextWorldExpress", color="#43A047")
        ax.axvline(0, color="black", ls="--", lw=1.0)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pdf["pair"], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Episode-Clustered Spearman rho")
        ax.set_title("Cross-Benchmark Comparison: ScienceWorld vs TextWorldExpress (Qwen/Qwen3-8B)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "Fig_cross_benchmark_overview.png", dpi=300)
        plt.close(fig)

    # 4. Markdown summary
    consistent = merged_e09[merged_e09["same_sign"] == True]  # noqa: E712
    divergent = merged_e09[merged_e09["same_sign"] == False]  # noqa: E712
    lines = [
        "# 跨 Benchmark 对照汇总：ScienceWorld vs TextWorldExpress（Qwen/Qwen3-8B）",
        "",
        "> 遵循统一口径：不直接比较两边原始 Reward 绝对数值，只对比各 Bench 内的无量纲指标分布与相关性规律。",
        "",
        "## 1. 核心无量纲指标分布对比",
        "",
        _df_to_md(merged_dist),
        "",
        "## 2. E09 相关性跨 Bench 逐项对照",
        "",
        _df_to_md(
            merged_e09[
                ["x", "y", "spearman_pooled_SW", "ci_low_SW", "ci_high_SW", "spearman_pooled_TW", "ci_low_TW", "ci_high_TW", "same_sign"]
            ]
        ),
        "",
        f"## 3. 方向一致性统计",
        f"- 两个 Benchmark 相关方向一致的指标对：`{len(consistent)}` 对",
        f"- 两个 Benchmark 相关方向相反（仅单环境成立）的指标对：`{len(divergent)}` 对",
        "",
        "对照总览图已保存至：`Fig_cross_benchmark_overview.png`。",
    ]
    (out_dir / "cross_benchmark_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[Cross-Benchmark Complete] Saved comparison report & figure to {out_dir}")


if __name__ == "__main__":
    main()

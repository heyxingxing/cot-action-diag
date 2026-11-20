#!/usr/bin/env python3
"""
P1 E14 语义一致性人工抽检工具：
1. sample 子命令：按 (成/败 × I_t 分层/非法动作) 分层抽取 80 turns，导出“隐藏 I_t 和熵的盲标 CSV”供人工打分；
2. evaluate 子命令：将填完的 CSV 与 turn_metrics.jsonl 合并，计算 I_t / H_C 与人工标签的 AUROC & 相关性并出图。
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
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import ensure_dir, load_jsonl


def cmd_sample(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    out_csv = Path(args.out_csv)
    ensure_dir(out_csv.parent)

    turns = load_jsonl(data_dir / "turns.jsonl")
    metrics = load_jsonl(data_dir / "turn_metrics.jsonl")
    t_df = pd.DataFrame(turns)[
        ["episode_id", "turn_id", "benchmark", "task_id", "observation", "cot", "action", "next_observation", "action_valid"]
    ]
    m_df = pd.DataFrame(metrics)[
        ["episode_id", "turn_id", "I_empty_norm", "I_neutral_norm", "H_C_mean", "A_turn_proxy", "terminal_success"]
    ]
    df = t_df.merge(m_df, on=["episode_id", "turn_id"], how="inner").dropna(subset=["I_empty_norm"])
    if df.empty:
        raise ValueError("No valid scored turns found to sample for E14.")

    # Stratified buckets covering high positive, near-zero, negative I_t, and invalid actions
    q25, q75 = df["I_empty_norm"].quantile(0.25), df["I_empty_norm"].quantile(0.75)

    def _bucket(row: pd.Series) -> str:
        if not row["action_valid"]:
            return "invalid_action"
        if row["I_empty_norm"] < 0:
            return "neg_I_fail" if not row["terminal_success"] else "neg_I_succ"
        if row["I_empty_norm"] >= q75:
            return "high_I_succ" if row["terminal_success"] else "high_I_fail"
        return "mid_I"

    df["stratum"] = df.apply(_bucket, axis=1)
    rng = np.random.default_rng(args.seed)
    n_target = min(args.n_samples, len(df))

    buckets = list(df["stratum"].unique())
    per_bucket = max(1, n_target // len(buckets))
    chosen_idx = []
    for b in buckets:
        b_idx = df[df["stratum"] == b].index.to_numpy()
        take = min(len(b_idx), per_bucket)
        chosen_idx.extend(rng.choice(b_idx, size=take, replace=False).tolist())

    # Top up remaining quota randomly
    remaining = list(set(df.index.tolist()) - set(chosen_idx))
    if len(chosen_idx) < n_target and remaining:
        need = min(n_target - len(chosen_idx), len(remaining))
        chosen_idx.extend(rng.choice(remaining, size=need, replace=False).tolist())

    sampled = df.loc[chosen_idx].sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    # Blind CSV: hide I_t / H_C so human annotator is unbiased!
    blind = sampled[
        ["episode_id", "turn_id", "benchmark", "task_id", "observation", "cot", "action", "next_observation"]
    ].copy()
    blind["cot_explicit_intent"] = ""      # 1=明确指出下一步动作意图, 0=否
    blind["action_contradicts_cot"] = ""   # 1=Action与CoT自相矛盾, 0=一致
    blind["cot_too_vague"] = ""            # 1=CoT过于笼统/废话, 0=具体
    blind["notes"] = ""
    blind.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[E14 Sample] Exported {len(blind)} blind turns to {out_csv} (strata: {sampled['stratum'].value_counts().to_dict()})")


def cmd_evaluate(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    labeled = pd.read_csv(args.labeled_csv)
    metrics = pd.DataFrame(load_jsonl(data_dir / "turn_metrics.jsonl"))
    df = labeled.merge(metrics, on=["episode_id", "turn_id"], how="inner")
    df["C_proxy"] = df["A_turn_proxy"] * df["I_empty_norm"]

    label_cols = ["cot_explicit_intent", "action_contradicts_cot", "cot_too_vague"]
    score_cols = ["I_empty_norm", "I_neutral_norm", "H_C_mean", "C_proxy"]

    rows = []
    for lc in label_cols:
        sub = df.dropna(subset=[lc]).copy()
        sub[lc] = pd.to_numeric(sub[lc], errors="coerce")
        sub = sub.dropna(subset=[lc])
        y = sub[lc].astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            continue
        for sc in score_cols:
            s2 = sub.dropna(subset=[sc])
            y2 = s2[lc].astype(int).to_numpy()
            x2 = s2[sc].to_numpy(dtype=float)
            if len(np.unique(y2)) < 2 or np.std(x2) < 1e-12:
                continue
            auc = roc_auc_score(y2, x2)
            r, p = stats.pointbiserialr(y2, x2)
            rows.append(
                {
                    "human_label": lc,
                    "metric": sc,
                    "n": len(y2),
                    "pos_rate": round(float(y2.mean()), 3),
                    "auroc": round(float(auc), 4),
                    "point_biserial_r": round(float(r), 4),
                    "p_value": round(float(p), 5),
                }
            )

    res_df = pd.DataFrame(rows)
    out_dir = ensure_dir(args.out_dir)
    res_df.to_csv(out_dir / "e14_alignment.csv", index=False)
    print("\n[E14 Alignment Results]")
    print(res_df.to_string(index=False) if not res_df.empty else "No valid binary labels found.")

    if not res_df.empty:
        piv = res_df.pivot(index="human_label", columns="metric", values="auroc")
        ax = piv.plot(kind="bar", figsize=(9, 4.5))
        ax.axhline(0.5, color="black", ls="--", lw=1.0)
        ax.set_ylabel("AUROC vs Human Semantic Label")
        ax.set_title(f"E14: Metric Alignment with Human Audit (n={len(df)})")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(out_dir / "Fig10b_E14_semantic_alignment.png", dpi=300)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="E14 Manual Semantic Consistency Audit Helper.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_s = sub.add_parser("sample", help="Export stratified blind CSV for manual labeling.")
    p_s.add_argument("--data-dir", required=True)
    p_s.add_argument("--out-csv", required=True)
    p_s.add_argument("--n-samples", type=int, default=80)
    p_s.add_argument("--seed", type=int, default=42)
    p_s.set_defaults(func=cmd_sample)

    p_e = sub.add_parser("evaluate", help="Evaluate labeled CSV against I_t and H_C.")
    p_e.add_argument("--data-dir", required=True)
    p_e.add_argument("--labeled-csv", required=True)
    p_e.add_argument("--out-dir", required=True)
    p_e.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

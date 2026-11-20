#!/usr/bin/env python3
"""
Stage 3: 统一统计分析、10 张标准图生成与 E12 结论自动汇总脚本
覆盖：E04, E05, E06, E07, E08, E09(Episode聚类), E10(无泄露联合分析), E11(零奖励5类),
     E12(统一交付), E13(空/中性基线对比), E15(失败倒计时), E16(控制变量回归),
     E17(CoT位置熵), E18(Teacher Gap + 负负得正比例检验)

A(ScienceWorld) 与 B(TextWorldExpress) 跑完全相同的本脚本，保证口径 100% 一致。
图表标签采用英文防止 Linux 无中文字体乱码；生成的 report_summary.md 为中文可读报告。
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_protocol
from src.io_utils import ensure_dir, load_jsonl

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sns.set_theme(style="whitegrid", context="notebook", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "axes.titlesize": 11, "axes.labelsize": 10})


def _spearman_safe(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 4 or np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return np.nan
    r, _ = stats.spearmanr(x, y)
    return float(r) if np.isfinite(r) else np.nan


def _pearson_safe(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 4 or np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return np.nan
    r, _ = stats.pearsonr(x, y)
    return float(r) if np.isfinite(r) else np.nan


def cluster_bootstrap_spearman(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Episode-level cluster bootstrap for Spearman rho + within-episode average rho."""
    sub = df[["episode_id", x_col, y_col]].dropna()
    n_turns = len(sub)
    ep_ids = sub["episode_id"].unique()
    n_eps = len(ep_ids)
    if n_turns < 6 or n_eps < 3:
        return {
            "x": x_col,
            "y": y_col,
            "n_episodes": n_eps,
            "n_turns": n_turns,
            "spearman_pooled": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "pearson_pooled": np.nan,
            "within_ep_spearman_mean": np.nan,
            "within_ep_valid_eps": 0,
        }

    pooled_rho = _spearman_safe(sub[x_col].to_numpy(), sub[y_col].to_numpy())
    pooled_r = _pearson_safe(sub[x_col].to_numpy(), sub[y_col].to_numpy())

    grouped = {eid: (g[x_col].to_numpy(), g[y_col].to_numpy()) for eid, g in sub.groupby("episode_id")}
    within_rhos = []
    for eid, (gx, gy) in grouped.items():
        r = _spearman_safe(gx, gy)
        if np.isfinite(r):
            within_rhos.append(r)

    rng = np.random.default_rng(seed)
    boot_vals = []
    ep_list = list(grouped.keys())
    for _ in range(n_resamples):
        sampled_eps = rng.choice(ep_list, size=n_eps, replace=True)
        bx = np.concatenate([grouped[e][0] for e in sampled_eps])
        by = np.concatenate([grouped[e][1] for e in sampled_eps])
        r = _spearman_safe(bx, by)
        if np.isfinite(r):
            boot_vals.append(r)

    ci_low = float(np.quantile(boot_vals, 0.025)) if boot_vals else np.nan
    ci_high = float(np.quantile(boot_vals, 0.975)) if boot_vals else np.nan
    return {
        "x": x_col,
        "y": y_col,
        "n_episodes": n_eps,
        "n_turns": n_turns,
        "spearman_pooled": round(pooled_rho, 4) if np.isfinite(pooled_rho) else np.nan,
        "ci_low": round(ci_low, 4) if np.isfinite(ci_low) else np.nan,
        "ci_high": round(ci_high, 4) if np.isfinite(ci_high) else np.nan,
        "pearson_pooled": round(pooled_r, 4) if np.isfinite(pooled_r) else np.nan,
        "within_ep_spearman_mean": round(float(np.mean(within_rhos)), 4) if within_rhos else np.nan,
        "within_ep_valid_eps": len(within_rhos),
    }


def run_leak_free_e10_cv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Leak-free GroupKFold CV comparing:
      (1) I_t only
      (2) r_t only
      (3) r_t + I_t + (I_t * r_t)
    when predicting strictly future return G3_strict (R^2) and terminal_success (AUROC).
    """
    rows = []
    sub = df[["episode_id", "I_empty_norm", "reward_raw", "G3_strict", "terminal_success"]].dropna().copy()
    sub["I_x_r"] = sub["I_empty_norm"] * sub["reward_raw"]
    n_eps = sub["episode_id"].nunique()
    n_splits = min(5, n_eps)
    if n_splits < 2 or len(sub) < 10:
        return pd.DataFrame(rows)

    gkf = GroupKFold(n_splits=n_splits)
    feature_sets = {
        "I_t_only": ["I_empty_norm"],
        "r_t_only": ["reward_raw"],
        "r_t + I_t + I_t*r_t": ["reward_raw", "I_empty_norm", "I_x_r"],
    }

    # Task A: predict G3_strict (regression R^2)
    y_reg = sub["G3_strict"].to_numpy(dtype=float)
    groups = sub["episode_id"].to_numpy()
    for name, cols in feature_sets.items():
        X = sub[cols].to_numpy(dtype=float)
        fold_scores = []
        for tr, te in gkf.split(X, y_reg, groups):
            if np.std(y_reg[te]) < 1e-12:
                continue
            model = Ridge(alpha=1.0).fit(X[tr], y_reg[tr])
            preds = model.predict(X[te])
            fold_scores.append(r2_score(y_reg[te], preds))
        rows.append(
            {
                "target": "G3_strict (R2)",
                "feature_set": name,
                "cv_mean": round(float(np.mean(fold_scores)), 4) if fold_scores else np.nan,
                "cv_std": round(float(np.std(fold_scores)), 4) if fold_scores else np.nan,
                "n_folds_used": len(fold_scores),
            }
        )

    # Task B: predict terminal_success (classification AUROC)
    y_cls = sub["terminal_success"].astype(int).to_numpy()
    if len(np.unique(y_cls)) >= 2:
        for name, cols in feature_sets.items():
            X = sub[cols].to_numpy(dtype=float)
            fold_aucs = []
            for tr, te in gkf.split(X, y_cls, groups):
                if len(np.unique(y_cls[tr])) < 2 or len(np.unique(y_cls[te])) < 2:
                    continue
                clf = LogisticRegression(max_iter=500).fit(X[tr], y_cls[tr])
                probs = clf.predict_proba(X[te])[:, 1]
                fold_aucs.append(roc_auc_score(y_cls[te], probs))
            rows.append(
                {
                    "target": "terminal_success (AUROC)",
                    "feature_set": name,
                    "cv_mean": round(float(np.mean(fold_aucs)), 4) if fold_aucs else np.nan,
                    "cv_std": round(float(np.std(fold_aucs)), 4) if fold_aucs else np.nan,
                    "n_folds_used": len(fold_aucs),
                }
            )
    return pd.DataFrame(rows)


def _df_to_md(df: pd.DataFrame) -> str:
    """Zero-dependency GitHub-flavored Markdown table renderer (no tabulate required)."""
    if df.empty:
        return "_(empty)_"
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for _, row in df.iterrows():
        vals = [str(row[c]) if pd.notna(row[c]) else "" for c in df.columns]
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body])


def run_e16_controlled_regression(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pure NumPy/SciPy OLS with Cameron-Miller CR1 episode-clustered robust standard errors.
    Controls for task_id fixed effects, turn_id, cot_token_len, action_token_len, action_valid.
    (Zero statsmodels dependency -> immune to scipy/statsmodels version conflicts!)
    """
    results = []
    cols = [
        "episode_id",
        "task_id",
        "turn_id",
        "cot_token_len",
        "action_token_len",
        "action_valid",
        "I_empty_norm",
        "H_C_mean",
        "H_C_p90",
        "G3_strict",
        "final_score_norm",
        "reward_raw",
    ]
    sub = df[[c for c in cols if c in df.columns]].dropna().copy()
    sub["action_valid_int"] = sub["action_valid"].astype(float)
    n_obs = len(sub)
    clusters = sub["episode_id"].to_numpy()
    unique_clusters = np.unique(clusters)
    G = len(unique_clusters)
    if n_obs < 15 or G < 4:
        return pd.DataFrame(results)

    num_terms = [
        "I_empty_norm",
        "H_C_mean",
        "H_C_p90",
        "turn_id",
        "cot_token_len",
        "action_token_len",
        "action_valid_int",
    ]
    # Keep only numeric terms with non-zero variance
    active_terms = [c for c in num_terms if c in sub.columns and sub[c].std() > 1e-12]
    dummies = pd.get_dummies(sub["task_id"], prefix="task", drop_first=True, dtype=float)
    X_df = pd.concat(
        [pd.Series(1.0, index=sub.index, name="intercept"), sub[active_terms].astype(float), dummies],
        axis=1,
    )
    col_names = list(X_df.columns)
    X = X_df.to_numpy(dtype=float)
    p = X.shape[1]
    if n_obs <= p + 1:
        return pd.DataFrame(results)

    bread = np.linalg.pinv(X.T @ X)
    dof_scale = (G / (G - 1.0)) * ((n_obs - 1.0) / max(1.0, n_obs - p))

    for outcome in ["reward_raw", "G3_strict", "final_score_norm"]:
        if outcome not in sub.columns or sub[outcome].std() < 1e-12:
            continue
        y = sub[outcome].to_numpy(dtype=float)
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        ss_res = float(np.sum(resid ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan

        meat = np.zeros((p, p), dtype=float)
        for g in unique_clusters:
            mask = (clusters == g)
            u_g = X[mask].T @ resid[mask]
            meat += np.outer(u_g, u_g)

        cov = dof_scale * (bread @ meat @ bread)
        se = np.sqrt(np.maximum(np.diag(cov), 1e-18))
        t_vals = beta / se
        p_vals = 2.0 * stats.t.sf(np.abs(t_vals), df=max(1, G - 1))

        for term in active_terms:
            j = col_names.index(term)
            results.append(
                {
                    "outcome": outcome,
                    "term": term,
                    "coef": round(float(beta[j]), 5),
                    "se_clustered": round(float(se[j]), 5),
                    "p_value": round(float(p_vals[j]), 5),
                    "r2": round(float(r2), 4),
                    "n_turns": int(n_obs),
                    "n_episodes": int(G),
                }
            )
    return pd.DataFrame(results)


# ==============================================================================
# Figure Generators (Fig 01 ~ Fig 10)
# ==============================================================================


def plot_fig01_e04_bridge_dist(df: pd.DataFrame, out_path: Path, bench: str) -> None:
    sub = df.dropna(subset=["I_empty_norm"])
    n_ep, n_t = sub["episode_id"].nunique(), len(sub)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: histogram of I_empty & I_neutral
    ax = axes[0]
    ax.hist(sub["I_empty_norm"], bins=35, alpha=0.65, label=f"I_empty (mean={sub['I_empty_norm'].mean():.3f})")
    if "I_neutral_norm" in sub and sub["I_neutral_norm"].notna().any():
        neu = sub["I_neutral_norm"].dropna()
        ax.hist(neu, bins=35, alpha=0.55, label=f"I_neutral (mean={neu.mean():.3f})")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    pos_pct = (sub["I_empty_norm"] > 0).mean() * 100
    ax.set_title(f"[{bench}] E04: Bridge Function I_t Distribution (pos={pos_pct:.1f}%)")
    ax.set_xlabel("I_t (mean log-prob shift per action token)")
    ax.set_ylabel("Turn Count")
    ax.legend()

    # Right: success vs failure boxplot
    ax = axes[1]
    sub_plot = sub.copy()
    sub_plot["Outcome"] = sub_plot["terminal_success"].map({True: "Success", False: "Failure"})
    sns.boxplot(data=sub_plot, x="Outcome", y="I_empty_norm", ax=ax, width=0.45)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title(f"I_empty by Episode Outcome (N_ep={n_ep}, N_turn={n_t})")
    ax.set_ylabel("I_empty_norm")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_fig02_e04_e16_stratified(df: pd.DataFrame, out_path: Path, bench: str) -> None:
    sub = df.dropna(subset=["I_empty_norm"]).copy()
    n_ep, n_t = sub["episode_id"].nunique(), len(sub)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1. By difficulty tier / task_id
    sns.boxplot(data=sub, x="task_id", y="I_empty_norm", ax=axes[0])
    axes[0].axhline(0, color="black", ls="--", lw=1)
    axes[0].set_title("I_t by Task Type")
    axes[0].tick_params(axis="x", rotation=20)

    # 2. By Turn depth bin
    sub["turn_bin"] = pd.cut(sub["turn_id"], bins=[-1, 4, 12, 24, 999], labels=["0-4", "5-12", "13-24", "25+"])
    sns.boxplot(data=sub, x="turn_bin", y="I_empty_norm", ax=axes[1])
    axes[1].axhline(0, color="black", ls="--", lw=1)
    axes[1].set_title("I_t by Turn Depth")
    axes[1].set_xlabel("Turn Index Bin")

    # 3. By Action token length bin
    sub["act_len_bin"] = pd.cut(
        sub["action_token_len"], bins=[0, 2, 4, 8, 999], labels=["1-2", "3-4", "5-8", "9+"]
    )
    sns.boxplot(data=sub, x="act_len_bin", y="I_empty_norm", ax=axes[2])
    axes[2].axhline(0, color="black", ls="--", lw=1)
    axes[2].set_title("I_t by Action Token Length")
    axes[2].set_xlabel("Action Token Length")

    fig.suptitle(f"[{bench}] E04 & E16: Stratified I_t Analysis (N_ep={n_ep}, N_turn={n_t})", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fig03_e05_e06_entropy(df: pd.DataFrame, out_path: Path, bench: str) -> None:
    sub = df.dropna(subset=["H_C_mean", "H_A_mean"])
    n_ep, n_t = sub["episode_id"].nunique(), len(sub)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(sub["H_C_mean"], bins=30, alpha=0.6, label=f"H_C_mean ({sub['H_C_mean'].mean():.3f})")
    axes[0].hist(sub["H_C_p90"], bins=30, alpha=0.45, label=f"H_C_p90 ({sub['H_C_p90'].mean():.3f})")
    axes[0].hist(sub["H_A_mean"], bins=30, alpha=0.55, label=f"H_A_mean ({sub['H_A_mean'].mean():.3f})")
    axes[0].set_title(f"[{bench}] E05/E06: Exact CoT vs Action Entropy (nats)")
    axes[0].set_xlabel("Entropy (nats)")
    axes[0].set_ylabel("Turn Count")
    axes[0].legend()

    rho = _spearman_safe(sub["H_C_mean"].to_numpy(), sub["H_A_mean"].to_numpy())
    axes[1].scatter(sub["H_C_mean"], sub["H_A_mean"], alpha=0.45, s=18)
    axes[1].set_title(f"H_C_mean vs H_A_mean (Spearman={rho:.3f}, N_ep={n_ep}, N_turn={n_t})")
    axes[1].set_xlabel("CoT Mean Entropy (nats)")
    axes[1].set_ylabel("Action Mean Entropy (nats)")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_fig04_e07_e11_rewards(df: pd.DataFrame, out_path: Path, bench: str) -> None:
    n_ep, n_t = df["episode_id"].nunique(), len(df)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # 1. E07 Reward sign breakdown
    r_signs = pd.Series(
        np.where(df["reward_raw"] > 0, "Positive", np.where(df["reward_raw"] < 0, "Negative", "Zero"))
    )
    counts = r_signs.value_counts().reindex(["Positive", "Zero", "Negative"]).fillna(0)
    pcts = counts / max(1, len(df)) * 100
    bars = axes[0].bar(pcts.index, pcts.values, color=["#4CAF50", "#90A4AE", "#E53935"])
    for b, cnt in zip(bars, counts.values):
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.1f}%\n(n={int(cnt)})", ha="center", fontsize=8.5)
    axes[0].set_ylim(0, max(105, pcts.max() + 15))
    axes[0].set_title("E07: Immediate Reward Coverage")
    axes[0].set_ylabel("Share of Turns (%)")

    # 2 & 3. E11 Zero-reward 5 categories -> I_t and H_C_mean
    zero_df = df[df["reward_raw"] == 0].copy()
    order = ["effective_setup", "aimless_explore", "repeat_noop", "illegal_action", "parse_fail"]
    present = [c for c in order if c in set(zero_df["zero_reward_category"])]
    if present:
        sns.boxplot(data=zero_df, x="zero_reward_category", y="I_empty_norm", order=present, ax=axes[1])
        axes[1].axhline(0, color="black", ls="--", lw=1)
        axes[1].set_title(f"E11: I_t across r_t=0 Categories (n={len(zero_df)})")
        axes[1].tick_params(axis="x", rotation=22)

        sns.boxplot(data=zero_df, x="zero_reward_category", y="H_C_mean", order=present, ax=axes[2])
        axes[2].set_title("E11: CoT Entropy across r_t=0 Categories")
        axes[2].tick_params(axis="x", rotation=22)
    else:
        axes[1].text(0.5, 0.5, "No r_t=0 turns", ha="center", va="center")
        axes[2].text(0.5, 0.5, "No r_t=0 turns", ha="center", va="center")

    fig.suptitle(f"[{bench}] E07 & E11: Reward Sparsity & Zero-Reward Dissection (N_ep={n_ep}, N_turn={n_t})", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fig05_e09_forest(e09_df: pd.DataFrame, out_path: Path, bench: str) -> None:
    sub = e09_df.dropna(subset=["spearman_pooled"]).copy()
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.42 * len(sub))))
    if sub.empty:
        ax.text(0.5, 0.5, "Insufficient variance for E09 correlations", ha="center", va="center")
        fig.savefig(out_path)
        plt.close(fig)
        return

    labels = [f"{r['x']}  vs  {r['y']} (n={int(r['n_turns'])})" for _, r in sub.iterrows()]
    y_pos = np.arange(len(sub))
    rhos = sub["spearman_pooled"].to_numpy()
    err_low = np.clip(rhos - sub["ci_low"].to_numpy(), 0, None)
    err_high = np.clip(sub["ci_high"].to_numpy() - rhos, 0, None)

    ax.errorbar(rhos, y_pos, xerr=[err_low, err_high], fmt="o", capsize=4, color="#1976D2", ecolor="#64B5F6", lw=1.8)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman rho (95% Episode-Cluster Bootstrap CI)")
    n_ep = int(sub["n_episodes"].max())
    ax.set_title(f"[{bench}] E09: Metric-Outcome Correlations (Clustered by N_ep={n_ep})")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_fig06_e10_joint(df: pd.DataFrame, cv_df: pd.DataFrame, out_path: Path, bench: str) -> None:
    sub = df.dropna(subset=["I_empty_norm", "H_C_mean", "A_turn_proxy"]).copy()
    n_ep, n_t = sub["episode_id"].nunique(), len(sub)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # 1. 2x2 High/Low H_C x Pos/Neg I_t -> mean G3_strict
    med_h = sub.groupby("task_id")["H_C_mean"].transform("median")
    sub["H_group"] = np.where(sub["H_C_mean"] >= med_h, "High H_C", "Low H_C")
    sub["I_group"] = np.where(sub["I_empty_norm"] > 0, "I_t > 0", "I_t <= 0")
    piv = sub.pivot_table(index="H_group", columns="I_group", values="G3_strict", aggfunc="mean")
    cnt = sub.pivot_table(index="H_group", columns="I_group", values="G3_strict", aggfunc="count").fillna(0)
    annot = piv.copy().astype(object)
    for r in piv.index:
        for c in piv.columns:
            v = piv.loc[r, c]
            n = int(cnt.loc[r, c]) if (r in cnt.index and c in cnt.columns) else 0
            annot.loc[r, c] = f"{v:.3f}\n(n={n})" if pd.notna(v) else "N/A"
    sns.heatmap(piv, annot=annot, fmt="", cmap="YlGnBu", ax=axes[0], cbar_kws={"label": "Mean G3_strict"})
    axes[0].set_title("E10: 2x2 (H_C x I_t) -> G3_strict")

    # 2. Four quadrants of (A_turn_proxy >< 0) x (I_t >< 0) -> action_valid rate & count
    sub["Quad"] = (
        np.where(sub["A_turn_proxy"] > 0, "A+", "A-")
        + " / "
        + np.where(sub["I_empty_norm"] > 0, "I+", "I-")
    )
    q_order = ["A+ / I+", "A+ / I-", "A- / I+", "A- / I-"]
    q_stats = sub.groupby("Quad").agg(valid_rate=("action_valid", "mean"), count=("action_valid", "count")).reindex(q_order)
    bars = axes[1].bar(q_stats.index, q_stats["valid_rate"].fillna(0) * 100, color=["#2E7D32", "#FB8C00", "#C62828", "#6A1B9A"])
    for b, cnt_v in zip(bars, q_stats["count"].fillna(0)):
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.1f}%\n(n={int(cnt_v)})", ha="center", fontsize=8.5)
    axes[1].set_ylim(0, 115)
    axes[1].set_ylabel("Valid Action Rate (%)")
    axes[1].set_title("E10: 4-Quadrant (A_proxy x I_t) Dissection")

    # 3. Leak-free CV comparison (I_t only vs r_t only vs r_t + I_t + I_t*r_t)
    if not cv_df.empty:
        sns.barplot(data=cv_df, x="target", y="cv_mean", hue="feature_set", ax=axes[2])
        axes[2].set_title("E10: Leak-Free GroupKFold CV")
        axes[2].set_ylabel("CV Score (R2 / AUROC)")
        axes[2].legend(fontsize=8)
    else:
        axes[2].text(0.5, 0.5, "Insufficient folds for CV", ha="center", va="center")

    fig.suptitle(f"[{bench}] E10: Joint Metric & Quadrant Analysis (N_ep={n_ep}, N_turn={n_t})", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fig07_e13_robustness(df: pd.DataFrame, out_path: Path, bench: str) -> None:
    sub = df.dropna(subset=["I_empty_norm", "I_neutral_norm"])
    n_ep, n_t = sub["episode_id"].nunique(), len(sub)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    if sub.empty:
        axes[0].text(0.5, 0.5, "Missing neutral baseline", ha="center", va="center")
        fig.savefig(out_path)
        plt.close(fig)
        return

    x = sub["I_empty_norm"].to_numpy()
    y = sub["I_neutral_norm"].to_numpy()
    rho = _spearman_safe(x, y)
    r = _pearson_safe(x, y)
    sign_agree = float(np.mean((x > 0) == (y > 0))) * 100

    axes[0].scatter(x, y, alpha=0.45, s=18)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=1.1, label="y = x")
    axes[0].axhline(0, color="gray", ls=":", lw=0.9)
    axes[0].axvline(0, color="gray", ls=":", lw=0.9)
    axes[0].set_title(f"E13: Empty vs Neutral No-CoT (rho={rho:.3f}, r={r:.3f}, sign={sign_agree:.1f}%)")
    axes[0].set_xlabel("I_empty_norm")
    axes[0].set_ylabel("I_neutral_norm")
    axes[0].legend()

    diff = x - y
    axes[1].hist(diff, bins=35, color="#FF7043", alpha=0.75)
    axes[1].axvline(0.0, color="black", ls="--", lw=1.2)
    axes[1].axvline(diff.mean(), color="navy", ls="-", lw=1.5, label=f"mean delta={diff.mean():.3f}")
    axes[1].set_title(f"Format Shift Delta = I_empty - I_neutral (N_ep={n_ep}, N_turn={n_t})")
    axes[1].set_xlabel("Delta I_t")
    axes[1].set_ylabel("Turn Count")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_fig08_e15_failure_countdown(df: pd.DataFrame, out_path: Path, bench: str) -> None:
    fail_df = df[~df["terminal_success"]].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    if fail_df.empty:
        axes[0].text(0.5, 0.5, "No failed episodes to plot", ha="center", va="center")
        fig.savefig(out_path)
        plt.close(fig)
        return

    max_turn = fail_df.groupby("episode_id")["turn_id"].transform("max")
    fail_df["steps_to_end"] = fail_df["turn_id"] - max_turn  # ..., -4, -3, -2, -1, 0
    win = fail_df[fail_df["steps_to_end"] >= -5]
    n_ep = win["episode_id"].nunique()

    agg = win.groupby("steps_to_end").agg(
        hc_mean=("H_C_mean", "mean"),
        hc_sem=("H_C_mean", "sem"),
        hc_p90=("H_C_p90", "mean"),
        i_mean=("I_empty_norm", "mean"),
        i_sem=("I_empty_norm", "sem"),
        n=("turn_id", "count"),
    )

    xs = agg.index.to_numpy()
    axes[0].errorbar(xs, agg["hc_mean"], yerr=agg["hc_sem"].fillna(0), marker="o", label="H_C_mean")
    axes[0].plot(xs, agg["hc_p90"], marker="s", linestyle="--", label="H_C_p90")
    axes[0].set_title(f"[{bench}] E15: CoT Entropy Before Failure (N_fail_ep={n_ep})")
    axes[0].set_xlabel("Steps Relative to Episode End (0 = final step)")
    axes[0].set_ylabel("CoT Entropy (nats)")
    axes[0].legend()

    axes[1].errorbar(xs, agg["i_mean"], yerr=agg["i_sem"].fillna(0), marker="o", color="#D81B60", label="I_empty_norm")
    axes[1].axhline(0.0, color="black", ls="--", lw=1.0)
    axes[1].set_title("E15: Bridge Function I_t Before Failure")
    axes[1].set_xlabel("Steps Relative to Episode End (0 = final step)")
    axes[1].set_ylabel("I_empty_norm")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_fig09_e17_cot_position(df: pd.DataFrame, tok_df: pd.DataFrame, out_path: Path, bench: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    cot_tok = tok_df[tok_df["segment"] == "cot"].copy() if not tok_df.empty else pd.DataFrame()

    if not cot_tok.empty:
        max_pos = cot_tok.groupby(["episode_id", "turn_id"])["token_pos"].transform("max").clip(lower=1)
        cot_tok["rel_pos"] = cot_tok["token_pos"] / max_pos
        cot_tok["pos_bin"] = pd.cut(cot_tok["rel_pos"], bins=np.linspace(0, 1, 11), include_lowest=True, labels=range(1, 11))
        merged = cot_tok.merge(df[["episode_id", "turn_id", "action_valid", "terminal_success"]], on=["episode_id", "turn_id"], how="inner")

        for valid_flag, lbl, col in [(True, "Valid Action", "#2E7D32"), (False, "Invalid Action", "#C62828")]:
            g = merged[merged["action_valid"] == valid_flag].groupby("pos_bin", observed=False)["entropy_nats"].agg(["mean", "sem"])
            if g["mean"].notna().any():
                axes[0].errorbar(range(1, 11), g["mean"], yerr=g["sem"].fillna(0), marker="o", label=lbl, color=col)
        axes[0].set_xticks(range(1, 11))
        axes[0].set_xticklabels([f"{i*10}%" for i in range(1, 11)], fontsize=8.5)
        axes[0].set_xlabel("Relative Position Inside CoT (Head -> Tail)")
        axes[0].set_ylabel("Token Entropy (nats)")
        axes[0].set_title(f"[{bench}] E17: CoT Positional Entropy Profile")
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "No token_stats.parquet CoT data", ha="center", va="center")

    # Right: compare H_C_mean / H_C_p90 / H_C_max / head / mid / tail correlation with action_valid & G3_strict
    cand_cols = ["H_C_mean", "H_C_p90", "H_C_max", "H_C_head_mean", "H_C_mid_mean", "H_C_tail_mean"]
    sub = df.dropna(subset=["G3_strict", "action_valid"])
    r_valid = [_spearman_safe(sub[c].dropna().to_numpy(), sub.loc[sub[c].notna(), "action_valid"].astype(float).to_numpy()) for c in cand_cols]
    r_g3 = [_spearman_safe(sub[c].dropna().to_numpy(), sub.loc[sub[c].notna(), "G3_strict"].to_numpy()) for c in cand_cols]
    x = np.arange(len(cand_cols))
    w = 0.36
    axes[1].bar(x - w / 2, [0 if np.isnan(v) else v for v in r_valid], width=w, label="vs action_valid")
    axes[1].bar(x + w / 2, [0 if np.isnan(v) else v for v in r_g3], width=w, label="vs G3_strict")
    axes[1].axhline(0, color="black", ls="--", lw=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["mean", "p90", "max", "head", "mid", "tail"])
    axes[1].set_ylabel("Spearman rho")
    axes[1].set_title(f"E17: Summary vs Peak Entropy Predictive Power (N_turn={len(sub)})")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_fig10_e18_teacher_gap(df: pd.DataFrame, out_path: Path, bench: str) -> Dict[str, Any]:
    sub = df.dropna(subset=["D_self_mean"]).copy()
    stats_out: Dict[str, Any] = {}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    if sub.empty:
        axes[0].text(0.5, 0.5, "E18 Teacher Gap not computed (--skip-teacher)", ha="center", va="center")
        fig.savefig(out_path)
        plt.close(fig)
        return stats_out

    n_ep, n_t = sub["episode_id"].nunique(), len(sub)
    axes[0].hist(sub["D_self_mean"], bins=35, color="#5C6BC0", alpha=0.75)
    axes[0].axvline(0.0, color="black", ls="--", lw=1.2)
    axes[0].set_title(f"[{bench}] E18: State-OPD Self-Teacher Gap D_self (mean={sub['D_self_mean'].mean():.3f})")
    axes[0].set_xlabel("Turn-mean D_self = log pi_T(z|s_t) - log pi_S(z|h_t)")
    axes[0].set_ylabel("Turn Count")

    # Right: scatter vs H_C_mean + quantify double-negative conflict rate when C_t < 0
    sub["C_proxy"] = sub["A_turn_proxy"] * sub["I_empty_norm"]
    neg_c = sub[sub["C_proxy"] < 0]
    conflict_tok_frac = float(neg_c["D_self_neg_frac"].mean()) * 100 if len(neg_c) > 0 else np.nan
    stats_out["neg_C_turns"] = int(len(neg_c))
    stats_out["double_negative_token_pct"] = round(conflict_tok_frac, 1) if np.isfinite(conflict_tok_frac) else np.nan

    sc = axes[1].scatter(
        sub["H_C_mean"],
        sub["D_self_mean"],
        c=np.where(sub["C_proxy"] < 0, "#E53935", "#2E7D32"),
        alpha=0.55,
        s=18,
    )
    axes[1].axhline(0.0, color="black", ls="--", lw=1.0)
    rho_dh = _spearman_safe(sub["H_C_mean"].to_numpy(), sub["D_self_mean"].to_numpy())
    axes[1].set_title(
        f"D_self vs H_C (rho={rho_dh:.3f} | D<0 in C<0 turns: {conflict_tok_frac:.1f}%, N_ep={n_ep})"
        if np.isfinite(conflict_tok_frac)
        else f"D_self vs H_C (rho={rho_dh:.3f}, N_ep={n_ep}, N_turn={n_t})"
    )
    axes[1].set_xlabel("CoT Mean Entropy H_C_mean (nats)")
    axes[1].set_ylabel("Turn-mean Teacher Gap D_self")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return stats_out


def write_chinese_summary_report(
    bench: str,
    ep_df: pd.DataFrame,
    df: pd.DataFrame,
    turns_raw: List[Dict[str, Any]],
    e09_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    e16_df: pd.DataFrame,
    e18_stats: Dict[str, Any],
    out_path: Path,
) -> None:
    """Auto-generate concise E12 Chinese diagnostic report with real anomaly examples."""
    n_ep = len(ep_df)
    n_t = len(df)
    parse_rate = df["parse_ok"].mean() * 100
    valid_rate = df["action_valid"].mean() * 100
    succ_rate = ep_df["terminal_success"].mean() * 100
    mean_score = ep_df["final_score_norm"].mean()

    i_emp = df["I_empty_norm"].dropna()
    i_neu = df["I_neutral_norm"].dropna()
    hc = df["H_C_mean"].dropna()
    ha = df["H_A_mean"].dropna()

    both_i = df[["I_empty_norm", "I_neutral_norm"]].dropna()
    e13_rho = _spearman_safe(both_i["I_empty_norm"].to_numpy(), both_i["I_neutral_norm"].to_numpy())
    e13_sign = float(np.mean((both_i["I_empty_norm"] > 0) == (both_i["I_neutral_norm"] > 0))) * 100 if len(both_i) else np.nan

    # Extract 3 concrete anomaly examples for human inspection
    turns_map = {(t["episode_id"], int(t["turn_id"])): t for t in turns_raw}
    anomalies: List[str] = []

    def _fmt_case(title: str, row: pd.Series) -> str:
        raw = turns_map.get((row["episode_id"], int(row["turn_id"])), {})
        return (
            f"- **{title}** (`{row['episode_id']}`, turn={int(row['turn_id'])}, "
            f"`I_empty={row['I_empty_norm']:.3f}`, `H_C={row['H_C_mean']:.3f}`, "
            f"`r_t={row['reward_raw']}`, `valid={row['action_valid']}`, `succ={row['terminal_success']}`)\n"
            f"  - CoT: `{str(raw.get('cot', ''))[:160]}`\n"
            f"  - Action: `{raw.get('action', '')}`"
        )

    sub_valid = df.dropna(subset=["I_empty_norm", "H_C_mean", "A_turn_proxy"])
    # 1. Negative A & Negative I_t (Row 4 check!)
    r4 = sub_valid[(sub_valid["A_turn_proxy"] < 0) & (sub_valid["I_empty_norm"] < 0)].sort_values("I_empty_norm")
    if not r4.empty:
        anomalies.append(_fmt_case("负收益且 I_t < 0（原表第 4 行）", r4.iloc[0]))
    # 2. High I_t but invalid action or failed episode
    hi_bad = sub_valid[(~sub_valid["action_valid"]) | (~sub_valid["terminal_success"])].sort_values(
        "I_empty_norm", ascending=False
    )
    if not hi_bad.empty:
        anomalies.append(_fmt_case("高 I_t 但动作非法或终局失败", hi_bad.iloc[0]))
    # 3. Confident wrong (low H_C but invalid action)
    conf_err = sub_valid[~sub_valid["action_valid"]].sort_values("H_C_mean")
    if not conf_err.empty:
        anomalies.append(_fmt_case("低熵却做出非法动作（自信犯错）", conf_err.iloc[0]))

    lines = [
        f"# 诊断实验总结报告：{bench}（Qwen/Qwen3-8B）",
        "",
        "## 1. 样本规模与基础体检（E01 / E02 / E07）",
        f"- **总计**：`N_episodes = {n_ep}`，`N_turns = {n_t}`（平均 `{n_t / max(1, n_ep):.1f}` 步/局）",
        f"- **格式解析成功率 (`parse_ok`)**：`{parse_rate:.1f}%`",
        f"- **动作合法率 (`action_valid`)**：`{valid_rate:.1f}%`",
        f"- **终局成功率 (`terminal_success`)**：`{succ_rate:.1f}%`，归一化平均终局得分：`{mean_score:.3f}`",
        f"- **单步 Reward 占比**：正 `{(df['reward_raw']>0).mean()*100:.1f}%` / 零 `{(df['reward_raw']==0).mean()*100:.1f}%` / 负 `{(df['reward_raw']<0).mean()*100:.1f}%`",
        "",
        "## 2. 核心指标分布（E03–E06 & E13）",
        f"- **桥梁函数 `I_empty_norm`**：均值 `{i_emp.mean():.4f}`，P10/P50/P90 = `{i_emp.quantile(0.1):.3f} / {i_emp.quantile(0.5):.3f} / {i_emp.quantile(0.9):.3f}`，正值占比 `{(i_emp>0).mean()*100:.1f}%`" if len(i_emp) else "- `I_empty_norm` 无有效样本",
        f"- **中性基线 `I_neutral_norm`**：均值 `{i_neu.mean():.4f}`，正值占比 `{(i_neu>0).mean()*100:.1f}%`；与空基线 Spearman = `{e13_rho:.3f}`，正负号一致率 = `{e13_sign:.1f}%`" if len(i_neu) else "- `I_neutral_norm` 未计算",
        f"- **CoT 精确熵 `H_C_mean`**：均值 `{hc.mean():.4f} nats`，P90 = `{df['H_C_p90'].mean():.4f} nats`；对照 Action 熵 `H_A_mean` 均值 `{ha.mean():.4f} nats`" if len(hc) else "- 熵指标无有效样本",
        "",
        "## 3. E09 按 Episode 聚类核心相关性（节选）",
        "",
        _df_to_md(e09_df) if not e09_df.empty else "_无有效相关结果_",
        "",
        "## 4. E10 无泄露预测力对比（GroupKFold CV）",
        "",
        _df_to_md(cv_df) if not cv_df.empty else "_样本/折数不足，未运行 CV_",
        "",
        "## 5. E16 控制变量回归（Episode 聚类稳健标准误）",
        "",
        _df_to_md(e16_df) if not e16_df.empty else "_样本不足或方差为 0，未拟合回归_",
        "",
        "## 6. E18 Teacher Gap 与符号冲突检验",
        f"- 在 `C_proxy = A_turn_proxy * I_t < 0` 的 `{e18_stats.get('neg_C_turns', 0)}` 个负向 CoT Turn 中，同时出现 `D_self < 0` 的 token 平均占比为 **`{e18_stats.get('double_negative_token_pct', 'N/A')}%`**（直接衡量若误将负权重乘在 `D` 上发生“负负得正”的比例）。",
        "",
        "## 7. 典型异常案例摘录（供人工复核）",
        *(anomalies if anomalies else ["- 暂无可摘录的异常案例"]),
        "",
        "## 8. 交付图表清单（位于 `figures/`）",
        "- `Fig01_E04_bridge_dist.png` ~ `Fig10_E18_teacher_gap.png`（共 10 张，300 DPI）",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: Run E04-E18 statistical analysis & generate 10 figures.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "protocol.yaml"))
    parser.add_argument("--data-dir", required=True, help="Directory with episodes.jsonl, turns.jsonl, turn_metrics.jsonl.")
    parser.add_argument("--out-dir", required=True, help="Output directory for figures/, tables/, report_summary.md.")
    parser.add_argument("--bootstrap", type=int, default=None, help="Override bootstrap resamples for E09.")
    args = parser.parse_args()

    cfg = load_protocol(args.config)
    data_dir = Path(args.data_dir)
    out_dir = ensure_dir(args.out_dir)
    fig_dir = ensure_dir(out_dir / "figures")
    tab_dir = ensure_dir(out_dir / "tables")

    episodes = load_jsonl(data_dir / "episodes.jsonl")
    turns_raw = load_jsonl(data_dir / "turns.jsonl")
    metrics = load_jsonl(data_dir / "turn_metrics.jsonl")
    if not episodes or not metrics:
        raise FileNotFoundError(f"Missing episodes.jsonl or turn_metrics.jsonl in {data_dir}")

    ep_df = pd.DataFrame(episodes)
    df = pd.DataFrame(metrics)
    tok_path = data_dir / "token_stats.parquet"
    tok_df = pd.read_parquet(tok_path) if tok_path.exists() else pd.DataFrame()

    bench = str(ep_df["benchmark"].iloc[0])
    n_boot = int(args.bootstrap or cfg["metrics"]["bootstrap_resamples"])
    seed = int(cfg["experiment"]["seed"])
    print(f"[Stage 3] Analyzing {bench}: {len(ep_df)} episodes, {len(df)} turns (bootstrap={n_boot})...")

    # 1. Summary stats table (E04/E05/E06/E07)
    summary_cols = [
        "I_empty_norm", "I_empty_raw", "I_neutral_norm", "I_fmt_delta",
        "H_C_mean", "H_C_p90", "H_C_max", "H_A_mean", "D_self_mean",
        "reward_raw", "G3_strict", "G5_strict", "cot_token_len", "action_token_len",
    ]
    summary_rows = []
    for c in summary_cols:
        if c not in df.columns:
            continue
        s = df[c].dropna()
        if s.empty:
            continue
        summary_rows.append(
            {
                "metric": c,
                "n": len(s),
                "mean": round(float(s.mean()), 5),
                "std": round(float(s.std()), 5),
                "p10": round(float(s.quantile(0.10)), 5),
                "p50": round(float(s.quantile(0.50)), 5),
                "p90": round(float(s.quantile(0.90)), 5),
                "pos_frac": round(float((s > 0).mean()), 4),
                "zero_frac": round(float((s == 0).mean()), 4),
                "neg_frac": round(float((s < 0).mean()), 4),
            }
        )
    pd.DataFrame(summary_rows).to_csv(tab_dir / "summary_stats.csv", index=False)

    # 2. E09 Episode-Clustered Correlations
    x_cols = ["I_empty_norm", "I_neutral_norm", "H_C_mean", "H_C_p90", "H_A_mean"]
    y_cols = ["reward_raw", "G3_strict", "G5_strict", "terminal_success", "action_valid"]
    e09_rows = []
    df_num = df.copy()
    df_num["terminal_success"] = df_num["terminal_success"].astype(float)
    df_num["action_valid"] = df_num["action_valid"].astype(float)
    for xc in x_cols:
        if xc not in df_num.columns or df_num[xc].dropna().empty:
            continue
        for yc in y_cols:
            e09_rows.append(cluster_bootstrap_spearman(df_num, xc, yc, n_resamples=n_boot, seed=seed))
    e09_df = pd.DataFrame(e09_rows)
    e09_df.to_csv(tab_dir / "e09_correlations.csv", index=False)

    # 3. E10 Quadrants & Leak-Free CV
    sub10 = df.dropna(subset=["I_empty_norm", "H_C_mean", "A_turn_proxy"]).copy()
    if not sub10.empty:
        med_h = sub10.groupby("task_id")["H_C_mean"].transform("median")
        sub10["H_group"] = np.where(sub10["H_C_mean"] >= med_h, "High_HC", "Low_HC")
        sub10["I_group"] = np.where(sub10["I_empty_norm"] > 0, "I_pos", "I_neg_or_zero")
        q_tab = sub10.groupby(["H_group", "I_group"]).agg(
            n_turns=("turn_id", "count"),
            action_valid_rate=("action_valid", "mean"),
            reward_mean=("reward_raw", "mean"),
            G3_strict_mean=("G3_strict", "mean"),
            G5_strict_mean=("G5_strict", "mean"),
            success_rate=("terminal_success", "mean"),
        ).reset_index()
        q_tab.to_csv(tab_dir / "e10_quadrants.csv", index=False)
    cv_df = run_leak_free_e10_cv(df)
    cv_df.to_csv(tab_dir / "e10_leak_free_cv.csv", index=False)

    # 4. E11 Zero-reward table
    z_df = df[df["reward_raw"] == 0]
    if not z_df.empty:
        e11_tab = z_df.groupby("zero_reward_category").agg(
            n_turns=("turn_id", "count"),
            I_empty_mean=("I_empty_norm", "mean"),
            I_empty_median=("I_empty_norm", "median"),
            H_C_mean=("H_C_mean", "mean"),
            H_C_p90_mean=("H_C_p90", "mean"),
        ).reset_index()
        e11_tab.to_csv(tab_dir / "e11_zero_reward.csv", index=False)

    # 5. E16 Controlled regression
    e16_df = run_e16_controlled_regression(df)
    e16_df.to_csv(tab_dir / "e16_controlled_regression.csv", index=False)

    # 6. Generate all 10 standard figures
    plot_fig01_e04_bridge_dist(df, fig_dir / "Fig01_E04_bridge_dist.png", bench)
    plot_fig02_e04_e16_stratified(df, fig_dir / "Fig02_E04_E16_bridge_stratified.png", bench)
    plot_fig03_e05_e06_entropy(df, fig_dir / "Fig03_E05_E06_entropy_compare.png", bench)
    plot_fig04_e07_e11_rewards(df, fig_dir / "Fig04_E07_E11_reward_and_zero_cats.png", bench)
    plot_fig05_e09_forest(e09_df, fig_dir / "Fig05_E09_clustered_correlations.png", bench)
    plot_fig06_e10_joint(df, cv_df, fig_dir / "Fig06_E10_joint_quadrants.png", bench)
    plot_fig07_e13_robustness(df, fig_dir / "Fig07_E13_baseline_robustness.png", bench)
    plot_fig08_e15_failure_countdown(df, fig_dir / "Fig08_E15_failure_trajectory.png", bench)
    plot_fig09_e17_cot_position(df, tok_df, fig_dir / "Fig09_E17_cot_position_entropy.png", bench)
    e18_stats = plot_fig10_e18_teacher_gap(df, fig_dir / "Fig10_E18_teacher_gap.png", bench)

    # 7. Auto-generate Chinese E12 report_summary.md
    write_chinese_summary_report(
        bench, ep_df, df, turns_raw, e09_df, cv_df, e16_df, e18_stats, out_dir / "report_summary.md"
    )
    print(f"[Stage 3 Complete] Wrote 10 figures, CSV tables, and report_summary.md to: {out_dir}")


if __name__ == "__main__":
    main()

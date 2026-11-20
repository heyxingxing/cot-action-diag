#!/usr/bin/env python3
"""
Build the interactive experiment report as `index.html` at the repository root.

The page is served by GitHub Pages from the repository root
(https://heyxingxing.github.io/cot-action-diag/). Figures (`results/*/figures`) and
KaTeX (`assets/katex`) are referenced by relative path, so the page works inside the
repository layout only (opened locally via file:// or served by Pages). Episode, turn
and token data are embedded as JSON.

Sections: summary and definitions, figures and tables, trajectory explorer,
case studies, setup and the E01-E18 checklist.

Reading `token_stats.parquet` needs a recent pyarrow (the files were written by
pyarrow 26; pyarrow 19 fails with "Repetition level histogram size mismatch").
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RES_DIR = ROOT / "results"
OUT_HTML = ROOT / "index.html"

BOOT_B = 2000
BOOT_SEED = 0
# |C_t| below this is treated as "CoT had no effect" in the f(C, H) analysis.
C_EPS = 0.01
ENTROPY_BINS = [(-np.inf, 0.05, "≤ 0.05"), (0.05, 0.2, "0.05–0.2"), (0.2, 0.5, "0.2–0.5"), (0.5, 1.0, "0.5–1"), (1.0, np.inf, "> 1")]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix() if path.exists() else ""


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rnd(x: Any, nd: int) -> Any:
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


def cluster_ratio_ci(df: pd.DataFrame, num: pd.Series, den: pd.Series, col: str, rng: np.random.Generator) -> List[Any]:
    """Point estimate and episode-cluster bootstrap 95% CI of mean(col | num) / mean(col | den)."""
    uniq, inv = np.unique(df["episode_id"].to_numpy(), return_inverse=True)
    vals = np.nan_to_num(df[col].to_numpy(float))

    def per_episode(mask: pd.Series):
        m = mask.to_numpy(bool)
        s = np.bincount(inv, weights=np.where(m, vals, 0.0), minlength=len(uniq))
        n = np.bincount(inv, weights=m.astype(float), minlength=len(uniq))
        return s, n

    s1, n1 = per_episode(num)
    s0, n0 = per_episode(den)
    idx = rng.integers(0, len(uniq), size=(BOOT_B, len(uniq)))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = (s1[idx].sum(1) / n1[idx].sum(1)) / (s0[idx].sum(1) / n0[idx].sum(1))
    point = (s1.sum() / n1.sum()) / (s0.sum() / n0.sum())
    lo, hi = np.nanpercentile(ratios, [2.5, 97.5])
    return [rnd(point, 2), rnd(lo, 2), rnd(hi, 2)]


def build_benchmark_bundle(bench_key: str, folder: str, label: str, rng: np.random.Generator) -> Dict[str, Any]:
    ddir = DATA_DIR / folder
    rdir = RES_DIR / folder
    eps = load_jsonl(ddir / "episodes.jsonl")
    turns = load_jsonl(ddir / "turns.jsonl")
    tms = load_jsonl(ddir / "turn_metrics.jsonl")
    tok_df = pd.read_parquet(ddir / "token_stats.parquet")

    ep_df = pd.DataFrame(eps)
    t_df = pd.DataFrame(turns)[
        ["episode_id", "turn_id", "observation", "next_observation", "cot", "action", "parse_error_type", "truncated"]
    ]
    df = pd.DataFrame(tms).merge(t_df, on=["episode_id", "turn_id"], how="left")
    df = df.sort_values(["episode_id", "turn_id"]).reset_index(drop=True)
    df["prev_valid"] = df.groupby("episode_id")["action_valid"].shift(1)
    df["prev_action"] = df.groupby("episode_id")["action"].shift(1)
    # Loop turn: the previous action was invalid and is repeated verbatim.
    df["loop"] = (df["action"] == df["prev_action"]) & (df["prev_valid"] == False)  # noqa: E712
    df["C"] = df["A_turn_proxy"] * df["I_empty_norm"]

    # CoT token stats grouped per turn into compact arrays: [text, entropy, logp_s, d_self]
    cot_tok = tok_df[tok_df["segment"] == "cot"].copy()
    cot_tok["d_self"] = cot_tok["logp_teacher_self"] - cot_tok["logp_student"]
    tok_map: Dict[tuple, List[List[Any]]] = {}
    for (eid, tid), grp in cot_tok.groupby(["episode_id", "turn_id"], sort=False):
        grp = grp.sort_values("token_pos")
        tok_map[(str(eid), int(tid))] = [
            [
                str(txt),
                round(float(ent), 3) if pd.notna(ent) else 0.0,
                round(float(lps), 3) if pd.notna(lps) else 0.0,
                round(float(ds), 3) if pd.notna(ds) else None,
            ]
            for txt, ent, lps, ds in zip(grp["token_text"], grp["entropy_nats"], grp["logp_student"], grp["d_self"])
        ]

    # Per-task stats + within-task correlations
    task_stats = []
    for (tier, tid), g_ep in ep_df.groupby(["difficulty_tier", "task_id"]):
        g_t = df[df["task_id"] == tid]
        r_ir = stats.spearmanr(g_t["I_empty_norm"], g_t["reward_raw"], nan_policy="omit")[0]
        r_hhv = stats.spearmanr(g_t["H_C_head_mean"], g_t["action_valid"].astype(float), nan_policy="omit")[0]
        task_stats.append(
            {
                "benchmark": label,
                "tier": tier,
                "task_id": tid,
                "n_ep": int(len(g_ep)),
                "n_turn": int(len(g_t)),
                "succ_pct": round(float(g_ep["terminal_success"].mean()) * 100, 1),
                "score_mean": round(float(g_ep["final_score_norm"].mean()), 3),
                "valid_pct": round(float(g_t["action_valid"].mean()) * 100, 1),
                "I_med": round(float(g_t["I_empty_norm"].median()), 4),
                "rho_I_r": rnd(r_ir, 3),
                "rho_Hhead_valid": rnd(r_hhv, 3),
            }
        )

    # Transitions conditioned on the previous step's validity; F->F is split into loops and new actions.
    pv, av = df["prev_valid"], df["action_valid"].astype(bool)
    masks = {
        "FF": (pv == False) & ~av,  # noqa: E712
        "FT": (pv == False) & av,  # noqa: E712
        "TF": (pv == True) & ~av,  # noqa: E712
        "TT": (pv == True) & av,  # noqa: E712
    }
    masks["FF_loop"] = masks["FF"] & df["loop"]
    masks["FF_new"] = masks["FF"] & ~df["loop"]
    rec_rows = []
    for key in ["FF", "FF_loop", "FF_new", "FT", "TF", "TT"]:
        g = df[masks[key]]
        rec_rows.append(
            {
                "key": key,
                "n": int(len(g)),
                "I_mean": rnd(g["I_empty_norm"].mean(), 3),
                "I_med": rnd(g["I_empty_norm"].median(), 4),
                "H_head": rnd(g["H_C_head_mean"].mean(), 3),
                "H_tail": rnd(g["H_C_tail_mean"].mean(), 3),
            }
        )
    rec_ci = {
        "FT_vs_FF": cluster_ratio_ci(df, masks["FT"], masks["FF"], "I_empty_norm", rng),
        "FT_vs_FF_new": cluster_ratio_ci(df, masks["FT"], masks["FF_new"], "I_empty_norm", rng),
    }

    # w = f(C, H) analysis: groups by sign of C_t and by H^C relative to the task median.
    C = df["C"]
    nz = C != 0
    high_h = df["H_C_mean"] > df.groupby("task_id")["H_C_mean"].transform("median")
    groups = []
    for c_key, c_mask in [("pos", C > C_EPS), ("neg", C < -C_EPS)]:
        for h_key, h_mask in [("high", high_h), ("low", ~high_h)]:
            g = df[c_mask & h_mask]
            groups.append(
                {
                    "c": c_key,
                    "h": h_key,
                    "n": int(len(g)),
                    "valid": rnd(g["action_valid"].mean(), 2),
                    "G3s": rnd(g["G3_strict"].mean(), 3),
                    "succ": rnd(g["terminal_success"].mean(), 2),
                    "loop": rnd(g["loop"].mean(), 2),
                    "D": rnd(g["D_self_mean"].mean(), 2),
                }
            )
    inv_df = df[~av]
    low_q = inv_df["H_C_mean"] <= inv_df["H_C_mean"].quantile(0.25)

    d, h = cot_tok["d_self"], cot_tok["entropy_nats"]
    neg_mass = float((-d.clip(upper=0)).sum())
    token_bins = []
    for lo, hi, lbl in ENTROPY_BINS:
        sel = (h > lo) & (h <= hi)
        db = d[sel]
        token_bins.append(
            {
                "label": lbl,
                "share": rnd(sel.mean() * 100, 1),
                "d_med": rnd(db.median(), 3),
                "d_p10": rnd(db.quantile(0.1), 2),
                "lt_m1": rnd((db < -1).mean() * 100, 1),
                "neg_mass": rnd(float((-db.clip(upper=0)).sum()) / neg_mass * 100, 1),
            }
        )

    def _turn_corr(g: pd.DataFrame) -> float:
        if len(g) < 3 or g["d_self"].std() == 0 or g["entropy_nats"].std() == 0:
            return np.nan
        return float(np.corrcoef(g["d_self"], g["entropy_nats"])[0, 1])

    turn_corr = cot_tok.groupby(["episode_id", "turn_id"])[["d_self", "entropy_nats"]].apply(_turn_corr).dropna()

    fch = {
        "C_neg_pct": rnd((C < 0).mean() * 100, 1),
        "C_small_pct": rnd((C.abs() < C_EPS).mean() * 100, 1),
        "sign_agree_pct": rnd((np.sign(C[nz]) == np.sign(df.loc[nz, "A_turn_proxy"])).mean() * 100, 1),
        "groups": groups,
        "loop_lowH_invalid_pct": rnd(inv_df[low_q]["loop"].mean() * 100, 0),
        "loop_other_invalid_pct": rnd(inv_df[~low_q]["loop"].mean() * 100, 0),
        "token_d_median": rnd(d.median(), 4),
        "turn_corr_mean": rnd(turn_corr.mean(), 3),
        "turn_corr_neg_pct": rnd((turn_corr < 0).mean() * 100, 1),
        "token_bins": token_bins,
    }

    # Episodes + turns payload for the trajectory explorer
    ep_payload = []
    ep_meta_map = {e["episode_id"]: e for e in eps}
    for eid, g in df.groupby("episode_id", sort=True):
        meta = ep_meta_map[eid]
        g = g.sort_values("turn_id")
        turns_list = []
        for _, r in g.iterrows():
            tid = int(r["turn_id"])
            act_len = max(1, int(r["action_token_len"]))

            def opt(col: str, nd: int, scale: float = 1.0) -> Any:
                return round(float(r[col]) / scale, nd) if pd.notna(r[col]) else None

            turns_list.append(
                {
                    "t": tid,
                    "obs": str(r["observation"])[:900],
                    "cot": str(r["cot"]),
                    "act": str(r["action"]),
                    "valid": bool(r["action_valid"]),
                    "pv": None if pd.isna(r["prev_valid"]) else bool(r["prev_valid"]),
                    "lp": bool(r["loop"]),
                    "r": round(float(r["reward_raw"]), 3),
                    "g3": round(float(r["G3_strict"]), 3),
                    "sc": round(float(r["score_norm"]), 3),
                    "ie": opt("I_empty_norm", 4),
                    "in": opt("I_neutral_norm", 4),
                    "hc": opt("H_C_mean", 4),
                    "hh": opt("H_C_head_mean", 4),
                    "ht": opt("H_C_tail_mean", 4),
                    "ha": opt("H_A_mean", 4),
                    "ds": opt("D_self_mean", 3),
                    "dnf": opt("D_self_neg_frac", 3),
                    "lpc": opt("logp_a_with_cot", 4, act_len),
                    "lpe": opt("logp_a_empty", 4, act_len),
                    "cat": str(r["zero_reward_category"]),
                    "toks": tok_map.get((str(eid), tid), []),
                }
            )
        ep_payload.append(
            {
                "id": eid,
                "bench": bench_key,
                "bench_label": label,
                "task": meta["task_id"],
                "tier": meta["difficulty_tier"],
                "seed": int(meta["seed"]),
                "succ": bool(meta["terminal_success"]),
                "score": round(float(meta["final_score_norm"]), 3),
                "n_turns": int(meta["num_turns"]),
                "inv": int(meta["invalid_action_count"]),
                "term": str(meta["termination_reason"]),
                "n_rec": int(masks["FT"][g.index].sum()),
                "turns": turns_list,
            }
        )

    tables = {}
    for tname in ["e11_zero_reward", "e16_controlled_regression"]:
        p = rdir / "tables" / f"{tname}.csv"
        tables[tname] = pd.read_csv(p).to_dict(orient="records") if p.exists() else []

    fig_files = {
        "Fig01": "Fig01_E04_bridge_dist.png",
        "Fig02": "Fig02_E04_E16_bridge_stratified.png",
        "Fig03": "Fig03_E05_E06_entropy_compare.png",
        "Fig04": "Fig04_E07_E11_reward_and_zero_cats.png",
        "Fig05": "Fig05_E09_clustered_correlations.png",
        "Fig06": "Fig06_E10_joint_quadrants.png",
        "Fig07": "Fig07_E13_baseline_robustness.png",
        "Fig08": "Fig08_E15_failure_trajectory.png",
        "Fig09": "Fig09_E17_cot_position_entropy.png",
        "Fig10": "Fig10_E18_teacher_gap.png",
    }
    figs = {k: rel(rdir / "figures" / fn) for k, fn in fig_files.items()}

    lp_emp = df["logp_a_empty"] / df["action_token_len"].clip(lower=1)
    both_i = df[["I_empty_norm", "I_neutral_norm"]].dropna()
    c_neg = df[C < 0]
    overview = {
        "label": label,
        "n_ep": int(len(ep_df)),
        "n_turn": int(len(df)),
        "avg_turns": round(float(ep_df["num_turns"].mean()), 1),
        "parse_pct": round(float(df["parse_ok"].mean()) * 100, 1),
        "valid_pct": round(float(av.mean()) * 100, 1),
        "succ_pct": round(float(ep_df["terminal_success"].mean()) * 100, 1),
        "score_mean": round(float(ep_df["final_score_norm"].mean()), 3),
        "r_pos_pct": round(float((df["reward_raw"] > 0).mean()) * 100, 1),
        "r_zero_pct": round(float((df["reward_raw"] == 0).mean()) * 100, 1),
        "loop_pct": round(float(df["loop"].mean()) * 100, 1),
        "I_emp_mean": round(float(df["I_empty_norm"].mean()), 3),
        "I_emp_p50": round(float(df["I_empty_norm"].median()), 4),
        "I_emp_p90": round(float(df["I_empty_norm"].quantile(0.9)), 2),
        "I_emp_pos_pct": round(float((df["I_empty_norm"] > 0).mean()) * 100, 1),
        "I_emp_neg_pct": round(float((df["I_empty_norm"] < 0).mean()) * 100, 1),
        "e13_rho": round(float(stats.spearmanr(both_i["I_empty_norm"], both_i["I_neutral_norm"])[0]), 3),
        "e13_sign_pct": round(float(np.mean((both_i["I_empty_norm"] > 0) == (both_i["I_neutral_norm"] > 0))) * 100, 1),
        "H_C_mean": round(float(df["H_C_mean"].mean()), 3),
        "H_A_mean": round(float(df["H_A_mean"].mean()), 3),
        "H_A_zero_pct": round(float((df["H_A_mean"] == 0).mean()) * 100, 1),
        "D_self_mean": round(float(df["D_self_mean"].mean()), 2),
        "D_neg_pct": round(float((df["D_self_mean"] < 0).mean()) * 100, 1),
        "rho_D_turn": round(float(stats.spearmanr(df["D_self_mean"], df["turn_id"], nan_policy="omit")[0]), 3),
        "negfrac_Cneg": round(float(c_neg["D_self_neg_frac"].mean()) * 100, 1),
        "negfrac_all": round(float(df["D_self_neg_frac"].mean()) * 100, 1),
        "rho_surp_emp": round(float(stats.spearmanr(df["I_empty_norm"], -lp_emp, nan_policy="omit")[0]), 3),
        "r_pos_I_med": round(float(df[df["reward_raw"] > 0]["I_empty_norm"].median()), 4),
    }

    return {
        "overview": overview,
        "tasks": task_stats,
        "recovery": rec_rows,
        "recovery_ci": rec_ci,
        "fch": fch,
        "tables": tables,
        "figures": figs,
        "episodes": ep_payload,
    }


# Curated case studies (episode_id / turn_id pairs and numbers checked against the data).
CURATED_CASES = [
    {
        "tag": "奖励与 $I_t$ 错位",
        "title": r"TextWorldExpress coin #1000，第 18–19 步：带来收益的转向步 $r_t=0$，拿到奖励的步 $I_t=0$",
        "episode_id": "textworld_express__coin__s1000",
        "turns": [18, 19],
        "mode": "entropy",
        "analysis": (
            r"第 17 步 <code>open dishwasher</code> 报错后，第 18 步 CoT 改为去其他房间寻找，执行 <code>move south</code>："
            r"$r_t=0$，$I_t=5.39$（无 Thought 时该动作的概率约 $2\times10^{-5}$）。"
            r"第 19 步进入客厅后执行 <code>take coin</code>，得到 $r_t=+1$，但 $I_t=0$：不写 Thought 模型也会选这个动作。"
            r"若按 $r_t\cdot I_t$ 分配信号，两步都得到 0；只有按未来收益 $G^{(k)}_t$ 分配，第 18 步才能得到功劳。"
        ),
    },
    {
        "tag": "报错后换路线（F→T）",
        "title": r"ScienceWorld find-living-thing #5，第 10–11 步：报错后改走走廊，$I_t=3.83$",
        "episode_id": "scienceworld__find-living-thing__s5",
        "turns": [10, 11],
        "mode": "entropy",
        "analysis": (
            r"第 10 步 <code>go to kitchen</code> 不合法（当前房间与厨房不相邻）。第 11 步 CoT 推断需要先确认相连的房间，"
            r"执行 <code>go to hallway</code>（$I_t=3.83$），此后 5 步内 $G^{(5)}_t=25$。"
            r"这是 F→T 中 $I_t$ 大且后续有收益的例子；但 SW 的 F→T 步 $I_t$ 中位数只有 0.017，这类步并不普遍。"
        ),
    },
    {
        "tag": "低熵的重复错误",
        "title": r"ScienceWorld chemistry-mix-paint #6，第 13–14 步：$\bar H^C_t=0.002$ nats 下重复同一不合法动作",
        "episode_id": "scienceworld__chemistry-mix-paint-secondary-color__s6",
        "turns": [13, 14],
        "mode": "entropy",
        "analysis": (
            r"第 12–14 步 CoT 逐字相同（“The correct command is open door to bee hive.”），三次执行同一个不合法动作，"
            r"第 14 步 $\bar H^C_t=0.002$ nats、$I_t=0.002$。TextWorldExpress coin #1018 第 13–15 步类似但不是打转："
            r"CoT 先指出上一个对象不是门，随后仍对另一个非门对象执行 <code>open</code>（第 14 步 $\bar H^C_t=0.0035$）。"
            r"不合法步中熵最低的四分之一里，打转步占 84%（SW）/ 46%（TWX），其余不合法步中为 50% / 33%。"
        ),
    },
    {
        "tag": "获得奖励的步 $I_t\\approx 0$",
        "title": r"ScienceWorld test-conductivity #3，第 16 步：+50 分的动作，$I_t=0$",
        "episode_id": "scienceworld__test-conductivity__s3",
        "turns": [16],
        "mode": "entropy",
        "analysis": (
            r"目标物体已在视野中，<code>focus on aluminum foil</code> 获得 +50 分。有无 Thought 时该动作的对数概率都接近 0，"
            r"因此 $I_t=0$。两个环境中 $r_t>0$ 的步，$I_t$ 中位数都约为 0，这是 $I_t$ 与即时奖励负相关的主要来源之一。"
        ),
    },
    {
        "tag": "$I_t$ 最大值",
        "title": r"TextWorldExpress cookingworld #1007，第 25 步：TWX 中最大的 $I_t=20.78$",
        "episode_id": "textworld_express__cookingworld__s1007",
        "turns": [25],
        "mode": "entropy",
        "analysis": (
            r"烤完鸡翅后，CoT 按菜谱转向切苹果，判断苹果可能在台面上，执行 <code>examine counter</code>。"
            r"下一步发现台面上没有苹果；该局最终超时失败，E11 因而把这一步归为“瞎逛”。"
            r"这一步的 CoT 是目标导向的：$I_t$ 大只说明 CoT 显著改变了动作分布，不说明这一步是好是坏。"
        ),
    },
    {
        "tag": "$C_t<0$ 与 $D<0$ 的符号冲突",
        "title": r"ScienceWorld test-conductivity #16，第 28 步：$C_t<0$，$\bar D^{\mathrm{self}}_t=-0.87$",
        "episode_id": "scienceworld__test-conductivity__s16",
        "turns": [28],
        "mode": "dself",
        "analysis": (
            r"该步 $\hat A_t=-0.197$、$I_t=15.4$，故 $C_t<0$。$\bar D^{\mathrm{self}}_t=-0.866$，56% 的 CoT token 满足 $D<0$，"
            r"但 token 级中位数约为 0：负 gap 集中在少数 token 上（下方按 $D$ 着色）。"
            r"若 $w_{t,i}$ 随 $C_t$ 取负号，$w_{t,i}D_{t,i}$ 会把这些 token 的更新方向反转。"
        ),
    },
]

# Figure captions. Numbers marked "读图" are read off the figure, the rest come from results/*/tables.
FIG_CATALOG = [
    {
        "key": "Fig01",
        "e_tag": "E04",
        "title": r"$I_t$ 的分布",
        "sw_note": r"均值 0.84，中位数 0.005，P90 2.94；93.8% 的步 $I_t>0$。",
        "tw_note": r"均值 1.29，中位数 0.027，P90 4.53；92.6% 的步 $I_t>0$。",
        "takeaway": r"$I_t$ 高度右偏：多数步接近 0，少数步很大；$I_t<0$ 的步很少，幅度也小。",
    },
    {
        "key": "Fig02",
        "e_tag": "E04 / E16",
        "title": r"$I_t$ 按任务、步数与动作长度分层",
        "sw_note": r"开局前几步 $I_t$ 较高，随步数下降；短动作的 $I_t$ 方差更大（读图）。",
        "tw_note": r"任务内 $I_t$ 中位数：coin 0.107，cookingworld 0.036，mapreader 0。",
        "takeaway": r"$I_t$ 随任务和步数变化，跨任务合并的相关需要用任务内相关和 E16 回归复核。",
    },
    {
        "key": "Fig03",
        "e_tag": "E05 / E06",
        "title": r"CoT 熵 $H^C$ 与动作熵 $H^A$",
        "sw_note": r"$\bar H^C_t$ 均值 0.153 nats；$\bar H^A_t$ 均值 0.015，57% 的步为 0。",
        "tw_note": r"$\bar H^C_t$ 均值 0.163 nats；$\bar H^A_t$ 均值 0.026，54% 的步为 0。",
        "takeaway": r"单步的不确定性主要在 CoT 中；Thought 写完后，动作基本确定。",
    },
    {
        "key": "Fig04",
        "e_tag": "E07 / E11",
        "title": r"奖励稀疏度与零奖励步分类",
        "sw_note": r"92.0% 的步 $r_t=0$。$I_t$ 中位数：瞎逛 0.126，非法 0.006，重复 0.002，有效铺垫 0.0008。",
        "tw_note": r"94.8% 的步 $r_t=0$。$I_t$ 中位数：瞎逛 0.569，非法 0.075，重复 0.004，有效铺垫 0.0003。",
        "takeaway": r"两边按 $I_t$ 中位数的排序相同（瞎逛 &gt; 非法 &gt; 重复 &gt; 有效铺垫），按均值的排序不同；$\bar H^C_t$ 在有效铺垫中最低（0.131 / 0.140）。",
    },
    {
        "key": "Fig05",
        "e_tag": "E09",
        "title": r"核心指标与结果的相关（episode 聚类 bootstrap，95% CI）",
        "sw_note": r"$I_t$ 与 $r_t$ 显著负相关（$-0.217$；中性基线 $-0.231$）；$I_t$ 与 $G^{(3)}_t$、$G^{(5)}_t$、成功、合法性均不显著；$\bar H^C_t$ 各项均不显著。",
        "tw_note": r"$I_t$ 与 $r_t$（$-0.268$）、$G^{(3)}_t$（$-0.210$）、成功（$-0.233$）、合法性（$-0.172$）均显著为负；$\bar H^C_t$ 为弱负相关（$-0.12$ 至 $-0.18$）。",
        "takeaway": r"两边都显著的只有 $I_t$ 与即时奖励的负相关；其余关系在 TWX 更强。",
    },
    {
        "key": "Fig06",
        "e_tag": "E10",
        "title": r"$H^C\times I_t$ 四象限与无泄露预测",
        "sw_note": r"低 $H^C$ 且 $I_t\le 0$（n=57）：合法率 84%，$G^{(3)}_t$ 均值 13.9；低 $H^C$ 且 $I_t>0$：2.6。预测终局成功的 AUROC：$I_t$ 0.463，$r_t$ 0.621。",
        "tw_note": r"$I_t\le 0$ 两格的成功率为 60% / 57%，$I_t>0$ 约 22%；但 $I_t\le 0$ 的步有 46% 来自全胜的 mapreader，任务内只有 coin 仍有差异（0.55 vs 0.27，n=22）。",
        "takeaway": r"预测 $G^{(3)}_t$ 的 5 折 CV $R^2$ 约为 0（$-0.012$ / $0.001$）。$I_t\le 0$ 的步结果较好，部分来自任务构成；SW find-living-thing 任务内仍成立（成功率 0.51 vs 0.26）。",
    },
    {
        "key": "Fig07",
        "e_tag": "E13",
        "title": r"无 CoT 基线：空 Thought 与中性 Thought",
        "sw_note": r"Spearman 0.777，符号一致率 92.2%；$I_t-I^{\mathrm{neu}}_t$ 均值 $-0.101$。",
        "tw_note": r"Spearman 0.849，符号一致率 87.1%；均值差 $-0.283$。",
        "takeaway": r"两种基线结论一致，空 Thought 的格式效应很小。",
    },
    {
        "key": "Fig08",
        "e_tag": "E15",
        "title": r"失败局结束前 5 步的熵与 $I_t$",
        "sw_note": r"52 局失败局。$H^C$ 的 P90 从 0.50 [0.43, 0.57] 降到 0.38 [0.30, 0.44]；但全体样本中它本身随步数下降（$\rho=-0.18$），而失败局大多在第 34 步结束，无法与步数深度区分。",
        "tw_note": r"30 局失败局。$H^C$ 的 P90 依次为 0.51、0.56、0.54、0.56、0.46，置信区间重叠；$I_t$ 在 0.88–1.34 之间波动。",
        "takeaway": r"没有发现可靠的失败前预警信号。",
    },
    {
        "key": "Fig09",
        "e_tag": "E17",
        "title": r"CoT 内部不同位置的熵",
        "sw_note": r"不合法步的前 1/3 熵更高（0.206 vs 0.137），后 1/3 略低（0.147 vs 0.165）。",
        "tw_note": r"前 1/3：0.237 vs 0.133。按十分位，不合法步在 10–40% 处更高，60–90% 处略低，末 10% 又更高。",
        "takeaway": r"前 1/3 熵与合法性的相关（$-0.227$ / $-0.293$）强于整段均值（$-0.038$ / $-0.149$），但它与上一步是否合法的相关更强（$-0.307$ / $-0.366$）；按上一步合法性分层后降至 $-0.08$ 至 $-0.18$。",
    },
    {
        "key": "Fig10",
        "e_tag": "E18",
        "title": r"Self-teacher gap $D^{\mathrm{self}}$",
        "sw_note": r"98.4% 的步均值为负（均值 $-1.12$），随步数加深更负（$\rho=-0.349$）。$C_t<0$ 的步中 $D<0$ 的 token 占 68.4%，全部步为 65.4%。",
        "tw_note": r"97.9% 为负（均值 $-1.55$，$\rho=-0.276$）。$C_t<0$ 的步中 $D<0$ 的 token 占 65.4%，全部步为 64.1%。",
        "takeaway": r"token 级 $D$ 的中位数约为 $-0.004$，量级大的负值集中在少数 token（每步 P10 约 $-3$ 至 $-4$）。$D<0$ 并不限于 $C_t<0$ 的步。teacher 上下文中历史轮只保留 Action 行，$D<0$ 可能部分来自格式偏移。",
    },
]


def main() -> None:
    print("[HTML Builder] Loading ScienceWorld and TextWorldExpress data and token stats...")
    rng = np.random.default_rng(BOOT_SEED)
    sw = build_benchmark_bundle("scienceworld", "scienceworld_full", "ScienceWorld", rng)
    tw = build_benchmark_bundle("textworld_express", "textworld_express_full", "TextWorldExpress", rng)
    payload = {
        "sw": sw,
        "tw": tw,
        "cross_fig": rel(RES_DIR / "cross_benchmark" / "Fig_cross_benchmark_overview.png"),
        "curated_cases": CURATED_CASES,
        "fig_catalog": FIG_CATALOG,
    }
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html_doc = TEMPLATE.replace("__BUILD_DATE__", dt.date.today().isoformat()).replace("__REPORT_DATA__", data_json)
    OUT_HTML.write_text(html_doc, encoding="utf-8")

    for b in (sw, tw):
        o, f = b["overview"], b["fch"]
        print(
            f"  {o['label']}: I>0 {o['I_emp_pos_pct']}%, I<0 {o['I_emp_neg_pct']}%, rho_surp {o['rho_surp_emp']}, "
            f"r>0 I_med {o['r_pos_I_med']}, loop {o['loop_pct']}%, H_A=0 {o['H_A_zero_pct']}%, "
            f"FT/FF {b['recovery_ci']['FT_vs_FF']}, FT/FF_new {b['recovery_ci']['FT_vs_FF_new']}, "
            f"C<0 {f['C_neg_pct']}%, sign agree {f['sign_agree_pct']}%, |C|<eps {f['C_small_pct']}%, "
            f"turn corr(D,H) {f['turn_corr_mean']} ({f['turn_corr_neg_pct']}% < 0)"
        )
    size_mb = OUT_HTML.stat().st_size / (1024 * 1024)
    print(f"[HTML Builder Complete] Wrote {OUT_HTML} ({size_mb:.2f} MB)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>CoT→Action 诊断实验报告</title>
<meta name="description" content="Qwen3-8B 在 ScienceWorld 与 TextWorldExpress 上的 CoT→Action 影响与长程信用分配诊断（E01–E18）" />
<link rel="stylesheet" href="assets/katex/katex.min.css" />
<script src="assets/katex/katex.min.js"></script>
<script src="assets/katex/contrib/auto-render.min.js"></script>
<style>
  :root {
    --bg: #ffffff;
    --soft: #f7f6f2;
    --soft-2: #efede6;
    --border: #e2dfd6;
    --rule: #222222;
    --text: #1b1b1b;
    --muted: #5b5850;
    --faint: #8d897f;
    --accent: #1f4e79;
    --accent-soft: #eaf0f6;
    --pos: #1e6b47;
    --neg: #a3262c;
    --warn: #8a5a00;
    --violet: #5b3f8c;
    --serif: "Charter", "Bitstream Charter", "Sitka Text", "Cambria", "Georgia", "Songti SC", "STSong", "Noto Serif CJK SC", "Source Han Serif SC", "SimSun", serif;
    --sans: -apple-system, BlinkMacSystemFont, "Helvetica Neue", "Segoe UI", "PingFang SC", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
    --mono: "SF Mono", "JetBrains Mono", Menlo, Consolas, "Liberation Mono", monospace;
  }
  html { color-scheme: light; }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font-family: var(--serif); font-size: 16px; line-height: 1.75;
    -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility;
  }
  .katex { font-size: 1.05em; }
  .katex-display { margin: 12px 0; overflow-x: auto; overflow-y: hidden; padding: 2px 0; }
  a { color: var(--accent); text-decoration: none; border-bottom: 1px solid rgba(31,78,121,.28); }
  a:hover { border-bottom-color: var(--accent); }
  code { font-family: var(--mono); font-size: .84em; background: var(--soft); padding: .08em .35em; border-radius: 3px; }
  p { margin: 0 0 12px; }

  /* Masthead */
  .masthead { border-bottom: 1px solid var(--border); padding: 46px 20px 28px; text-align: center; }
  .masthead h1 { font-size: 30px; font-weight: 600; line-height: 1.35; margin: 0 0 8px; }
  .masthead .sub { font-size: 17px; color: var(--muted); margin: 0 0 14px; }
  .masthead .meta { font-family: var(--sans); font-size: 13px; color: var(--muted); }
  .masthead .meta span + span::before { content: "·"; margin: 0 9px; color: var(--faint); }

  /* Tabs */
  nav.tabs { position: sticky; top: 0; z-index: 40; background: rgba(255,255,255,.96); backdrop-filter: blur(6px); border-bottom: 1px solid var(--border); }
  .tabs-inner { max-width: 1240px; margin: 0 auto; padding: 0 16px; display: flex; gap: 2px; overflow-x: auto; }
  .tab-btn {
    appearance: none; border: 0; background: transparent; cursor: pointer;
    font-family: var(--sans); font-size: 14px; font-weight: 500; color: var(--muted);
    padding: 12px 14px; border-bottom: 2px solid transparent; white-space: nowrap;
  }
  .tab-btn:hover { color: var(--text); }
  .tab-btn.active { color: var(--text); border-bottom-color: var(--text); }

  main { max-width: 1240px; margin: 0 auto; padding: 30px 16px 90px; }
  .tab-panel { display: none; }
  .tab-panel.active { display: block; }
  .prose { max-width: 880px; }

  h2.sec { font-size: 22px; font-weight: 600; margin: 44px 0 14px; padding-bottom: 6px; border-bottom: 1px solid var(--border); }
  .tab-panel > h2.sec:first-child, .tab-panel > .prose > h2.sec:first-child { margin-top: 0; }
  h2.sec .n { color: var(--faint); font-weight: 500; margin-right: 10px; }
  h3.sub { font-size: 17px; font-weight: 600; margin: 26px 0 8px; }
  .note { font-size: 14px; color: var(--muted); }
  .abstract { background: var(--soft); border-left: 3px solid var(--accent); padding: 16px 20px; margin: 0 0 8px; }
  .abstract p:last-child { margin-bottom: 0; }

  ol.findings { padding-left: 1.5em; margin: 0; }
  ol.findings > li { margin-bottom: 16px; padding-left: 4px; }
  ol.findings > li > b:first-child { font-weight: 600; }
  ul.plain { padding-left: 1.3em; margin: 0 0 12px; }
  ul.plain li { margin-bottom: 6px; }

  /* booktabs-style tables */
  .table-wrap { overflow-x: auto; margin: 10px 0 6px; }
  table { border-collapse: collapse; width: 100%; font-size: 14.5px; font-variant-numeric: tabular-nums lining-nums; border-top: 1.5px solid var(--rule); border-bottom: 1.5px solid var(--rule); }
  thead th { border-bottom: 1px solid var(--rule); font-weight: 600; text-align: left; padding: 7px 10px; white-space: nowrap; vertical-align: bottom; }
  thead tr.group th { border-bottom: 1px solid var(--border); text-align: center; }
  td { padding: 6px 10px; vertical-align: top; border-bottom: 1px solid #efede6; }
  tbody tr:last-child td { border-bottom: 0; }
  th.num, td.num { text-align: right; white-space: nowrap; }
  td.lbl { min-width: 180px; }
  tr.subrow td:first-child { padding-left: 28px; color: var(--muted); }
  tr.em td { background: #f6f3ea; }
  tr.sep td { border-top: 1px solid var(--rule); }
  .caption { font-size: 14px; color: var(--muted); margin: 6px 0 0; }
  .pos { color: var(--pos); }
  .neg { color: var(--neg); }
  .dim { color: var(--faint); }
  .ci { color: var(--muted); font-size: .92em; }

  /* Figures */
  .fig-toolbar { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; margin: 0 0 18px; }
  figure.fig { margin: 0 0 40px; padding-bottom: 26px; border-bottom: 1px solid var(--border); }
  .fig-label { font-family: var(--sans); font-size: 12.5px; color: var(--muted); letter-spacing: .03em; }
  .fig-title { font-size: 18px; font-weight: 600; margin: 2px 0 12px; }
  .fig-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }
  .fig-single { max-width: 820px; }
  .fig-img { width: 100%; height: auto; display: block; border: 1px solid var(--border); cursor: zoom-in; background: #fff; }
  .bench-name { font-family: var(--sans); font-size: 12.5px; font-weight: 600; color: var(--muted); margin-bottom: 6px; }
  .fig-note { font-size: 14.5px; margin-top: 8px; }
  .fig-take { font-size: 15px; margin: 14px 0 0; }

  .seg { display: inline-flex; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; font-family: var(--sans); }
  .seg button { appearance: none; border: 0; background: #fff; padding: 5px 12px; font-size: 13px; color: var(--muted); cursor: pointer; font-family: var(--sans); }
  .seg button + button { border-left: 1px solid var(--border); }
  .seg button.active { background: var(--text); color: #fff; }
  .btn { font-family: var(--sans); font-size: 13px; border: 1px solid var(--border); background: #fff; border-radius: 6px; padding: 5px 12px; cursor: pointer; color: var(--text); white-space: nowrap; }
  .btn:hover { background: var(--soft); }

  .tag { display: inline-block; font-family: var(--sans); font-size: 11.5px; font-weight: 500; padding: 0 7px; border-radius: 3px; border: 1px solid var(--border); color: var(--muted); background: #fff; white-space: nowrap; line-height: 1.75; }
  .tag.pos { color: var(--pos); border-color: #b9d6c6; background: #f1f8f4; }
  .tag.neg { color: var(--neg); border-color: #e3c0c2; background: #fbf2f2; }
  .tag.acc { color: var(--accent); border-color: #c4d4e4; background: var(--accent-soft); }
  .tag.vio { color: var(--violet); border-color: #d6cde6; background: #f6f3fa; }
  .tag.warn { color: var(--warn); border-color: #e6d4ad; background: #fbf6ea; }

  /* Explorer */
  .explorer { display: grid; grid-template-columns: 300px minmax(0, 1fr); gap: 22px; align-items: start; }
  .ep-side { position: sticky; top: 58px; max-height: calc(100vh - 74px); display: flex; flex-direction: column; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; font-family: var(--sans); }
  .ep-filters { padding: 12px; background: var(--soft); border-bottom: 1px solid var(--border); display: grid; gap: 8px; font-size: 13px; }
  .ep-filters select, .ep-filters input { width: 100%; padding: 5px 8px; border: 1px solid var(--border); border-radius: 5px; background: #fff; font-size: 13px; font-family: var(--sans); color: var(--text); }
  .ep-list { overflow-y: auto; flex: 1; }
  .ep-item { padding: 9px 12px; border-bottom: 1px solid #f0eee8; cursor: pointer; }
  .ep-item:hover { background: var(--soft); }
  .ep-item.active { background: var(--accent-soft); box-shadow: inset 3px 0 0 var(--accent); }
  .ep-item .t { font-size: 13px; font-weight: 600; }
  .ep-item .m { font-size: 12px; color: var(--muted); display: flex; gap: 8px; flex-wrap: wrap; margin-top: 2px; }
  .panel { border: 1px solid var(--border); border-radius: 6px; padding: 16px 18px; margin-bottom: 16px; background: #fff; }
  .ep-head { display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px; }
  .ep-head h3 { margin: 6px 0 0; font-size: 18px; font-weight: 600; font-family: var(--mono); word-break: break-all; }
  .ep-stats { display: flex; gap: 18px; font-family: var(--sans); font-size: 13px; color: var(--muted); }
  .ep-stats b { color: var(--text); font-size: 16px; font-weight: 600; }

  .turn { border: 1px solid var(--border); border-radius: 6px; padding: 12px 16px; margin-bottom: 12px; background: #fff; }
  .turn.k-rec { box-shadow: inset 3px 0 0 var(--violet); }
  .turn.k-rew { box-shadow: inset 3px 0 0 var(--pos); }
  .turn.k-inv { box-shadow: inset 3px 0 0 var(--neg); }
  .turn.hl { outline: 2px solid var(--accent); outline-offset: 1px; }
  .turn-head { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; font-family: var(--sans); font-size: 13px; margin-bottom: 8px; }
  .turn-head .tt { font-weight: 600; font-size: 14px; margin-right: 4px; }
  .turn-head .tags { display: flex; gap: 5px; flex-wrap: wrap; align-items: center; }
  .turn-head .right { color: var(--muted); }
  .obs { font-family: var(--sans); font-size: 13px; color: var(--muted); margin-bottom: 8px; }
  .obs summary { cursor: pointer; }
  .obs pre { white-space: pre-wrap; font-family: var(--mono); font-size: 12.5px; background: var(--soft); padding: 8px 10px; border-radius: 4px; margin: 6px 0 0; color: var(--text); }
  .cot-box { font-family: var(--serif); font-size: 15px; line-height: 1.95; margin-bottom: 8px; }
  .lbl-s { font-family: var(--sans); font-size: 12px; color: var(--faint); margin-right: 6px; }
  .tok { padding: 1px 0; border-radius: 2px; }
  .tok:hover { outline: 1px solid var(--text); }
  .act-row { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; border-top: 1px dashed var(--border); padding-top: 8px; font-size: 14px; }
  .act-cmd { font-family: var(--mono); font-weight: 600; font-size: 13.5px; }
  .metrics { display: flex; flex-wrap: wrap; gap: 4px 18px; margin-top: 6px; font-size: 13.5px; color: var(--muted); }
  .metrics b { color: var(--text); font-weight: 500; }

  .legend { font-family: var(--sans); font-size: 12px; color: var(--muted); display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
  .sw { display: inline-block; width: 12px; height: 12px; border-radius: 2px; vertical-align: -1px; margin-right: 4px; border: 1px solid rgba(0,0,0,.12); }
  .dot { border-radius: 50%; }

  /* Cases */
  .case { margin-bottom: 40px; padding-bottom: 26px; border-bottom: 1px solid var(--border); }
  .case-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; flex-wrap: wrap; }
  .case h3 { font-size: 18px; font-weight: 600; margin: 4px 0 10px; }
  .case .analysis { max-width: 900px; margin-bottom: 14px; }

  .lightbox { position: fixed; inset: 0; z-index: 100; background: rgba(20,20,20,.85); display: none; align-items: center; justify-content: center; padding: 24px; cursor: zoom-out; }
  .lightbox.open { display: flex; }
  .lightbox img { max-width: 96vw; max-height: 92vh; background: #fff; }
  #tok-tip { position: fixed; z-index: 90; pointer-events: none; display: none; background: #1b1b1b; color: #f5f5f5; padding: 7px 10px; border-radius: 4px; font-size: 12px; font-family: var(--mono); line-height: 1.55; max-width: 300px; }

  @media (max-width: 960px) {
    .explorer, .fig-pair { grid-template-columns: 1fr; }
    .ep-side { position: static; max-height: 420px; }
    .masthead h1 { font-size: 24px; }
  }
</style>
</head>
<body>

<header class="masthead">
  <h1>CoT→Action 影响与长程信用分配：诊断实验报告</h1>
  <p class="sub">Qwen3-8B 在 ScienceWorld 与 TextWorldExpress 上的离线诊断（E01–E18）</p>
  <div class="meta">
    <span>生成于 __BUILD_DATE__</span>
    <span><a href="https://github.com/heyxingxing/cot-action-diag">代码与数据</a></span>
    <span><a href="https://github.com/heyxingxing/cot-action-diag/blob/main/PROGRESS.md">进度与待办</a></span>
    <span><a href="https://github.com/heyxingxing/cot-action-diag/blob/main/docs/experiment_plan_qwen3_8b.md">实验手册</a></span>
  </div>
</header>

<nav class="tabs">
  <div class="tabs-inner">
    <button class="tab-btn active" data-tab="summary">摘要与结论</button>
    <button class="tab-btn" data-tab="figures">图表</button>
    <button class="tab-btn" data-tab="explorer">轨迹浏览器</button>
    <button class="tab-btn" data-tab="cases">案例</button>
    <button class="tab-btn" data-tab="setup">设置与清单</button>
  </div>
</nav>

<main>
<!-- ============================ SUMMARY ============================ -->
<section id="summary" class="tab-panel active">
  <div class="prose">
    <h2 class="sec"><span class="n">1</span>摘要</h2>
    <div class="abstract">
      <p>Qwen3-8B 以零样本 ReAct 格式（先写 Thought，再写 Action）在两个文本环境中各完成 60 局（3 个任务 × 20 个种子），共 3,299 步。
      随后用 teacher forcing 离线计算每步 CoT 对动作的影响 $I_t$、CoT 与动作的逐 token 熵 $H^C$、$H^A$，以及去掉历史 CoT 后的 self-teacher gap $D_{t,i}$，并与即时奖励、未来收益和终局结果对照。</p>
      <p>主要发现：(1) CoT 对动作的影响集中在少数步，多数步 $I_t\approx 0$；(2) $I_t$ 与即时奖励稳定负相关，它度量 CoT 改变动作分布的程度，不度量动作好坏；(3) 单步指标对未来收益几乎没有预测力；(4) $D_{t,i}$ 几乎处处为负，若与带符号的权重相乘，会在大多数步上反转更新方向。
      据此，CoT 优势中的权重应取非负值，“好坏”的符号应来自单独的 RL 项。这些结论来自诊断数据，尚未经训练验证。</p>
    </div>

    <h2 class="sec"><span class="n">2</span>记号与指标</h2>
    <p>记 $h_t$ 为第 $t$ 步之前的完整历史（含历史 Thought），$z_t=(z_{t,1},\dots,z_{t,n})$ 为本步 CoT token，$a_t$ 为动作，$s_t$ 为去掉历史 Thought 后的状态上下文（历史 assistant 轮只保留 <code>Action: …</code> 行），$v_t\in\{0,1\}$ 为动作是否合法。</p>
    $$I_t=\frac{1}{|a_t|}\Big[\log\pi(a_t\mid h_t,z_t)-\log\pi(a_t\mid h_t,\varnothing)\Big]$$
    <p class="note">$\varnothing$ 为空 Thought；E13 另以中性 Thought 为基线，记作 $I^{\mathrm{neu}}_t$。除非特别说明，$I_t$ 指空 Thought 基线。</p>
    $$H^C_{t,i}=-\sum_{v\in\mathcal V}\pi(v\mid h_t,z_{t,&lt;i})\log\pi(v\mid h_t,z_{t,&lt;i})$$
    <p class="note">$H^A$ 的定义相同，作用于动作 token；单位为 nats，在全词表上以 fp32 计算。$\bar H^C_t$ 为一步内的均值，head / tail 为 CoT 前、后三分之一 token 的均值。</p>
    $$D_{t,i}=\log\pi_T(z_{t,i}\mid s_t,z_{t,&lt;i})-\log\pi_S(z_{t,i}\mid h_t,z_{t,&lt;i})$$
    <p class="note">本实验 $\pi_T=\pi_S$（同一 Qwen3-8B），记作 $D^{\mathrm{self}}$；$\bar D^{\mathrm{self}}_t$ 为一步内的均值。</p>
    $$G^{(k)}_t=\sum_{j=1}^{k}r_{t+j},\qquad \hat A_t=\frac{\tilde G^{(3)}_t-\mu_{\text{task}}}{\sigma_{\text{task}}},\qquad C_t=\hat A_t\,I_t$$
    <p class="note">$G^{(k)}_t$ 为严格未来收益（不含当前步）；$\tilde G^{(3)}_t=\sum_{j=0}^{2}r_{t+j}$ 含当前步，在任务内标准化后作为优势代理 $\hat A_t$。<b>打转步</b>：$a_t=a_{t-1}$ 且上一步动作不合法。</p>

    <h2 class="sec"><span class="n">3</span>基础统计</h2>
    <div class="table-wrap" id="stats-table"></div>
    <p class="caption">E01 设定的门槛为动作合法率 $\ge 60\%$，两边均未达到；SW 的不合法步主要来自失败后原样重试（打转）。所有图表均包含打转步。</p>

    <h2 class="sec"><span class="n">4</span>主要结论</h2>
    <ol class="findings">
      <li><b>CoT 对动作的影响集中在少数步。</b>
        $I_t$ 的中位数为 0.005（SW）/ 0.027（TWX），P90 为 2.94 / 4.53。$\bar H^A_t$ 在 57% / 54% 的步上为 0，均值 0.015 / 0.026 nats，远低于 CoT 熵 0.153 / 0.163 nats：Thought 写完后，动作基本确定。</li>
      <li><b>$I_t$ 度量 CoT 改变动作分布的程度，不度量动作好坏。</b>
        $I_t$ 与即时奖励的 Spearman 相关为 $-0.217$ / $-0.268$（episode 聚类 bootstrap 95% CI 不含 0），控制任务、步数、长度与合法性后仍显著（E16，$p=0.004$ / $0.014$）。
        由于有 Thought 时 $\log\pi(a_t\mid h_t,z_t)\approx 0$，$I_t$ 近似等于无 Thought 时动作的 surprisal（Spearman <span id="rho-surp"></span>）；获得奖励的步多为显而易见的动作，其 $I_t$ 中位数约为 0。
        空 Thought 与中性 Thought 两种基线结论一致（E13：Spearman 0.777 / 0.849，符号一致率 92.2% / 87.1%）。</li>
      <li><b>单步指标几乎预测不了未来收益。</b>
        以 $I_t$、$r_t$ 及其交互项预测 $G^{(3)}_t$，5 折 GroupKFold 的 $R^2$ 约为 0（$-0.012$ / $0.001$）；失败局结束前 5 步没有可靠的预警信号（E15，图 8）。</li>
      <li><b>“报错后自救步的 $I_t$ 更高”主要来自打转步。</b>
        上一步不合法时，本步合法（F→T）的 $I_t$ 均值是本步仍不合法（F→F）的 7.3 倍（SW）和 2.2 倍（TWX）。但 F→F 中 76% / 46% 是打转步，其 $I_t$ 接近 0。只与 F→F 中换了新动作的步比较，SW 约为 2.0 倍（95% CI 不含 1），TWX 约为 1.4 倍（CI 含 1，不显著）。见表 2。</li>
      <li><b>CoT 熵与结果的关系弱，且混有“上一步是否报错”。</b>
        整段 $\bar H^C_t$ 与各项结果的相关在 SW 均不显著，在 TWX 为弱负相关（$-0.12$ 至 $-0.18$）。CoT 前三分之一的熵与本步合法性的相关最强（$-0.227$ / $-0.293$），但它与上一步是否合法的相关更强（$-0.307$ / $-0.366$），按上一步合法性分层后只剩 $-0.08$ 至 $-0.18$。</li>
      <li><b>$D^{\mathrm{self}}$ 几乎处处为负，不能与带符号的权重相乘。</b>
        98.4% / 97.9% 的步 $\bar D^{\mathrm{self}}_t<0$，且随步数加深更负（$\rho=-0.349$ / $-0.276$）。token 级分布高度偏斜：中位数约为 $-0.003$ / $-0.004$，但熵越高的 token $D$ 越负；熵 $\le 0.05$ 的 token 中仍有 15% / 16% 满足 $D<-1$，贡献了全部负 gap 的 45% / 47%（表 4）。
        $C_t<0$ 的步占 80.1% / 82.8%；若 $w_{t,i}$ 随 $C_t$ 取负号，$w_{t,i}D_{t,i}$ 会在这些步上把更新方向反转。另需注意，teacher 上下文中历史轮只有 Action 行，$D<0$ 可能部分来自格式偏移，尚未用对照排除。</li>
    </ol>

    <h3 class="sub">表 2　上一步不合法之后：$I_t$ 与 CoT 熵</h3>
    <div class="table-wrap" id="recovery-table"></div>
    <p class="caption">F / T 表示上一步、本步动作是否合法。比值为 $I_t$ 均值之比，方括号内为 episode 聚类 bootstrap（2,000 次）95% CI。</p>

    <h2 class="sec"><span class="n">5</span>对方法设计的含义</h2>
    <p>设计中的 CoT 优势写作 $A^C_{t,i}=w_{t,i}D_{t,i}$，$w_{t,i}=f(C_t,H^C_{t,i})$。诊断数据对 $f$ 的约束如下。</p>
    <div class="table-wrap">
      <table>
        <thead><tr><th>环节</th><th>原设想</th><th>诊断证据</th><th>建议</th></tr></thead>
        <tbody>
          <tr><td>无 CoT 基线</td><td>空 Thought</td><td>空与中性基线一致（E13）</td><td>两者均可，报告中同时给出</td></tr>
          <tr><td>方向与幅度</td><td>$C_t=\hat A_t I_t$ 同时给出方向与幅度</td><td>$I_t>0$ 占 93.8% / 92.6%；$C_t\ne 0$ 时 $\operatorname{sign}(C_t)=\operatorname{sign}(\hat A_t)$ 的比例为 <span id="sign-agree"></span>；$|C_t|&lt;0.01$ 占 <span id="c-small"></span></td><td>$I_t$ 只用作非负门控 $g(I_t)$；方向由 $\hat A_t$ 给出，需改进 $\hat A_t$（critic 或 MC rollout）</td></tr>
          <tr><td>熵加权</td><td>按 $H^C_{t,i}$ 调制 token 权重</td><td>整段熵弱；前段熵混有上一步报错；两条熵假设在步级均无一致证据（表 3）</td><td>不先验加入；在训练中把熵乘在 RL 项上做消融</td></tr>
          <tr><td>OPD 项</td><td>$A^C_{t,i}=w_{t,i}D_{t,i}$</td><td>$D$ 几乎处处为负；$C_t&lt;0$ 的步占八成以上</td><td>$w\ge 0$；RL 与 OPD 相加；补格式对照 teacher 与 32B teacher</td></tr>
        </tbody>
      </table>
    </div>
    <p>据此建议在训练中比较如下形式，其中 $g(I_t)\in[0,1]$ 随 $I_t$ 单调增加：</p>
    $$A^C_{t,i}=\alpha\,g(I_t)\,\hat A_t+\beta\,D_{t,i}$$
    <p class="note">$\hat A_t$ 给出好坏的符号，$D_{t,i}$ 给出逐 token 的方向且不被翻转；CoT 不起作用的步（$I_t\approx 0$）不分到 RL 信号。该形式尚未经训练验证。</p>

    <h3 class="sub">表 3　按 $C_t$ 符号与 CoT 熵分组（检验两条熵假设）</h3>
    <p>假设 (a)：正向 CoT 中高熵 token 更值得更新；假设 (b)：负向 CoT 中低熵 token 应罚得更重。下表以 $|C_t|&gt;0.01$ 分组，熵的高低按任务内 $\bar H^C_t$ 中位数划分。</p>
    <div class="table-wrap" id="fch-table"></div>
    <p class="caption">(a) $C_t&gt;0.01$ 的步很少，且两边方向相反（SW 低熵组更好，TWX 高熵组略好）。(b) 在 $C_t&lt;-0.01$ 中，低熵组没有一致地更差：SW 合法率略低但 $G^{(3)}_t$ 更高，TWX 合法率更高。不合法步中熵最低的四分之一里，打转步占 <span id="loop-lowh"></span>，其余不合法步为 <span id="loop-other"></span>：“自信地错”多数是打转。</p>

    <h3 class="sub">表 4　token 级 $D^{\mathrm{self}}$ 按 CoT 熵分段</h3>
    <div class="table-wrap" id="token-table"></div>
    <p class="caption">“负 gap 份额”为该段 token 的 $\sum\max(-D,0)$ 占全部 CoT token 的比例。步内 token 熵与 $D$ 的 Pearson 相关均值为 <span id="turn-corr"></span>。</p>

    <h2 class="sec"><span class="n">6</span>局限</h2>
    <ul class="plain">
      <li>两边的动作合法率（42.5% / 59.9%）都低于 E01 设定的 60% 门槛；SW 三分之一的步是打转步。</li>
      <li>终局结果的样本很窄：mapreader 20/20 全胜，SW chemistry / test-conductivity 与 TWX cookingworld 均为 0/20，与成功率相关的结论主要来自 find-living-thing 与 coin。</li>
      <li>$\hat A_t$ 只是基于 3 步收益的代理，92%–95% 的即时奖励为 0。</li>
      <li>E14 人工盲标尚未完成，$I_t$ 在人看来度量的是什么尚未验证；32B strong teacher 与格式对照 teacher 尚未运行。</li>
      <li>结论均来自诊断数据，未经训练验证。</li>
    </ul>
  </div>
</section>

<!-- ============================ FIGURES ============================ -->
<section id="figures" class="tab-panel">
  <h2 class="sec"><span class="n">A</span>跨 Benchmark 对照与按任务统计</h2>
  <div class="fig-pair" style="align-items:start;">
    <div>
      <img id="cross-img" class="fig-img" alt="跨 Benchmark 相关性总览" />
      <p class="caption">Spearman 相关，episode 聚类 bootstrap（1,000 次）95% CI。25 个指标对中 24 个两边同号，唯一不同号的是 $H^A$ 与合法性。</p>
    </div>
    <div class="table-wrap" id="task-table" style="margin-top:0;"></div>
  </div>

  <h2 class="sec"><span class="n">B</span>诊断图（每个 Benchmark 10 张）</h2>
  <div class="fig-toolbar">
    <p class="note" style="margin:0;">点击图片放大。“读图”表示数值从图上读出，其余数值来自 <code>results/*/tables</code>。</p>
    <div class="seg" id="fig-mode">
      <button class="active" data-mode="both">并排</button>
      <button data-mode="sw">ScienceWorld</button>
      <button data-mode="tw">TextWorldExpress</button>
    </div>
  </div>
  <div id="fig-gallery"></div>

  <h2 class="sec"><span class="n">C</span>零奖励步分类（E11）</h2>
  <div class="table-wrap" id="e11-table"></div>
  <p class="caption">分类按顺序互斥：非法动作 → 原地重复（观察未变，或同一观察下重复同一动作）→ 有效铺垫（未来 5 步有奖励，或该局成功）→ 瞎逛（其余）。</p>

  <h2 class="sec"><span class="n">D</span>控制变量回归（E16）</h2>
  <div class="table-wrap" id="e16-table"></div>
  <p class="caption">OLS，含任务固定效应，控制步数、CoT 与动作长度及合法性；episode 聚类稳健标准误。* $p&lt;0.05$，** $p&lt;0.01$。</p>
</section>

<!-- ============================ EXPLORER ============================ -->
<section id="explorer" class="tab-panel">
  <div class="explorer">
    <aside class="ep-side">
      <div class="ep-filters">
        <div style="display:flex;justify-content:space-between;font-weight:600;"><span>全部轨迹</span><span id="ep-count" class="dim"></span></div>
        <select id="f-bench">
          <option value="all">全部 Benchmark</option>
          <option value="scienceworld">ScienceWorld</option>
          <option value="textworld_express">TextWorldExpress</option>
        </select>
        <select id="f-task"></select>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
          <select id="f-succ">
            <option value="all">全部结局</option>
            <option value="succ">成功</option>
            <option value="fail">失败</option>
          </select>
          <select id="f-rec">
            <option value="all">全部</option>
            <option value="has_rec">含 F→T 步</option>
          </select>
        </div>
        <input id="f-search" type="text" placeholder="搜索 episode、动作或 CoT" />
      </div>
      <div id="ep-list" class="ep-list"></div>
    </aside>

    <div>
      <div class="panel" id="ep-header"></div>
      <div class="panel">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:8px;">
          <div style="font-weight:600;">逐步指标（点击圆点跳到该步）</div>
          <div style="display:flex;align-items:center;gap:8px;font-family:var(--sans);font-size:12.5px;color:var(--muted);">
            CoT 着色
            <div class="seg" id="tok-mode">
              <button class="active" data-cmode="entropy">熵 H</button>
              <button data-cmode="dself">Gap D</button>
            </div>
          </div>
        </div>
        <div id="ep-timeline"></div>
        <div id="tok-legend" class="legend" style="margin-top:8px;"></div>
      </div>
      <div id="ep-turns"></div>
    </div>
  </div>
</section>

<!-- ============================ CASES ============================ -->
<section id="cases" class="tab-panel">
  <div class="prose">
    <h2 class="sec">案例</h2>
    <p class="note">以下各步均取自本次采集的数据。CoT 按逐 token 熵或 $D$ 着色，悬停查看数值。</p>
  </div>
  <div id="cases-box"></div>
</section>

<!-- ============================ SETUP ============================ -->
<section id="setup" class="tab-panel">
  <div class="prose">
    <h2 class="sec"><span class="n">1</span>实验设置</h2>
    <div class="table-wrap">
      <table>
        <tbody>
          <tr><td class="lbl">模型</td><td>Qwen3-8B，bf16，<code>enable_thinking=False</code>，显式 ReAct（<code>Thought:</code> / <code>Action:</code>）</td></tr>
          <tr><td class="lbl">Prompt</td><td>零样本：只给接口说明（命令格式、前置条件、效果），不给示范轨迹，也不附每步的合法动作列表</td></tr>
          <tr><td class="lbl">采样</td><td>vLLM，temperature 0.7，top_p 0.8，top_k 20；每步最多 256 token，每局最多 35 步</td></tr>
          <tr><td class="lbl">离线打分</td><td>HF Transformers，4 次前向：有 CoT、空 Thought、中性 Thought、self-teacher；熵在全词表上以 fp32 计算；沿用生成时的 token ids，并断言各次前向的动作 token 逐位一致</td></tr>
          <tr><td class="lbl">ScienceWorld</td><td>find-living-thing（L1）、chemistry-mix-paint-secondary-color（L2）、test-conductivity（L3），各 20 个 variation，无简化</td></tr>
          <tr><td class="lbl">TextWorldExpress</td><td>coin（L1，7 个地点）、mapreader（L2，11 个地点）、cookingworld（L3，5 个地点、3 种食材），dev fold，各 20 个种子</td></tr>
          <tr><td class="lbl">动作合法性</td><td>SW：执行后未出现失败回显；TWX：命中环境的合法动作（大小写不敏感）</td></tr>
          <tr><td class="lbl">硬件</td><td>1× A800-80GB；torch 2.11，vLLM 0.25.1，transformers 5.14.1</td></tr>
        </tbody>
      </table>
    </div>

    <h2 class="sec"><span class="n">2</span>E01–E18 完成情况</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>编号</th><th>内容</th><th>状态</th><th>说明</th></tr></thead>
        <tbody>
          <tr><td>E01</td><td>试水（每 Bench 5 局）</td><td><span class="tag warn">未达门槛</span></td><td>解析率 100%；合法率 28.0% / 53.1%，低于 60%；决定照常全量</td></tr>
          <tr><td>E02</td><td>正式采集（60 + 60 局）</td><td><span class="tag pos">完成</span></td><td>合法率 42.5% / 59.9%；打转步 33.5% / 14.6%</td></tr>
          <tr><td>E03–E04</td><td>$I_t$ 计算与分层</td><td><span class="tag pos">完成</span></td><td>图 1、图 2</td></tr>
          <tr><td>E05–E06</td><td>CoT 与动作的逐 token 熵</td><td><span class="tag pos">完成</span></td><td>图 3；token 级数据见 <code>token_stats.parquet</code></td></tr>
          <tr><td>E07–E08</td><td>奖励覆盖率与未来收益</td><td><span class="tag pos">完成</span></td><td>含当前步与严格未来两种口径</td></tr>
          <tr><td>E09</td><td>episode 聚类相关</td><td><span class="tag pos">完成</span></td><td>图 5</td></tr>
          <tr><td>E10</td><td>联合分组与无泄露预测</td><td><span class="tag pos">完成</span></td><td>图 6；结果部分受任务构成影响</td></tr>
          <tr><td>E11</td><td>零奖励步分类</td><td><span class="tag pos">完成</span></td><td>图 4</td></tr>
          <tr><td>E12</td><td>数据打包与报告</td><td><span class="tag pos">完成</span></td><td>本页与 JSONL / parquet 数据</td></tr>
          <tr><td>E13</td><td>空与中性基线</td><td><span class="tag pos">完成</span></td><td>图 7</td></tr>
          <tr><td>E14</td><td>人工语义盲标（每 Bench 80 步）</td><td><span class="tag">未完成</span></td><td>盲标 CSV 已导出，待人工标注</td></tr>
          <tr><td>E15</td><td>失败前 5 步</td><td><span class="tag pos">完成</span></td><td>图 8；SW 的下降趋势与步数深度混杂</td></tr>
          <tr><td>E16</td><td>控制变量回归</td><td><span class="tag pos">完成</span></td><td>图表页 D 节</td></tr>
          <tr><td>E17</td><td>CoT 内部位置熵</td><td><span class="tag pos">完成</span></td><td>图 9；混有上一步是否报错</td></tr>
          <tr><td>E18</td><td>Teacher gap</td><td><span class="tag warn">部分完成</span></td><td>self-teacher 完成；32B strong teacher 与格式对照未运行</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</section>
</main>

<div id="lightbox" class="lightbox" onclick="closeLightbox()"><img id="lightbox-img" src="" alt="放大的图" /></div>
<div id="tok-tip"></div>

<script>
const DATA = __REPORT_DATA__;

const KATEX_OPTS = {
  delimiters: [
    { left: "$$", right: "$$", display: true },
    { left: "$", right: "$", display: false },
  ],
  ignoredClasses: ["cot-box", "obs", "act-cmd", "ep-list", "no-math"],
  throwOnError: false,
};
function typeset(el) {
  if (el && window.renderMathInElement) renderMathInElement(el, KATEX_OPTS);
}

const state = {
  figMode: "both",
  selectedEpId: "textworld_express__coin__s1000",
  highlightTurn: null,
  tokMode: "entropy",
};
const ALL_EPISODES = [...DATA.sw.episodes, ...DATA.tw.episodes];
const EP_MAP = Object.fromEntries(ALL_EPISODES.map(e => [e.id, e]));
const BENCHES = [["sw", "ScienceWorld"], ["tw", "TextWorldExpress"]];
const CAT_ZH = {
  nonzero_reward: "有奖励",
  effective_setup: "有效铺垫",
  aimless_explore: "瞎逛",
  illegal_action: "非法动作",
  repeat_noop: "原地重复",
  parse_fail: "解析失败",
};

function esc(s) {
  return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
const fx = (v, d) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const pct = (v, d = 1) => (v == null ? "–" : `${Number(v).toFixed(d)}%`);
const ci = arr => `${fx(arr[0], 2)} <span class="ci">[${fx(arr[1], 2)}, ${fx(arr[2], 2)}]</span>`;

// ---------------- Tabs ----------------
const TAB_IDS = ["summary", "figures", "explorer", "cases", "setup"];
function showTab(id) {
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.toggle("active", b.dataset.tab === id));
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.toggle("active", p.id === id));
}
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    showTab(btn.dataset.tab);
    history.replaceState(null, "", "#" + btn.dataset.tab);
    window.scrollTo({ top: 0 });
  });
});

function openLightbox(src) {
  document.getElementById("lightbox-img").src = src;
  document.getElementById("lightbox").classList.add("open");
}
function closeLightbox() {
  document.getElementById("lightbox").classList.remove("open");
}
document.addEventListener("keydown", e => { if (e.key === "Escape") closeLightbox(); });

// ---------------- Summary ----------------
function renderSummary() {
  const a = DATA.sw.overview, b = DATA.tw.overview;
  const rows = [
    ["局数 / 步数", o => `${o.n_ep} / ${o.n_turn.toLocaleString()}`],
    ["平均每局步数", o => fx(o.avg_turns, 1)],
    ["格式解析成功率", o => pct(o.parse_pct)],
    ["动作合法率", o => pct(o.valid_pct)],
    ["终局成功率", o => pct(o.succ_pct)],
    ["平均终局得分（归一化）", o => fx(o.score_mean, 3)],
    ["$r_t>0$ / $r_t=0$ 的步", o => `${pct(o.r_pos_pct)} / ${pct(o.r_zero_pct)}`],
    ["打转步", o => pct(o.loop_pct)],
    ["$I_t$ 均值 / 中位数 / P90", o => `${fx(o.I_emp_mean, 3)} / ${fx(o.I_emp_p50, 3)} / ${fx(o.I_emp_p90, 2)}`],
    ["$I_t>0$ / $I_t<0$ 的步", o => `${pct(o.I_emp_pos_pct)} / ${pct(o.I_emp_neg_pct)}`],
    ["$r_t>0$ 的步的 $I_t$ 中位数", o => fx(o.r_pos_I_med, 4)],
    ["$\\bar H^C_t$ 均值（nats）", o => fx(o.H_C_mean, 3)],
    ["$\\bar H^A_t$ 均值（nats）；$\\bar H^A_t=0$ 的步", o => `${fx(o.H_A_mean, 3)}；${pct(o.H_A_zero_pct)}`],
    ["$\\bar D^{\\mathrm{self}}_t$ 均值；$\\bar D^{\\mathrm{self}}_t<0$ 的步", o => `${fx(o.D_self_mean, 2)}；${pct(o.D_neg_pct)}`],
  ];
  document.getElementById("stats-table").innerHTML = `
    <table>
      <thead><tr><th></th><th class="num">ScienceWorld</th><th class="num">TextWorldExpress</th></tr></thead>
      <tbody>${rows.map(([k, f]) => `<tr><td>${k}</td><td class="num">${f(a)}</td><td class="num">${f(b)}</td></tr>`).join("")}</tbody>
    </table>
    <p class="caption">表 1　基础统计。$I_t$ 为逐 token 平均的对数概率差（nats），$\\bar D^{\\mathrm{self}}_t$ 为一步内 token 的均值。</p>`;

  document.getElementById("rho-surp").textContent = `${fx(a.rho_surp_emp, 3)} / ${fx(b.rho_surp_emp, 3)}`;
  document.getElementById("sign-agree").textContent = `${pct(DATA.sw.fch.sign_agree_pct)} / ${pct(DATA.tw.fch.sign_agree_pct)}`;
  document.getElementById("c-small").textContent = `${pct(DATA.sw.fch.C_small_pct)} / ${pct(DATA.tw.fch.C_small_pct)}`;
  document.getElementById("loop-lowh").textContent = `${pct(DATA.sw.fch.loop_lowH_invalid_pct, 0)} / ${pct(DATA.tw.fch.loop_lowH_invalid_pct, 0)}`;
  document.getElementById("loop-other").textContent = `${pct(DATA.sw.fch.loop_other_invalid_pct, 0)} / ${pct(DATA.tw.fch.loop_other_invalid_pct, 0)}`;
  document.getElementById("turn-corr").textContent =
    `${fx(DATA.sw.fch.turn_corr_mean, 3)}（SW，${pct(DATA.sw.fch.turn_corr_neg_pct)} 的步为负）/ ${fx(DATA.tw.fch.turn_corr_mean, 3)}（TWX，${pct(DATA.tw.fch.turn_corr_neg_pct)}）`;

  // Table 2: transitions after an invalid step
  const REC_LABEL = {
    FF: "F→F", FF_loop: "其中：打转步", FF_new: "其中：换了新动作",
    FT: "F→T", TF: "T→F", TT: "T→T",
  };
  const recBy = k => Object.fromEntries(DATA[k].recovery.map(r => [r.key, r]));
  const rs = recBy("sw"), rt = recBy("tw");
  const recCells = r => `<td class="num">${r.n}</td><td class="num">${fx(r.I_mean, 3)}</td><td class="num">${fx(r.I_med, 3)}</td><td class="num">${fx(r.H_tail, 3)}</td>`;
  const recRow = k => `<tr class="${k.startsWith("FF_") ? "subrow" : ""}${k === "FT" ? " em" : ""}"><td>${REC_LABEL[k]}</td>${recCells(rs[k])}${recCells(rt[k])}</tr>`;
  const csw = DATA.sw.recovery_ci, ctw = DATA.tw.recovery_ci;
  document.getElementById("recovery-table").innerHTML = `
    <table>
      <thead>
        <tr class="group"><th></th><th colspan="4">ScienceWorld</th><th colspan="4">TextWorldExpress</th></tr>
        <tr><th>转移</th><th class="num">n</th><th class="num">$I_t$ 均值</th><th class="num">中位数</th><th class="num">$H^C$ tail</th><th class="num">n</th><th class="num">$I_t$ 均值</th><th class="num">中位数</th><th class="num">$H^C$ tail</th></tr>
      </thead>
      <tbody>
        ${["FF", "FF_loop", "FF_new", "FT", "TF", "TT"].map(recRow).join("")}
        <tr class="sep"><td>F→T 与 F→F 的均值比</td><td class="num" colspan="4">${ci(csw.FT_vs_FF)}</td><td class="num" colspan="4">${ci(ctw.FT_vs_FF)}</td></tr>
        <tr><td>F→T 与 F→F（新动作）的均值比</td><td class="num" colspan="4">${ci(csw.FT_vs_FF_new)}</td><td class="num" colspan="4">${ci(ctw.FT_vs_FF_new)}</td></tr>
      </tbody>
    </table>`;

  // Table 3: f(C, H) groups
  const GROUP_LABEL = {
    "pos|high": "$C_t>0.01$，高熵", "pos|low": "$C_t>0.01$，低熵",
    "neg|high": "$C_t<-0.01$，高熵", "neg|low": "$C_t<-0.01$，低熵",
  };
  const groupRows = BENCHES.map(([k, name]) => DATA[k].fch.groups.map((g, i) => `
    <tr class="${i === 0 && k === "tw" ? "sep" : ""}">
      <td>${i === 0 ? name : ""}</td><td>${GROUP_LABEL[g.c + "|" + g.h]}</td>
      <td class="num">${g.n}</td><td class="num">${fx(g.valid, 2)}</td><td class="num">${fx(g.G3s, 3)}</td>
      <td class="num">${fx(g.succ, 2)}</td><td class="num">${fx(g.loop, 2)}</td><td class="num">${fx(g.D, 2)}</td>
    </tr>`).join("")).join("");
  document.getElementById("fch-table").innerHTML = `
    <table>
      <thead><tr><th>Benchmark</th><th>组</th><th class="num">n</th><th class="num">合法率</th><th class="num">$G^{(3)}_t$</th><th class="num">成功率</th><th class="num">打转步</th><th class="num">$\\bar D^{\\mathrm{self}}_t$</th></tr></thead>
      <tbody>${groupRows}</tbody>
    </table>`;

  // Table 4: token-level D by entropy bin
  const bins = DATA.sw.fch.token_bins.map((b0, i) => [b0, DATA.tw.fch.token_bins[i]]);
  const binCells = b0 => `<td class="num">${pct(b0.share)}</td><td class="num">${fx(b0.d_med, 3)}</td><td class="num">${pct(b0.lt_m1)}</td><td class="num">${pct(b0.neg_mass)}</td>`;
  document.getElementById("token-table").innerHTML = `
    <table>
      <thead>
        <tr class="group"><th></th><th colspan="4">ScienceWorld</th><th colspan="4">TextWorldExpress</th></tr>
        <tr><th>$H^C_{t,i}$（nats）</th><th class="num">token 占比</th><th class="num">$D$ 中位数</th><th class="num">$D<-1$</th><th class="num">负 gap 份额</th><th class="num">token 占比</th><th class="num">$D$ 中位数</th><th class="num">$D<-1$</th><th class="num">负 gap 份额</th></tr>
      </thead>
      <tbody>${bins.map(([s, t]) => `<tr><td>${s.label}</td>${binCells(s)}${binCells(t)}</tr>`).join("")}</tbody>
    </table>`;

  typeset(document.getElementById("summary"));
}

// ---------------- Figures and tables ----------------
const VAR_TEX = {
  reward_raw: "$r_t$", G3_strict: "$G^{(3)}_t$", final_score_norm: "终局得分",
  I_empty_norm: "$I_t$", H_C_mean: "$\\bar H^C_t$", H_C_p90: "$H^C_{t,\\mathrm{P90}}$", action_valid_int: "合法性 $v_t$",
};

function renderFigures() {
  const cImg = document.getElementById("cross-img");
  cImg.src = DATA.cross_fig;
  cImg.onclick = () => openLightbox(DATA.cross_fig);

  const tasks = [...DATA.sw.tasks, ...DATA.tw.tasks];
  document.getElementById("task-table").innerHTML = `
    <table>
      <thead><tr><th>任务</th><th class="num">成功率</th><th class="num">得分</th><th class="num">合法率</th><th class="num">$I_t$ 中位数</th><th class="num">$\\rho(I_t,r_t)$</th><th class="num">$\\rho(H^C_{\\mathrm{head}},v_t)$</th></tr></thead>
      <tbody>${tasks.map((t, i) => `
        <tr class="${i === DATA.sw.tasks.length ? "sep" : ""}">
          <td>${t.benchmark === "ScienceWorld" ? "SW" : "TWX"} · ${esc(t.tier)} ${esc(t.task_id)}</td>
          <td class="num">${pct(t.succ_pct, 0)}</td><td class="num">${fx(t.score_mean, 2)}</td><td class="num">${pct(t.valid_pct, 0)}</td>
          <td class="num">${fx(t.I_med, 3)}</td><td class="num">${fx(t.rho_I_r, 3)}</td><td class="num">${fx(t.rho_Hhead_valid, 3)}</td>
        </tr>`).join("")}</tbody>
    </table>
    <p class="caption">任务内 Spearman 相关；mapreader 全胜，其 $I_t$ 中位数为 0。</p>`;

  const cats = [["effective_setup", "有效铺垫"], ["aimless_explore", "瞎逛"], ["illegal_action", "非法动作"], ["repeat_noop", "原地重复"]];
  const t11 = k => Object.fromEntries(DATA[k].tables.e11_zero_reward.map(r => [r.zero_reward_category, r]));
  const s11 = t11("sw"), w11 = t11("tw");
  const c11 = r => r ? `<td class="num">${r.n_turns}</td><td class="num">${fx(r.I_empty_mean, 2)} / ${fx(r.I_empty_median, 4)}</td><td class="num">${fx(r.H_C_mean, 3)}</td>` : `<td class="num" colspan="3">–</td>`;
  document.getElementById("e11-table").innerHTML = `
    <table>
      <thead>
        <tr class="group"><th></th><th colspan="3">ScienceWorld</th><th colspan="3">TextWorldExpress</th></tr>
        <tr><th>类别</th><th class="num">n</th><th class="num">$I_t$ 均值 / 中位数</th><th class="num">$\\bar H^C_t$</th><th class="num">n</th><th class="num">$I_t$ 均值 / 中位数</th><th class="num">$\\bar H^C_t$</th></tr>
      </thead>
      <tbody>${cats.map(([k, lbl]) => `<tr><td>${lbl}</td>${c11(s11[k])}${c11(w11[k])}</tr>`).join("")}</tbody>
    </table>`;

  const m16 = k => Object.fromEntries(DATA[k].tables.e16_controlled_regression.map(r => [`${r.outcome}|${r.term}`, r]));
  const s16 = m16("sw"), w16 = m16("tw");
  const pairs = [
    ["reward_raw", "I_empty_norm"], ["G3_strict", "I_empty_norm"], ["final_score_norm", "I_empty_norm"],
    ["reward_raw", "H_C_mean"], ["G3_strict", "H_C_p90"], ["final_score_norm", "H_C_mean"], ["reward_raw", "action_valid_int"],
  ];
  const reg = r => {
    if (!r) return "–";
    const star = r.p_value < 0.01 ? "**" : r.p_value < 0.05 ? "*" : "";
    return `${fx(r.coef, 4)}${star} <span class="ci">(p = ${fx(r.p_value, 3)})</span>`;
  };
  document.getElementById("e16-table").innerHTML = `
    <table>
      <thead><tr><th>因变量</th><th>自变量</th><th class="num">ScienceWorld</th><th class="num">TextWorldExpress</th></tr></thead>
      <tbody>${pairs.map(([o, t]) => `<tr><td>${VAR_TEX[o]}</td><td>${VAR_TEX[t]}</td><td class="num">${reg(s16[o + "|" + t])}</td><td class="num">${reg(w16[o + "|" + t])}</td></tr>`).join("")}</tbody>
    </table>`;

  document.querySelectorAll("#fig-mode button").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#fig-mode button").forEach(b => b.classList.toggle("active", b === btn));
      state.figMode = btn.dataset.mode;
      renderGallery();
    });
  });
  renderGallery();
  typeset(document.getElementById("figures"));
}

function renderGallery() {
  const mode = state.figMode;
  const html = DATA.fig_catalog.map((it, i) => {
    const sw = DATA.sw.figures[it.key], tw = DATA.tw.figures[it.key];
    const one = (src, name, note) => `
      <div>
        <div class="bench-name">${name}</div>
        <img class="fig-img" src="${src}" alt="${name} ${it.key}" loading="lazy" onclick="openLightbox(this.src)" />
        <p class="fig-note">${note}</p>
      </div>`;
    const body = mode === "both"
      ? `<div class="fig-pair">${one(sw, "ScienceWorld", it.sw_note)}${one(tw, "TextWorldExpress", it.tw_note)}</div>`
      : `<div class="fig-single">${mode === "sw" ? one(sw, "ScienceWorld", it.sw_note) : one(tw, "TextWorldExpress", it.tw_note)}</div>`;
    return `
      <figure class="fig" id="fig-${it.key}">
        <div class="fig-label">图 ${i + 1} · ${it.e_tag}</div>
        <div class="fig-title">${it.title}</div>
        ${body}
        <p class="fig-take"><b>解读：</b>${it.takeaway}</p>
      </figure>`;
  }).join("");
  const box = document.getElementById("fig-gallery");
  box.innerHTML = html;
  typeset(box);
}

// ---------------- Token heatmap ----------------
function tokenBg(tok, mode) {
  if (mode === "entropy") {
    const h = tok[1] || 0;
    if (h < 0.05) return "transparent";
    const t = Math.min(1, h / 1.5);
    return `rgba(196, 120, 20, ${(0.12 + t * 0.5).toFixed(2)})`;
  }
  const d = tok[3];
  if (d == null || Math.abs(d) < 0.05) return "transparent";
  if (d > 0) return `rgba(30, 107, 71, ${(0.12 + Math.min(1, d / 2.5) * 0.45).toFixed(2)})`;
  return `rgba(163, 38, 44, ${(0.10 + Math.min(1, -d / 3.5) * 0.5).toFixed(2)})`;
}
function renderTokens(toks, fallback, mode) {
  if (!toks || !toks.length) return esc(fallback);
  return toks.map(tk => {
    const ds = tk[3] == null ? "–" : tk[3].toFixed(3);
    return `<span class="tok" style="background:${tokenBg(tk, mode)}" data-t="${esc(tk[0])}" data-h="${tk[1].toFixed(3)}" data-lp="${tk[2].toFixed(3)}" data-ds="${ds}">${esc(tk[0])}</span>`;
  }).join("");
}
const tipEl = document.getElementById("tok-tip");
document.addEventListener("mousemove", e => {
  const el = e.target.closest(".tok");
  if (!el) { tipEl.style.display = "none"; return; }
  tipEl.innerHTML =
    `token ${esc(JSON.stringify(el.dataset.t))}<br/>` +
    `H = ${el.dataset.h} nats<br/>` +
    `log π_S = ${el.dataset.lp}<br/>` +
    `D = ${el.dataset.ds}`;
  tipEl.style.display = "block";
  tipEl.style.left = Math.min(window.innerWidth - 300, e.clientX + 14) + "px";
  tipEl.style.top = Math.min(window.innerHeight - 100, e.clientY + 14) + "px";
});
function tokenLegend(mode) {
  if (mode === "entropy") {
    return `<span>CoT 底色：token 熵 $H^C_{t,i}$（nats）</span>
      <span><span class="sw" style="background:transparent"></span>&lt; 0.05</span>
      <span><span class="sw" style="background:rgba(196,120,20,0.28)"></span>约 0.5</span>
      <span><span class="sw" style="background:rgba(196,120,20,0.62)"></span>≥ 1.5</span>`;
  }
  return `<span>CoT 底色：$D_{t,i}=\\log\\pi_T-\\log\\pi_S$</span>
    <span><span class="sw" style="background:rgba(30,107,71,0.45)"></span>$D>0$</span>
    <span><span class="sw" style="background:rgba(163,38,44,0.25)"></span>$D<0$</span>
    <span><span class="sw" style="background:rgba(163,38,44,0.6)"></span>$D\\ll 0$</span>`;
}

// ---------------- Explorer ----------------
function populateTaskFilter() {
  const b = document.getElementById("f-bench").value;
  const sel = document.getElementById("f-task");
  const cur = sel.value;
  const tasks = [...new Set(ALL_EPISODES.filter(e => b === "all" || e.bench === b).map(e => e.task))];
  sel.innerHTML = `<option value="all">全部任务</option>` + tasks.map(t => `<option value="${esc(t)}">${esc(t)}</option>`).join("");
  if (tasks.includes(cur)) sel.value = cur;
}

function renderEpisodeList() {
  const b = document.getElementById("f-bench").value;
  const t = document.getElementById("f-task").value;
  const s = document.getElementById("f-succ").value;
  const rec = document.getElementById("f-rec").value;
  const q = document.getElementById("f-search").value.trim().toLowerCase();
  const list = ALL_EPISODES.filter(e => {
    if (b !== "all" && e.bench !== b) return false;
    if (t !== "all" && e.task !== t) return false;
    if (s === "succ" && !e.succ) return false;
    if (s === "fail" && e.succ) return false;
    if (rec === "has_rec" && e.n_rec === 0) return false;
    if (q) {
      const hit = e.id.toLowerCase().includes(q) || e.turns.some(tr => tr.act.toLowerCase().includes(q) || tr.cot.toLowerCase().includes(q));
      if (!hit) return false;
    }
    return true;
  });
  document.getElementById("ep-count").textContent = `${list.length} / ${ALL_EPISODES.length}`;
  document.getElementById("ep-list").innerHTML = list.map(e => `
    <div class="ep-item ${e.id === state.selectedEpId ? "active" : ""}" onclick="selectEpisode('${e.id}')">
      <div class="t">${e.bench === "scienceworld" ? "SW" : "TWX"} · ${esc(e.task)} #${e.seed}</div>
      <div class="m">
        <span class="${e.succ ? "pos" : "neg"}">${e.succ ? "成功" : "失败"} ${fx(e.score, 2)}</span>
        <span>${e.n_turns} 步</span><span>不合法 ${e.inv}</span>
        ${e.n_rec > 0 ? `<span style="color:var(--violet)">F→T ${e.n_rec}</span>` : ""}
      </div>
    </div>`).join("") || `<div style="padding:18px;text-align:center;color:var(--muted);">无匹配轨迹</div>`;
}

function selectEpisode(id) {
  state.selectedEpId = id;
  state.highlightTurn = null;
  renderEpisodeList();
  renderEpisodeDetail();
}

function jumpToEpisode(id, turn = null) {
  if (!EP_MAP[id]) return;
  state.selectedEpId = id;
  state.highlightTurn = turn;
  ["f-bench", "f-succ", "f-rec"].forEach(x => { document.getElementById(x).value = "all"; });
  populateTaskFilter();
  document.getElementById("f-task").value = "all";
  document.getElementById("f-search").value = "";
  renderEpisodeList();
  renderEpisodeDetail();
  showTab("explorer");
  history.replaceState(null, "", "#explorer");
  if (turn !== null) setTimeout(() => scrollToTurn(turn), 60);
}

function timelineSVG(ep) {
  const ts = ep.turns;
  if (!ts.length) return "";
  const W = 880, H = 210, L = 48, R = 48, T = 22, B = 34;
  const cw = W - L - R, ch = H - T - B;
  const maxI = Math.max(2, ...ts.map(x => Math.max(0, x.ie ?? 0)));
  const maxH = Math.max(0.4, ...ts.map(x => x.hc ?? 0));
  const xp = i => L + (ts.length === 1 ? cw / 2 : (i / (ts.length - 1)) * cw);
  const yI = v => T + ch - (Math.max(0, v ?? 0) / maxI) * ch;
  const yH = v => T + ch - ((v ?? 0) / maxH) * ch;
  const pI = ts.map((tr, i) => `${i ? "L" : "M"}${xp(i).toFixed(1)},${yI(tr.ie).toFixed(1)}`).join(" ");
  const pH = ts.map((tr, i) => `${i ? "L" : "M"}${xp(i).toFixed(1)},${yH(tr.hc).toFixed(1)}`).join(" ");
  let marks = "";
  const every = Math.max(1, Math.ceil(ts.length / 18));
  ts.forEach((tr, i) => {
    const x = xp(i);
    const isRec = tr.pv === false && tr.valid;
    const fill = tr.r > 0 ? "#1e6b47" : isRec ? "#5b3f8c" : !tr.valid ? "#a3262c" : "#1f4e79";
    if (isRec) marks += `<rect x="${x - 6}" y="${T}" width="12" height="${ch}" fill="rgba(91,63,140,0.10)"/>`;
    marks += `<circle cx="${x.toFixed(1)}" cy="${yI(tr.ie).toFixed(1)}" r="${tr.r > 0 || isRec ? 5 : 3.5}" fill="${fill}" stroke="#fff" stroke-width="1.2" style="cursor:pointer" onclick="scrollToTurn(${tr.t})"><title>第 ${tr.t} 步：I=${tr.ie}，H=${tr.hc}，r=${tr.r}，合法=${tr.valid}</title></circle>`;
    marks += `<circle cx="${x.toFixed(1)}" cy="${yH(tr.hc).toFixed(1)}" r="2.4" fill="#b26b00" style="cursor:pointer" onclick="scrollToTurn(${tr.t})"><title>第 ${tr.t} 步：H=${tr.hc} nats</title></circle>`;
    if (i % every === 0 || i === ts.length - 1) marks += `<text x="${x.toFixed(1)}" y="${H - 10}" text-anchor="middle" font-size="10" fill="#8d897f">${tr.t}</text>`;
  });
  return `
    <svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;font-family:var(--sans);">
      <line x1="${L}" y1="${T + ch}" x2="${W - R}" y2="${T + ch}" stroke="#cfcbc0"/>
      <line x1="${L}" y1="${T}" x2="${L}" y2="${T + ch}" stroke="#cfcbc0"/>
      <line x1="${W - R}" y1="${T}" x2="${W - R}" y2="${T + ch}" stroke="#cfcbc0"/>
      <text x="${L - 6}" y="${T + 8}" text-anchor="end" font-size="10" fill="#1f4e79">${maxI.toFixed(1)}</text>
      <text x="${L - 6}" y="${T + ch}" text-anchor="end" font-size="10" fill="#1f4e79">0</text>
      <text x="${L}" y="13" font-size="11" fill="#1f4e79">左轴：Iₜ（实线）</text>
      <text x="${W - R + 6}" y="${T + 8}" font-size="10" fill="#b26b00">${maxH.toFixed(2)}</text>
      <text x="${W - R + 6}" y="${T + ch}" font-size="10" fill="#b26b00">0</text>
      <text x="${W - R}" y="13" text-anchor="end" font-size="11" fill="#b26b00">右轴：CoT 熵均值（虚线）</text>
      <path d="${pI}" fill="none" stroke="#1f4e79" stroke-width="1.8"/>
      <path d="${pH}" fill="none" stroke="#b26b00" stroke-width="1.5" stroke-dasharray="4 3"/>
      ${marks}
    </svg>
    <div class="legend" style="margin-top:6px;">
      <span><span class="sw dot" style="background:#1e6b47"></span>获得奖励</span>
      <span><span class="sw dot" style="background:#5b3f8c"></span>F→T（底纹）</span>
      <span><span class="sw dot" style="background:#a3262c"></span>不合法</span>
      <span><span class="sw dot" style="background:#1f4e79"></span>其他</span>
      <span>横轴为步序号</span>
    </div>`;
}

function scrollToTurn(t) {
  state.highlightTurn = t;
  document.querySelectorAll("#ep-turns .turn").forEach(el => el.classList.remove("hl"));
  const el = document.getElementById(`turn-${t}`);
  if (el) {
    el.classList.add("hl");
    el.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

function turnCard(tr, mode, prefix = "turn") {
  const isRec = tr.pv === false && tr.valid;
  const kind = isRec ? "k-rec" : tr.r > 0 ? "k-rew" : !tr.valid ? "k-inv" : "";
  const tags = [];
  if (tr.r > 0) tags.push(`<span class="tag pos">r = +${tr.r}</span>`);
  else if (tr.r < 0) tags.push(`<span class="tag neg">r = ${tr.r}</span>`);
  else tags.push(`<span class="tag">r = 0 · ${CAT_ZH[tr.cat] || esc(tr.cat)}</span>`);
  if (isRec) tags.push(`<span class="tag vio">F→T</span>`);
  else if (tr.pv === false && !tr.valid) tags.push(`<span class="tag neg">F→F${tr.lp ? " · 打转" : ""}</span>`);
  tags.push(tr.valid ? `<span class="tag">合法</span>` : `<span class="tag neg">不合法</span>`);
  const hl = prefix === "turn" && state.highlightTurn === tr.t ? " hl" : "";
  const dnf = tr.dnf != null ? `（负 token ${Math.round(tr.dnf * 100)}%）` : "";
  return `
    <div id="${prefix}-${tr.t}" class="turn ${kind}${hl}">
      <div class="turn-head">
        <div class="tags"><span class="tt">第 ${tr.t} 步</span>${tags.join("")}</div>
        <span class="right">累计得分 ${tr.sc} · $G^{(3)}_t=${tr.g3}$</span>
      </div>
      <details class="obs"><summary>观察</summary><pre>${esc(tr.obs)}</pre></details>
      <div class="cot-box"><span class="lbl-s">Thought</span>${renderTokens(tr.toks, tr.cot, mode)}</div>
      <div class="act-row">
        <div><span class="lbl-s">Action</span><span class="act-cmd">${esc(tr.act)}</span></div>
        <span class="note">$\\log\\pi(a_t\\mid h_t,z_t)=${fx(tr.lpc, 3)}$，$\\log\\pi(a_t\\mid h_t,\\varnothing)=${fx(tr.lpe, 3)}$</span>
      </div>
      <div class="metrics">
        <span>$I_t$ <b>${fx(tr.ie, 3)}</b></span>
        <span>$I^{\\mathrm{neu}}_t$ <b>${fx(tr.in, 3)}</b></span>
        <span>$\\bar H^C_t$ <b>${fx(tr.hc, 3)}</b></span>
        <span>head / tail <b>${fx(tr.hh, 3)} / ${fx(tr.ht, 3)}</b></span>
        <span>$\\bar H^A_t$ <b>${fx(tr.ha, 3)}</b></span>
        <span>$\\bar D^{\\mathrm{self}}_t$ <b>${fx(tr.ds, 3)}</b>${dnf}</span>
      </div>
    </div>`;
}

function renderEpisodeDetail() {
  const ep = EP_MAP[state.selectedEpId];
  if (!ep) return;
  document.getElementById("ep-header").innerHTML = `
    <div class="ep-head">
      <div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;">
          <span class="tag acc">${ep.bench_label}</span><span class="tag">${esc(ep.tier)}</span>
          <span class="tag ${ep.succ ? "pos" : "neg"}">${ep.succ ? "成功" : "失败（" + esc(ep.term) + "）"}</span>
          ${ep.n_rec > 0 ? `<span class="tag vio">F→T ${ep.n_rec} 次</span>` : ""}
        </div>
        <h3>${esc(ep.id)}</h3>
      </div>
      <div class="ep-stats">
        <div>终局得分 <b>${fx(ep.score, 3)}</b></div>
        <div>步数 <b>${ep.n_turns}</b></div>
        <div>不合法 <b>${ep.inv}</b></div>
      </div>
    </div>`;
  document.getElementById("ep-timeline").innerHTML = timelineSVG(ep);
  const legend = document.getElementById("tok-legend");
  legend.innerHTML = tokenLegend(state.tokMode);
  const box = document.getElementById("ep-turns");
  box.innerHTML = ep.turns.map(tr => turnCard(tr, state.tokMode)).join("");
  typeset(legend);
  typeset(box);
}

// ---------------- Cases ----------------
function renderCases() {
  const box = document.getElementById("cases-box");
  box.innerHTML = DATA.curated_cases.map((c, idx) => {
    const ep = EP_MAP[c.episode_id];
    if (!ep) return "";
    const turns = c.turns.map(t => ep.turns.find(x => x.t === t)).filter(Boolean).map(tr => turnCard(tr, c.mode, `case${idx}`)).join("");
    return `
      <div class="case">
        <div class="case-top">
          <div>
            <span class="fig-label">案例 ${idx + 1} · ${c.tag}</span>
            <h3>${c.title}</h3>
          </div>
          <button class="btn" onclick="jumpToEpisode('${c.episode_id}', ${c.turns[0]})">在轨迹浏览器中打开</button>
        </div>
        <p class="analysis">${c.analysis}</p>
        <div class="legend" style="margin-bottom:8px;">${tokenLegend(c.mode)}</div>
        ${turns}
      </div>`;
  }).join("");
  typeset(box);
}

// ---------------- Init ----------------
["f-bench", "f-task", "f-succ", "f-rec"].forEach(id => {
  document.getElementById(id).addEventListener("change", () => {
    if (id === "f-bench") populateTaskFilter();
    renderEpisodeList();
  });
});
document.getElementById("f-search").addEventListener("input", renderEpisodeList);
document.querySelectorAll("#tok-mode button").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#tok-mode button").forEach(b => b.classList.toggle("active", b === btn));
    state.tokMode = btn.dataset.cmode;
    renderEpisodeDetail();
  });
});

renderSummary();
renderFigures();
populateTaskFilter();
renderEpisodeList();
renderEpisodeDetail();
renderCases();
typeset(document.getElementById("setup"));
typeset(document.getElementById("cases"));
const initial = location.hash.slice(1);
if (TAB_IDS.includes(initial)) showTab(initial);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()

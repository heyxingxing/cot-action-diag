#!/usr/bin/env python3
"""
生成单文件、零外链、可离线直接双击打开的团队汇报 HTML 实验网站：
  results/experiment_report.html

包含 5 个交互标签页：
1. 核心结论总览（KPI + 5 大发现 + 报错自救对比 SVG + RL 拍板建议 + 明星案例一键跳转）
2. 可视化图表分析（全 21 张 300DPI 高清图，支持双 Bench 左右并排对比 + 每图精准图解 + 全屏放大灯箱 + E09/E11/E16 交互表）
3. 交互式全量轨迹浏览器（120 局 / 3,299 步 + 全轨迹 SVG 演变折线图 + 逐 Token 精确熵/Teacher Gap 热力图）
4. 精选典型案例剖析（5 大核心现象 × 双 Bench 真实切片，组会直接投屏讲）
5. 实验配置与 E01–E18 完成状态追溯
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RES_DIR = ROOT / "results"
OUT_HTML = RES_DIR / "experiment_report.html"


def b64_png(path: Path) -> str:
    if not path.exists():
        return ""
    raw = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_benchmark_bundle(bench_key: str, folder: str, label: str) -> Dict[str, Any]:
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
    tm_df = pd.DataFrame(tms)
    df = tm_df.merge(t_df, on=["episode_id", "turn_id"], how="left")
    df = df.sort_values(["episode_id", "turn_id"]).reset_index(drop=True)
    df["prev_valid"] = df.groupby("episode_id")["action_valid"].shift(1)
    df["prev_action"] = df.groupby("episode_id")["action"].shift(1)

    # Group CoT token stats by (episode_id, turn_id) into compact arrays: [text, entropy, logp_s, d_self]
    cot_tok = tok_df[tok_df["segment"] == "cot"].copy()
    cot_tok["d_self"] = cot_tok["logp_teacher_self"] - cot_tok["logp_student"]
    tok_map: Dict[tuple, List[List[Any]]] = {}
    for (eid, tid), grp in cot_tok.groupby(["episode_id", "turn_id"], sort=False):
        grp = grp.sort_values("token_pos")
        arr = []
        for txt, ent, lps, ds in zip(
            grp["token_text"].tolist(),
            grp["entropy_nats"].tolist(),
            grp["logp_student"].tolist(),
            grp["d_self"].tolist(),
        ):
            arr.append(
                [
                    str(txt),
                    round(float(ent), 3) if pd.notna(ent) else 0.0,
                    round(float(lps), 3) if pd.notna(lps) else 0.0,
                    round(float(ds), 3) if pd.notna(ds) else None,
                ]
            )
        tok_map[(str(eid), int(tid))] = arr

    # Compute per-task stats + within-task correlations
    task_stats = []
    for (tier, tid), g_ep in ep_df.groupby(["difficulty_tier", "task_id"]):
        g_t = df[df["task_id"] == tid]
        r_ir = stats.spearmanr(g_t["I_empty_norm"], g_t["reward_raw"], nan_policy="omit")[0]
        r_iv = stats.spearmanr(g_t["I_empty_norm"], g_t["action_valid"].astype(float), nan_policy="omit")[0]
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
                "turns_mean": round(float(g_ep["num_turns"].mean()), 1),
                "valid_pct": round(float(g_t["action_valid"].mean()) * 100, 1),
                "I_med": round(float(g_t["I_empty_norm"].median()), 4),
                "I_mean": round(float(g_t["I_empty_norm"].mean()), 3),
                "rho_I_r": round(float(r_ir), 3) if np.isfinite(r_ir) else None,
                "rho_I_valid": round(float(r_iv), 3) if np.isfinite(r_iv) else None,
                "rho_Hhead_valid": round(float(r_hhv), 3) if np.isfinite(r_hhv) else None,
            }
        )

    # Error-recovery 2x2 breakdown: (prev_valid, action_valid)
    rec_rows = []
    sub_pv = df.dropna(subset=["prev_valid"])
    for pv, av, tag in [
        (False, False, "报错后继续错 (F→F)"),
        (False, True, "报错后成功自救 (F→T)"),
        (True, False, "正常后突然出错 (T→F)"),
        (True, True, "连续正常推进 (T→T)"),
    ]:
        g = sub_pv[(sub_pv["prev_valid"] == pv) & (sub_pv["action_valid"] == av)]
        rec_rows.append(
            {
                "tag": tag,
                "pv": bool(pv),
                "av": bool(av),
                "n": int(len(g)),
                "I_mean": round(float(g["I_empty_norm"].mean()), 3) if len(g) else 0.0,
                "I_med": round(float(g["I_empty_norm"].median()), 4) if len(g) else 0.0,
                "H_head": round(float(g["H_C_head_mean"].mean()), 3) if len(g) else 0.0,
                "H_mid": round(float(g["H_C_mid_mean"].mean()), 3) if len(g) else 0.0,
                "H_tail": round(float(g["H_C_tail_mean"].mean()), 3) if len(g) else 0.0,
                "G5s_mean": round(float(g["G5_strict"].mean()), 3) if len(g) else 0.0,
            }
        )

    # Decomposition stats
    lp_cot = df["logp_a_with_cot"] / df["action_token_len"].clip(lower=1)
    lp_emp = df["logp_a_empty"] / df["action_token_len"].clip(lower=1)
    lp_neu = df["logp_a_neutral"] / df["action_token_len"].clip(lower=1)
    rho_surp_emp = float(stats.spearmanr(df["I_empty_norm"], -lp_emp, nan_policy="omit")[0])
    rho_surp_neu = float(stats.spearmanr(df["I_neutral_norm"], -lp_neu, nan_policy="omit")[0])

    # Sign conflict in C_proxy < 0
    c_px = df["A_turn_proxy"] * df["I_empty_norm"]
    neg_c = df[c_px < 0]
    dbl_neg_pct = round(float(neg_c["D_self_neg_frac"].mean()) * 100, 1) if len(neg_c) else 0.0

    # E13 agreement
    both_i = df[["I_empty_norm", "I_neutral_norm"]].dropna()
    e13_rho = round(float(stats.spearmanr(both_i["I_empty_norm"], both_i["I_neutral_norm"])[0]), 3)
    e13_sign = round(float(np.mean((both_i["I_empty_norm"] > 0) == (both_i["I_neutral_norm"] > 0))) * 100, 1)

    # Compact episodes + turns payload for interactive explorer (ALL 60 episodes!)
    ep_payload = []
    ep_meta_map = {e["episode_id"]: e for e in eps}
    for eid, g in df.groupby("episode_id", sort=True):
        meta = ep_meta_map[eid]
        g = g.sort_values("turn_id")
        n_rec = int(((g["prev_valid"] == False) & (g["action_valid"] == True)).sum())  # noqa: E712
        turns_list = []
        for _, r in g.iterrows():
            tid = int(r["turn_id"])
            act_len = max(1, int(r["action_token_len"]))
            turns_list.append(
                {
                    "t": tid,
                    "obs": str(r["observation"])[:900],
                    "cot": str(r["cot"]),
                    "act": str(r["action"]),
                    "valid": bool(r["action_valid"]),
                    "pv": None if pd.isna(r["prev_valid"]) else bool(r["prev_valid"]),
                    "r": round(float(r["reward_raw"]), 3),
                    "g3": round(float(r["G3_strict"]), 3),
                    "g5": round(float(r["G5_strict"]), 3),
                    "sc": round(float(r["score_norm"]), 3),
                    "ie": round(float(r["I_empty_norm"]), 4) if pd.notna(r["I_empty_norm"]) else None,
                    "in": round(float(r["I_neutral_norm"]), 4) if pd.notna(r["I_neutral_norm"]) else None,
                    "hc": round(float(r["H_C_mean"]), 4) if pd.notna(r["H_C_mean"]) else None,
                    "hh": round(float(r["H_C_head_mean"]), 4) if pd.notna(r["H_C_head_mean"]) else None,
                    "ht": round(float(r["H_C_tail_mean"]), 4) if pd.notna(r["H_C_tail_mean"]) else None,
                    "ha": round(float(r["H_A_mean"]), 4) if pd.notna(r["H_A_mean"]) else None,
                    "ds": round(float(r["D_self_mean"]), 3) if pd.notna(r["D_self_mean"]) else None,
                    "dnf": round(float(r["D_self_neg_frac"]), 3) if pd.notna(r["D_self_neg_frac"]) else None,
                    "apx": round(float(r["A_turn_proxy"]), 3) if pd.notna(r["A_turn_proxy"]) else None,
                    "lpc": round(float(r["logp_a_with_cot"]) / act_len, 4) if pd.notna(r["logp_a_with_cot"]) else None,
                    "lpe": round(float(r["logp_a_empty"]) / act_len, 4) if pd.notna(r["logp_a_empty"]) else None,
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
                "n_rec": n_rec,
                "turns": turns_list,
            }
        )

    # Load CSV tables
    tables = {}
    for tname in [
        "summary_stats",
        "e09_correlations",
        "e10_quadrants",
        "e10_leak_free_cv",
        "e11_zero_reward",
        "e16_controlled_regression",
    ]:
        p = rdir / "tables" / f"{tname}.csv"
        tables[tname] = pd.read_csv(p).to_dict(orient="records") if p.exists() else []

    # Load all 10 figures as base64
    fig_files = [
        ("Fig01", "Fig01_E04_bridge_dist.png"),
        ("Fig02", "Fig02_E04_E16_bridge_stratified.png"),
        ("Fig03", "Fig03_E05_E06_entropy_compare.png"),
        ("Fig04", "Fig04_E07_E11_reward_and_zero_cats.png"),
        ("Fig05", "Fig05_E09_clustered_correlations.png"),
        ("Fig06", "Fig06_E10_joint_quadrants.png"),
        ("Fig07", "Fig07_E13_baseline_robustness.png"),
        ("Fig08", "Fig08_E15_failure_trajectory.png"),
        ("Fig09", "Fig09_E17_cot_position_entropy.png"),
        ("Fig10", "Fig10_E18_teacher_gap.png"),
    ]
    figs = {k: b64_png(rdir / "figures" / fn) for k, fn in fig_files}

    overview = {
        "bench": bench_key,
        "label": label,
        "n_ep": int(len(ep_df)),
        "n_turn": int(len(df)),
        "avg_turns": round(float(ep_df["num_turns"].mean()), 1),
        "parse_pct": round(float(df["parse_ok"].mean()) * 100, 1),
        "valid_pct": round(float(df["action_valid"].mean()) * 100, 1),
        "succ_pct": round(float(ep_df["terminal_success"].mean()) * 100, 1),
        "score_mean": round(float(ep_df["final_score_norm"].mean()), 3),
        "r_pos_pct": round(float((df["reward_raw"] > 0).mean()) * 100, 1),
        "r_zero_pct": round(float((df["reward_raw"] == 0).mean()) * 100, 1),
        "r_neg_pct": round(float((df["reward_raw"] < 0).mean()) * 100, 1),
        "I_emp_mean": round(float(df["I_empty_norm"].mean()), 3),
        "I_emp_p50": round(float(df["I_empty_norm"].median()), 4),
        "I_emp_p90": round(float(df["I_empty_norm"].quantile(0.9)), 3),
        "I_emp_pos_pct": round(float((df["I_empty_norm"] > 0).mean()) * 100, 1),
        "I_emp_neg_pct": round(float((df["I_empty_norm"] < 0).mean()) * 100, 1),
        "I_neu_mean": round(float(df["I_neutral_norm"].mean()), 3),
        "I_neu_p50": round(float(df["I_neutral_norm"].median()), 4),
        "e13_rho": e13_rho,
        "e13_sign_pct": e13_sign,
        "H_C_mean": round(float(df["H_C_mean"].mean()), 4),
        "H_C_p90": round(float(df["H_C_p90"].mean()), 4),
        "H_A_mean": round(float(df["H_A_mean"].mean()), 4),
        "D_self_mean": round(float(df["D_self_mean"].mean()), 3),
        "D_self_pos_pct": round(float((df["D_self_mean"] > 0).mean()) * 100, 1),
        "dbl_neg_pct": dbl_neg_pct,
        "neg_C_turns": int(len(neg_c)),
        "lp_cot_p50": round(float(lp_cot.median()), 4),
        "rho_surp_emp": round(rho_surp_emp, 4),
        "rho_surp_neu": round(rho_surp_neu, 4),
        "r_pos_I_med": round(float(df[df["reward_raw"] > 0]["I_empty_norm"].median()), 4),
        "r_zero_I_med": round(float(df[df["reward_raw"] == 0]["I_empty_norm"].median()), 4),
    }

    return {
        "overview": overview,
        "tasks": task_stats,
        "recovery": rec_rows,
        "tables": tables,
        "figures": figs,
        "episodes": ep_payload,
    }


def main() -> None:
    print("[HTML Builder] Loading ScienceWorld & TextWorldExpress data + token stats + 21 PNGs...")
    sw = build_benchmark_bundle("scienceworld", "scienceworld_full", "ScienceWorld")
    tw = build_benchmark_bundle("textworld_express", "textworld_express_full", "TextWorldExpress")
    cross_fig = b64_png(RES_DIR / "cross_benchmark" / "Fig_cross_benchmark_overview.png")
    cross_e09 = pd.read_csv(RES_DIR / "cross_benchmark" / "cross_benchmark_e09.csv").to_dict(orient="records")

    # Curated case studies (verified real episode_id + turn_id pairs!)
    curated_cases = [
        {
            "category": "spotlight",
            "badge": "头条明星案例 · 一眼看懂信用分配错位",
            "color": "violet",
            "title": "TextWorldExpress `coin__s1000` 连续两步（Turn 18 → Turn 19）：真正救命的那步 r=0、I=5.39；摘果子那步 r=+1.0、I=0.000",
            "episode_id": "textworld_express__coin__s1000",
            "turns": [18, 19],
            "analysis": (
                "这是整批实验里最具代表性的两步对比："
                "在 <b>Turn 18</b>，模型刚因 <code>open dishwasher</code> 收到环境报错，CoT 立刻反思并切换搜索方向（<code>I_t = 5.39</code>，CoT 开头和尾段熵均高 <code>0.396 / 0.355</code>），执行 <code>move south</code> 进入客厅——<b>这一步真正靠 CoT 救活了整局，但当下奖励 r=0！</b>"
                "到了 <b>Turn 19</b>，进入客厅一眼看见硬币，模型执行显而易见的 <code>take coin</code> 直接拿到 <b>r=+1.0 通关</b>，但因为不写 CoT 也会捡眼前硬币，这一步 <b>I_t = 0.0000、H_A = 0.0000</b>。"
                "如果粗暴用 <code>r_t × I_t</code> 给 CoT 分配信号，这两步的乘积竟然<b>都是 0</b>！"
            ),
        },
        {
            "category": "recovery",
            "badge": "💡 报错后成功自救 (False → True)",
            "color": "emerald",
            "title": "ScienceWorld `find-living-thing__s5` (Turn 11)：报错后靠 CoT 重规划路线，I_t=3.83，随后 5 步拿 +25 分",
            "episode_id": "scienceworld__find-living-thing__s5",
            "turns": [10, 11],
            "analysis": (
                "Turn 10 直接尝试 <code>go to kitchen</code> 报错（温室不直通厨房）。"
                "Turn 11 CoT 在开头犹豫（<code>H_head = 0.214</code>）后明确推导出“必须先经走廊转去厨房”，以 <b>I_t = 3.83</b> 把动作掰到 <code>go to hallway</code>，成功跳出报错循环并在后续 5 步内收获 <b>G5_strict = +25 分</b>。"
            ),
        },
        {
            "category": "confident_error",
            "badge": "💥 低熵自信犯错 / 死循环 (False → False)",
            "color": "rose",
            "title": "ScienceWorld `chemistry-mix-paint__s6` (Turn 14) & TextWorld `coin__s1018` (Turn 14–15)：H_C ≈ 0.002–0.005 却连环非法操作",
            "episode_id": "scienceworld__chemistry-mix-paint-secondary-color__s6",
            "turns": [13, 14],
            "analysis": (
                "在连续报错后，模型并没有发散探索，反而陷入极低熵固执状态：<b>H_C = 0.0020 nats</b>（几乎 99.8% 自信！），CoT 煞有介事地复读 <i>\"The correct command is open door to bee hive.\"</i>，然后以 <code>I_t = 0.002</code> 继续撞同一堵墙。"
                "在 TextWorld <code>coin__s1018</code> 的 Turn 14–15 也有一模一样的现象：嘴上刚分析完“不是门所以不能 open”，下一秒以 <code>H_C = 0.0035</code> 自信执行 <code>open trash can</code> / <code>open dishwasher</code>！"
            ),
        },
        {
            "category": "obvious_reward",
            "badge": "🎯 拿分步为何 I_t ≈ 0？（揭秘负相关根源）",
            "color": "sky",
            "title": "ScienceWorld `test-conductivity__s3` (Turn 16)：一步拿 +50 分的关键动作，但 I_t = 0.0000",
            "episode_id": "scienceworld__test-conductivity__s3",
            "turns": [16],
            "analysis": (
                "模型已经在 workshop 里看到目标物体 aluminum foil，这一步执行 <code>focus on aluminum foil</code> 直接拿到 <b>+50 分</b>。"
                "但由于这个动作在当前画面下过于显而易见，有 CoT 时 <code>log π(a|h,z) ≈ 0</code>、无 CoT 时 <code>log π(a|h) ≈ 0</code>，导致 <b>I_t = 0.0000</b>。"
                "两个游戏里所有 <code>reward > 0</code> 步的 I_t 中位数都是 <code>0.0000</code>，直接解释了为什么全集上 I_t 与即时奖励呈 <b>-0.22 / -0.27 显著负相关</b>！"
            ),
        },
        {
            "category": "aimless_high_i",
            "badge": "🌀 高 I_t 却在瞎逛 / 走偏 (aimless_explore)",
            "color": "amber",
            "title": "TextWorld `cookingworld__s1007` (Turn 25)：全场最高 I_t = 20.78，只是 CoT 把模型带去检查无关台面",
            "episode_id": "textworld_express__cookingworld__s1007",
            "turns": [25],
            "analysis": (
                "这一步 <b>I_t 高达 20.78</b>（无 CoT 时模型完全不想选 <code>examine counter</code>，靠 CoT 硬掰过去），但其实只是长程做饭陷入瞎找食材的死胡同，最终超时失败。"
                "这直观说明：<b>高 I_t 只代表“CoT 强行改写了直觉动作”，绝不等价于“好 CoT”！</b>"
            ),
        },
        {
            "category": "sign_conflict",
            "badge": "⚠️ E18 符号反转实证（负负得正）",
            "color": "orange",
            "title": "ScienceWorld `test-conductivity__s16` (Turn 28)：负向 CoT 同时 D_self = -1.78，若相乘将变重奖",
            "episode_id": "scienceworld__test-conductivity__s16",
            "turns": [28],
            "analysis": (
                "在两个 Bench 所有被判为负向（<code>A_proxy × I_t < 0</code>）的 CoT 中，高达 <b>68.4%（SW）</b> 和 <b>65.4%（TW）</b> 的 token 同时满足 <code>D_self < 0</code>。"
                "把这一轮的着色切换到 <b>D_self</b> 视图就能直观看到满屏红色负值 token——如果照旧公式 <code>A_C = w · D</code> 乘上负权重，接近七成想惩罚的 token 会瞬间反转成正向强化！"
            ),
        },
    ]

    # Figure metadata + plain-Chinese interpretation for all 10 paired figures
    fig_catalog = [
        {
            "key": "Fig01",
            "e_tag": "E04",
            "title": "桥梁函数 I_t 整体分布 & 终局成败对比",
            "sw_note": "均值 0.840 但中位数仅 0.005（极度右偏），93.8% 为正；失败局反而出现更多高 I_t 长尾。",
            "tw_note": "均值 1.290、中位数 0.027，92.6% 为正；失败局（如 cookingworld）的 I_t 显著高于成功局。",
            "takeaway": "两个游戏完全一致：I_t 不是对称正态，而是“大量挤在 0 附近 + 少数超高正尾”，且 I_t < 0 仅占 2.7%–3.4%。",
        },
        {
            "key": "Fig02",
            "e_tag": "E04 + E16",
            "title": "桥梁函数 I_t 按任务难度、步数深度与动作长度分层",
            "sw_note": "L2/L3 难任务 I_t 分散度更大；随步数加深 I_t 略有变化，短动作（1–2 tok，如 inventory/wait）I_t 方差更大。",
            "tw_note": "mapreader（100% 赢）I_t 最低且紧贴 0；cookingworld（0% 赢）I_t 显著偏高——提示必须做 E16 控制任务难度！",
            "takeaway": "不同难度任务间 I_t 差异显著，因此跨任务混在一起看相关性会混入难度偏差，必须看任务内相关与 E16 回归。",
        },
        {
            "key": "Fig03",
            "e_tag": "E05 + E06",
            "title": "CoT 精确熵 (H_C) vs Action 精确熵 (H_A) 分布对照",
            "sw_note": "CoT 均值 0.153 nats（P90 0.471），而 Action 熵几乎全贴着 0（均值 0.015，中位数 0.000）。",
            "tw_note": "CoT 均值 0.163 nats（P90 0.502），Action 熵均值仅 0.026（中位数 0.000），两者分布量级差 6–10 倍。",
            "takeaway": "✅ 强力验证核心立项前提：模型几乎所有决策不确定性都在 CoT 里，写完 CoT 后吐 Action 几乎是 99% 确定性照抄。",
        },
        {
            "key": "Fig04",
            "e_tag": "E07 + E11",
            "title": "Reward 稀疏度覆盖率 & 零奖励五类动作大解剖",
            "sw_note": "92.0% 的步子 r_t=0；五类 0 奖励中，aimless_explore（瞎逛）I_t 最高（1.79），effective_setup 中位数仅 0.0008。",
            "tw_note": "94.8% 的步子 r_t=0；同样是 aimless_explore（2.05）I_t 最高，illegal_action CoT 熵最高（0.186），effective_setup 中位数 0.0003。",
            "takeaway": "两个游戏排序一模一样：0 奖励里“有效铺垫”其实很丝滑（低 I_t、低熵），真正把 I_t 顶上去的反而是“瞎逛”。",
        },
        {
            "key": "Fig05",
            "e_tag": "E09",
            "title": "核心指标与收益的 Episode-Cluster Bootstrap 相关性森林图",
            "sw_note": "I_empty/I_neutral 与 r_t 显著负相关（-0.217/-0.231）；H_C_mean 与 H_C_p90 的 95% CI 全部跨过 0（不显著）。",
            "tw_note": "I_t 和 H_C 几乎全部显著负相关（-0.11 ~ -0.30）；但按任务内控制后 H_C 对终局分不再显著。",
            "takeaway": "25 对跨 Bench 指标里有 24 对方向一致（几乎全负），彻底推翻“简单认为高 I_t = 好 CoT”的直觉。",
        },
        {
            "key": "Fig06",
            "e_tag": "E10",
            "title": "2×2 联合热力图 (H_C × I_t)、四象限拆解 & 无泄露预测 CV",
            "sw_note": "Low_HC & I<=0 合法率 84.2%、G3_strict=13.86 全面碾压 I>0 组；CV 里加 I_t×r_t 后预测成功 AUROC 反从 0.621 降到 0.596。",
            "tw_note": "I<=0 组成功率 57%–60% 远超 I>0 组 21%–22%；CV 里 I_t 看似 AUROC 0.667，但 E16 控制任务难度后 p=0.653 消失。",
            "takeaway": "❌ 原表四象限与 r_t×I_t 乘积直接拿来做 advantage 不成立，I<=0 根本不是“坏 CoT”，而是“顺风稳步”。",
        },
        {
            "key": "Fig07",
            "e_tag": "E13",
            "title": "无 CoT 基线稳健性：空 Thought: vs 中性 Thought:",
            "sw_note": "两者 Spearman=0.777、Pearson=0.904、正负号一致率 92.2%，均值差仅 -0.101。",
            "tw_note": "两者 Spearman=0.849、正负号一致率 87.1%，中性基线的 I_t 均值甚至略高于空基线（1.57 vs 1.29）。",
            "takeaway": "✅ 排除了“空 Thought: 格式错位”的担忧：I_t 高度偏正且与即时奖励负相关，是真实概率特性而非格式噪音。",
        },
        {
            "key": "Fig08",
            "e_tag": "E15",
            "title": "失败轨迹终止前最后 5 步的 CoT 熵与 I_t 演化",
            "sw_note": "52 局失败轨迹临近结束时，H_C_p90 从 0.50 稳步下降到 0.38（越到失败越固执/低熵），I_t 维持在 0.6–0.8。",
            "tw_note": "30 局失败轨迹最后一步 H_C_p90 从 0.56 骤降到 0.45，I_t 也从 1.7 降到 1.0。",
            "takeaway": "模型失败前并不是“越来越慌乱”，而是逐渐陷入“低熵麻木死循环”——这为针对性惩罚“低熵重复犯错”提供了依据。",
        },
        {
            "key": "Fig09",
            "e_tag": "E17",
            "title": "CoT 内部位置熵曲线（前 10% → 100%）& 汇总量预测力对比",
            "sw_note": "绝美 X 型交叉：非法动作在 CoT 开头 (10%-20%) 熵高达 0.28（合法的 2 倍），中后段反低于合法动作！H_head 相关 -0.227 吊打均值 -0.038。",
            "tw_note": "完全相同的 X 型交叉！非法动作开头熵 0.33 vs 合法 0.16，H_head 对合法性相关高达 -0.293，远强于均值 -0.149。",
            "takeaway": "💡 跨 Bench 最强细粒度规律：整段取平均会把开头“报错慌乱”和中后段“探索纠错”正负抵消！必须按位置分段看 CoT 熵。",
        },
        {
            "key": "Fig10",
            "e_tag": "E18",
            "title": "State-OPD Self-Teacher Gap (D_self) 分布与符号冲突检验",
            "sw_note": "98.4% 为负（均值 -1.124），随步数越深越负（rho=-0.35）；负向 CoT 中 68.4% token 同时 D_self<0！",
            "tw_note": "97.9% 为负（均值 -1.553），随步数越深越负（rho=-0.28）；负向 CoT 中 65.4% token 同时 D_self<0！",
            "takeaway": "❌ 两记重锤：① 8B 同模型删历史 CoT 会失忆；② 乘积公式 A=w·D 会把近 7 成错词负负得正变成重奖，OPD 必须改相加+强 Teacher。",
        },
    ]

    payload = {
        "sw": sw,
        "tw": tw,
        "cross_fig": cross_fig,
        "cross_e09": cross_e09,
        "curated_cases": curated_cases,
        "fig_catalog": fig_catalog,
    }
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # Escape </script> inside JSON safely
    data_json = data_json.replace("</", "<\\/")

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Qwen3-8B CoT→Action 长程信用分配诊断实验报告</title>
<style>
  :root {{
    --bg: #f8fafc;
    --surface: #ffffff;
    --surface-2: #f1f5f9;
    --border: #e2e8f0;
    --text: #0f172a;
    --muted: #475569;
    --faint: #94a3b8;
    --primary: #2563eb;
    --primary-soft: #eff6ff;
    --emerald: #059669;
    --emerald-soft: #ecfdf5;
    --rose: #e11d48;
    --rose-soft: #fff1f2;
    --amber: #d97706;
    --amber-soft: #fffbeb;
    --violet: #7c3aed;
    --violet-soft: #f5f3ff;
    --sky: #0284c7;
    --sky-soft: #f0f9ff;
    --radius: 12px;
    --shadow: 0 1px 3px rgba(15,23,42,.06), 0 1px 2px rgba(15,23,42,.04);
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    font-size: 14px;
  }}
  code {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    background: #f1f5f9;
    color: #0f172a;
    padding: 1px 5px;
    border-radius: 5px;
    font-size: 12.5px;
  }}
  /* Header */
  .hero {{
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #1e3a5f 100%);
    color: #f8fafc;
    padding: 26px 28px 20px;
    border-bottom: 1px solid #334155;
  }}
  .hero-inner {{ max-width: 1420px; margin: 0 auto; }}
  .hero h1 {{ margin: 0 0 8px; font-size: 23px; font-weight: 700; letter-spacing: -.01em; }}
  .hero-sub {{ color: #cbd5e1; font-size: 13.5px; margin: 0 0 14px; }}
  .pill-row {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .pill {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 500;
    background: rgba(255,255,255,.11); color: #e2e8f0; border: 1px solid rgba(255,255,255,.15);
  }}
  .pill.ok {{ background: rgba(16,185,129,.22); color: #a7f3d0; border-color: rgba(16,185,129,.35); }}

  /* Sticky Tabs */
  .nav-wrap {{
    position: sticky; top: 0; z-index: 40;
    background: rgba(255,255,255,.94); backdrop-filter: blur(8px);
    border-bottom: 1px solid var(--border);
  }}
  .nav-inner {{
    max-width: 1420px; margin: 0 auto; padding: 0 24px;
    display: flex; gap: 4px; overflow-x: auto;
  }}
  .tab-btn {{
    appearance: none; border: none; background: transparent; cursor: pointer;
    padding: 13px 16px; font-size: 14px; font-weight: 600; color: var(--muted);
    border-bottom: 2.5px solid transparent; white-space: nowrap; transition: all .15s;
  }}
  .tab-btn:hover {{ color: var(--text); }}
  .tab-btn.active {{ color: var(--primary); border-bottom-color: var(--primary); }}

  /* Main Container */
  .container {{ max-width: 1420px; margin: 0 auto; padding: 22px 24px 60px; }}
  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: block; }}

  /* Cards & Grids */
  .grid-4 {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }}
  .grid-2 {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
  .grid-3 {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
  @media (max-width: 1080px) {{
    .grid-4 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .grid-2, .grid-3 {{ grid-template-columns: 1fr; }}
  }}
  .card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 18px 20px; box-shadow: var(--shadow);
  }}
  .card + .card, .section {{ margin-top: 18px; }}
  .kpi-label {{ font-size: 12px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: .03em; }}
  .kpi-val {{ font-size: 26px; font-weight: 750; margin: 4px 0 4px; letter-spacing: -.02em; }}
  .kpi-sub {{ font-size: 12.5px; color: var(--muted); }}

  /* Badges */
  .badge {{
    display: inline-flex; align-items: center; padding: 2px 8px;
    border-radius: 999px; font-size: 12px; font-weight: 600; white-space: nowrap;
  }}
  .b-rose {{ background: var(--rose-soft); color: var(--rose); border: 1px solid #fecdd3; }}
  .b-emerald {{ background: var(--emerald-soft); color: var(--emerald); border: 1px solid #a7f3d0; }}
  .b-amber {{ background: var(--amber-soft); color: var(--amber); border: 1px solid #fde68a; }}
  .b-violet {{ background: var(--violet-soft); color: var(--violet); border: 1px solid #ddd6fe; }}
  .b-sky {{ background: var(--sky-soft); color: var(--sky); border: 1px solid #bae6fd; }}
  .b-slate {{ background: #f1f5f9; color: #334155; border: 1px solid #cbd5e1; }}
  .b-orange {{ background: #fff7ed; color: #c2410c; border: 1px solid #fed7aa; }}

  /* Finding Cards */
  .finding-card {{
    background: var(--surface); border: 1px solid var(--border); border-left: 4px solid var(--primary);
    border-radius: var(--radius); padding: 16px 20px; box-shadow: var(--shadow); margin-bottom: 14px;
  }}
  .finding-card.rose {{ border-left-color: var(--rose); }}
  .finding-card.emerald {{ border-left-color: var(--emerald); }}
  .finding-card.amber {{ border-left-color: var(--amber); }}
  .finding-card.violet {{ border-left-color: var(--violet); }}
  .finding-head {{ display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap; margin-bottom: 8px; }}
  .finding-title {{ font-size: 16px; font-weight: 700; margin: 0; }}
  .finding-body {{ color: #334155; font-size: 13.8px; margin: 0; }}
  .stat-strip {{
    display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; padding-top: 10px;
    border-top: 1px dashed var(--border);
  }}
  .stat-chip {{
    background: var(--surface-2); padding: 4px 10px; border-radius: 8px;
    font-size: 12.5px; color: var(--text);
  }}
  .stat-chip strong {{ font-variant-numeric: tabular-nums; }}

  /* Tables */
  .table-wrap {{ overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }}
  th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: left; }}
  th {{ background: var(--surface-2); font-weight: 600; color: #334155; position: sticky; top: 0; }}
  tr:hover td {{ background: #f8fafc; }}
  .num {{ text-align: right; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  .pos {{ color: var(--emerald); font-weight: 600; }}
  .neg {{ color: var(--rose); font-weight: 600; }}

  /* Section titles */
  .sec-title {{
    font-size: 17px; font-weight: 700; margin: 0 0 12px;
    display: flex; align-items: center; gap: 8px;
  }}
  .sec-desc {{ color: var(--muted); font-size: 13px; margin: -6px 0 14px; }}

  /* Controls & Buttons */
  .btn {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 12px; border-radius: 8px; font-size: 13px; font-weight: 600;
    border: 1px solid var(--border); background: var(--surface); color: var(--text);
    cursor: pointer; transition: all .12s;
  }}
  .btn:hover {{ background: var(--surface-2); border-color: #cbd5e1; }}
  .btn.primary {{ background: var(--primary); color: #fff; border-color: var(--primary); }}
  .btn.primary:hover {{ background: #1d4ed8; }}
  .btn.active {{ background: #0f172a; color: #fff; border-color: #0f172a; }}
  .btn-group {{ display: inline-flex; gap: 6px; flex-wrap: wrap; }}

  /* Figures */
  .fig-card {{
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 16px 18px; margin-bottom: 18px; box-shadow: var(--shadow);
  }}
  .fig-img {{
    width: 100%; height: auto; border-radius: 8px; border: 1px solid #e2e8f0;
    cursor: zoom-in; background: #fff; display: block;
  }}
  .fig-caption {{
    margin-top: 8px; padding: 8px 10px; background: var(--surface-2); border-radius: 8px;
    font-size: 12.8px; color: #334155;
  }}
  .fig-takeaway {{
    margin-top: 10px; padding: 9px 12px; background: var(--primary-soft); border: 1px solid #bfdbfe;
    border-radius: 8px; font-size: 13px; color: #1e3a8a; font-weight: 500;
  }}

  /* Trajectory Explorer */
  .explorer-layout {{
    display: grid; grid-template-columns: 330px minmax(0, 1fr); gap: 16px; align-items: start;
  }}
  @media (max-width: 1024px) {{ .explorer-layout {{ grid-template-columns: 1fr; }} }}
  .ep-sidebar {{
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
    box-shadow: var(--shadow); overflow: hidden; position: sticky; top: 62px;
    max-height: calc(100vh - 82px); display: flex; flex-direction: column;
  }}
  .ep-filters {{ padding: 12px; border-bottom: 1px solid var(--border); background: var(--surface-2); display: grid; gap: 8px; }}
  .ep-filters select, .ep-filters input {{
    width: 100%; padding: 6px 8px; border-radius: 7px; border: 1px solid #cbd5e1;
    font-size: 12.5px; background: #fff; color: var(--text);
  }}
  .ep-list {{ overflow-y: auto; flex: 1; }}
  .ep-item {{
    padding: 10px 12px; border-bottom: 1px solid #f1f5f9; cursor: pointer; transition: background .1s;
  }}
  .ep-item:hover {{ background: #f8fafc; }}
  .ep-item.active {{ background: var(--primary-soft); border-left: 3px solid var(--primary); }}
  .ep-item-title {{ font-weight: 600; font-size: 12.8px; word-break: break-all; }}
  .ep-item-meta {{ display: flex; gap: 6px; flex-wrap: wrap; margin-top: 4px; font-size: 11.5px; color: var(--muted); }}

  /* Turn Cards & Token Heatmap */
  .turn-card {{
    background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
    padding: 14px 16px; margin-bottom: 12px; transition: box-shadow .15s;
  }}
  .turn-card.highlight-turn {{ border-color: var(--violet); box-shadow: 0 0 0 3px rgba(124,58,237,.16); }}
  .turn-card.recovery-turn {{ border-left: 4px solid var(--violet); }}
  .turn-card.reward-turn {{ border-left: 4px solid var(--emerald); }}
  .turn-card.invalid-turn {{ border-left: 4px solid var(--rose); }}
  .turn-head {{
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;
    gap: 8px; padding-bottom: 8px; border-bottom: 1px solid #f1f5f9; margin-bottom: 10px;
  }}
  .obs-box {{
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 7px; padding: 8px 10px;
    font-size: 12.5px; color: #475569; white-space: pre-wrap; margin-bottom: 8px;
  }}
  .obs-box summary {{ cursor: pointer; font-weight: 600; color: #334155; }}
  .cot-box {{
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 12px;
    line-height: 1.9; font-size: 13.5px; margin-bottom: 8px;
  }}
  .tok {{
    display: inline; padding: 1px 1.5px; border-radius: 3px; cursor: default;
    border-bottom: 1px solid transparent; transition: outline .08s;
  }}
  .tok:hover {{ outline: 1.5px solid #0f172a; }}
  .action-bar {{
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 8px 12px;
  }}
  .action-cmd {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-weight: 700; font-size: 13.5px; }}
  .metric-pills {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }}
  .mpill {{
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11.8px;
    padding: 2px 7px; border-radius: 6px; background: #f1f5f9; color: #334155;
  }}

  /* Spotlight Callout */
  .spotlight {{
    background: linear-gradient(135deg, #f5f3ff 0%, #eff6ff 100%);
    border: 1.5px solid #c4b5fd; border-radius: var(--radius); padding: 18px 20px;
  }}

  /* Lightbox */
  .lightbox {{
    position: fixed; inset: 0; z-index: 100; background: rgba(15,23,42,.86);
    display: none; align-items: center; justify-content: center; padding: 24px; cursor: zoom-out;
  }}
  .lightbox.open {{ display: flex; }}
  .lightbox img {{ max-width: 96vw; max-height: 92vh; border-radius: 8px; background: #fff; box-shadow: 0 20px 60px rgba(0,0,0,.5); }}

  /* Floating Token Tooltip */
  #tok-tip {{
    position: fixed; z-index: 90; pointer-events: none; display: none;
    background: #0f172a; color: #f8fafc; padding: 7px 10px; border-radius: 8px;
    font-size: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    box-shadow: 0 8px 24px rgba(0,0,0,.35); line-height: 1.5; max-width: 280px;
  }}

  /* Color Legend */
  .legend-bar {{ display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--muted); flex-wrap: wrap; }}
  .swatch {{ display: inline-block; width: 14px; height: 14px; border-radius: 3px; border: 1px solid rgba(0,0,0,.1); vertical-align: middle; }}
</style>
</head>
<body>

<header class="hero">
  <div class="hero-inner">
    <h1>CoT → Action 影响驱动的长程 Agent 信号分配 · 诊断实验报告</h1>
    <p class="hero-sub">基于 <b>Qwen/Qwen3-8B</b> 在 ScienceWorld 与 TextWorldExpress 的全量离线诊断（E01–E18）—— 含跨 Bench 图表对比、全量 120 局真实轨迹与逐 Token 精确熵热力图</p>
    <div class="pill-row">
      <span class="pill">模型：Qwen/Qwen3-8B (BF16, enable_thinking=False)</span>
      <span class="pill">ScienceWorld：60 局 / 1,924 步</span>
      <span class="pill">TextWorldExpress：60 局 / 1,375 步</span>
      <span class="pill ok">格式解析率 100.0% · 零截断 · 零 NaN</span>
      <span class="pill ok">跨 Bench 相关性方向一致：24 / 25 对</span>
    </div>
  </div>
</header>

<nav class="nav-wrap">
  <div class="nav-inner">
    <button class="tab-btn active" data-tab="tab-overview">1. 核心结论总览</button>
    <button class="tab-btn" data-tab="tab-charts">2. 可视化图表分析 (21张图)</button>
    <button class="tab-btn" data-tab="tab-explorer">3. 交互式轨迹 & Token 熵浏览器 (120局)</button>
    <button class="tab-btn" data-tab="tab-cases">4. 精选典型案例剖析</button>
    <button class="tab-btn" data-tab="tab-protocol">5. 实验设置与 E01–E18 清单</button>
  </div>
</nav>

<main class="container">
  <!-- ===================== TAB 1: OVERVIEW ===================== -->
  <section id="tab-overview" class="tab-panel active">
    <div id="kpi-grid" class="grid-4"></div>

    <div class="spotlight section">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;margin-bottom:8px;">
        <div>
          <span class="badge b-violet">一句话核心结论</span>
          <h2 style="margin:6px 0 0;font-size:18px;">原始假设中“高 I_t = 好 CoT”与“A = w·D 相乘”被双 Bench 实证证伪；但实验挖出了更硬核的新信号——“报错后自救步的 I_t 暴涨 + CoT 分段位置熵”！</h2>
        </div>
        <button class="btn primary" onclick="jumpToEpisode('textworld_express__coin__s1000', 18)">打开头条案例 (coin__s1000 T18→19) →</button>
      </div>
      <p style="margin:0;color:#334155;font-size:13.5px;">
        顺风步里拿分动作太显而易见，没 CoT 模型也会选（<code>r_t &gt; 0</code> 步的中位数 <code>I_t = 0.0000</code>）；
        真正靠 CoT 力挽狂澜的时刻，是<b>上一步报错卡壳后（<code>False → True</code>），CoT 通过中后段高探索熵 + 暴涨的 I_t 把动作掰回正轨的那一步</b>（ScienceWorld I_t 暴涨 <b>7.3×</b>，TextWorldExpress 翻 <b>2.2×</b>）！
      </p>
    </div>

    <div class="section">
      <h2 class="sec-title">五大实证发现（双 Benchmark 交叉验证）</h2>
      <div id="findings-container"></div>
    </div>

    <div class="grid-2 section">
      <div class="card">
        <h3 class="sec-title">💡 核心新发现可视化：上一步报错后“继续错 vs 成功自救”对比</h3>
        <p class="sec-desc">跨 ScienceWorld 与 TextWorldExpress 完全重现：报错后成功跳出的步子，I_t 与 CoT 后段熵同步暴涨。</p>
        <div id="recovery-svg-box"></div>
        <div class="table-wrap" style="margin-top:10px;" id="recovery-table-box"></div>
      </div>

      <div class="card">
        <h3 class="sec-title">🧭 基于本批实验数据的下一步 RL 设计拍板建议</h3>
        <p class="sec-desc">每一条建议均直接由双 Bench 数据支撑，供组会讨论方法修订。</p>
        <div class="table-wrap">
          <table>
            <thead>
              <tr><th>设计环节</th><th>原方案构想</th><th>实证检验结论</th><th>建议修订方案</th></tr>
            </thead>
            <tbody>
              <tr>
                <td><b>无 CoT 基线</b><br>(E03/E13)</td>
                <td>空 <code>Thought:</code></td>
                <td><span class="badge b-emerald">✅ 可靠</span> 空/中性符号一致率 87–92%，非格式假象</td>
                <td>两者皆可用，优先用中性基线或 Qwen3 原生空思考做对照</td>
              </tr>
              <tr>
                <td><b>桥梁信号 I_t</b><br>(E03/E04/E10)</td>
                <td><code>C_t = A_t · I_t</code><br>直接当 CoT 奖惩</td>
                <td><span class="badge b-rose">❌ 失效</span> I_t ≈ No-CoT Surprisal (ρ=0.98)，与 r_t 在 6/6 任务全负相关；I&lt;0 仅 3%</td>
                <td><b>不可直接全局鼓励高 I_t</b>；改作“转折/纠错步门控”或结合未来收益 G_t，禁止 <code>(-,-)</code> 负负得正</td>
              </tr>
              <tr>
                <td><b>CoT 熵加权</b><br>(E05/E15/E17)</td>
                <td>用整段 H_C 均值或 P90 调制更新</td>
                <td><span class="badge b-amber">⚠️ 粒度过粗</span> Action 熵≈0 验证立项；但 CoT 开头与中后段呈 X 型交叉，均值互相抵消</td>
                <td>按位置/状态细分：<b>惩罚低熵死循环</b>（E15 失败前熵下降）+ 区分开头报错慌乱 vs 中后段有效探索</td>
              </tr>
              <tr>
                <td><b>Teacher OPD</b><br>(E18)</td>
                <td><code>A_C = w · D</code><br>8B 删历史 CoT</td>
                <td><span class="badge b-rose">❌ 符号反转+失忆</span> 负向 CoT 中 65–68% token 满足 D&lt;0；8B 删历史 98% 为负</td>
                <td><b>绝不可相乘</b>，改为 RL 与 OPD <b>相加</b>（<code>α·A_RL + β·D</code>）；并测 <code>Qwen3-32B</code> 强 Teacher</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </section>

  <!-- ===================== TAB 2: CHARTS ===================== -->
  <section id="tab-charts" class="tab-panel">
    <div class="card">
      <h2 class="sec-title">跨 Benchmark 相关性总览（ScienceWorld vs TextWorldExpress）</h2>
      <p class="sec-desc">不比较两边原始 Reward 绝对值，严格按 Episode-Cluster Bootstrap（1,000 次重采样）对比无量纲 Spearman 相关系数。25 组里 24 组方向完全一致。</p>
      <div class="grid-2" style="align-items:center;">
        <div>
          <img id="cross-overview-img" class="fig-img" alt="Cross-Benchmark Overview" />
        </div>
        <div class="table-wrap" style="max-height:420px;overflow-y:auto;" id="task-summary-table"></div>
      </div>
    </div>

    <div class="card section">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;margin-bottom:14px;">
        <div>
          <h2 class="sec-title" style="margin-bottom:4px;">全套 20 张实验诊断图逐对解析（Fig 01 – Fig 10）</h2>
          <p style="margin:0;color:var(--muted);font-size:13px;">点击任意图片可<b>全屏高清放大</b>；支持双 Bench 左右并排对照或单独查看。</p>
        </div>
        <div class="btn-group" id="chart-mode-btns">
          <button class="btn active" data-mode="both">双 Bench 并排对比</button>
          <button class="btn" data-mode="sw">仅 ScienceWorld</button>
          <button class="btn" data-mode="tw">仅 TextWorldExpress</button>
        </div>
      </div>
      <div id="fig-gallery"></div>
    </div>

    <div class="grid-2 section">
      <div class="card">
        <h3 class="sec-title">E11 零奖励五分类量化对比（SW vs TW）</h3>
        <div class="table-wrap" id="e11-compare-table"></div>
      </div>
      <div class="card">
        <h3 class="sec-title">E16 控制变量回归摘要（Episode 聚类稳健标准误）</h3>
        <p class="sec-desc">控制任务类别、步数深度、CoT/Action 长度与动作合法性后，观察 I_t 与 H_C 是否仍显著。</p>
        <div class="table-wrap" id="e16-compare-table"></div>
      </div>
    </div>
  </section>

  <!-- ===================== TAB 3: TRAJECTORY EXPLORER ===================== -->
  <section id="tab-explorer" class="tab-panel">
    <div class="explorer-layout">
      <aside class="ep-sidebar">
        <div class="ep-filters">
          <div style="font-weight:700;font-size:13px;display:flex;justify-content:space-between;">
            <span>全部真实轨迹 (120 局)</span>
            <span id="ep-count-badge" style="color:var(--primary);">120</span>
          </div>
          <select id="f-bench">
            <option value="all">全部 Benchmark (SW + TW)</option>
            <option value="scienceworld">ScienceWorld (60 局)</option>
            <option value="textworld_express">TextWorldExpress (60 局)</option>
          </select>
          <select id="f-task"><option value="all">全部任务类别</option></select>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
            <select id="f-succ">
              <option value="all">全部结局</option>
              <option value="succ">仅成功局</option>
              <option value="fail">仅失败局</option>
            </select>
            <select id="f-rec">
              <option value="all">全部轨迹</option>
              <option value="has_rec">含报错自救步</option>
            </select>
          </div>
          <input id="f-search" type="text" placeholder="搜索 episode / 动作 / CoT 关键词..." />
        </div>
        <div id="ep-list" class="ep-list"></div>
      </aside>

      <div id="ep-detail">
        <div class="card" id="ep-header-card"></div>
        <div class="card section" id="ep-chart-card">
          <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:8px;">
            <div>
              <h3 class="sec-title" style="margin:0;font-size:15px;">全轨迹指标演变图（点击任意步跳转详情）</h3>
            </div>
            <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
              <span style="font-size:12px;color:var(--muted);">CoT 逐词热力着色：</span>
              <div class="btn-group" id="tok-color-mode">
                <button class="btn active" data-cmode="entropy" style="padding:3px 9px;font-size:12px;">精确熵 H_C</button>
                <button class="btn" data-cmode="dself" style="padding:3px 9px;font-size:12px;">Teacher Gap D_self</button>
              </div>
            </div>
          </div>
          <div id="ep-timeline-svg"></div>
          <div id="tok-legend" class="legend-bar" style="margin-top:8px;"></div>
        </div>
        <div id="ep-turns-list" class="section"></div>
      </div>
    </div>
  </section>

  <!-- ===================== TAB 4: CURATED CASES ===================== -->
  <section id="tab-cases" class="tab-panel">
    <div class="card" style="margin-bottom:16px;">
      <h2 class="sec-title" style="margin-bottom:4px;">精选 6 大真实轨迹案例剖析（可直接用于组会投屏讲解）</h2>
      <p style="margin:0;color:var(--muted);font-size:13px;">所有案例均直接抽自本次 Qwen3-8B 真实采集数据，完整呈现逐 Token 精确熵热力图（悬停任意词查看精确数值），点击右上角按钮可一键跳转查看该局完整上下文。</p>
    </div>
    <div id="cases-container"></div>
  </section>

  <!-- ===================== TAB 5: PROTOCOL ===================== -->
  <section id="tab-protocol" class="tab-panel">
    <div class="grid-2">
      <div class="card">
        <h2 class="sec-title">核心实验参数与防泄露设计</h2>
        <div class="table-wrap">
          <table>
            <tbody>
              <tr><td><b>基座模型</b></td><td><code>Qwen/Qwen3-8B</code> (BF16, <code>enable_thinking=False</code>, ReAct <code>Thought:/Action:</code>)</td></tr>
              <tr><td><b>硬件平台</b></td><td>1× NVIDIA A800-SXM4-80GB (`torch 2.11.0+cu130`, `vllm 0.25.1`, `transformers 5.14.1`)</td></tr>
              <tr><td><b>Stage 1 Rollout</b></td><td>vLLM + Prefix Caching, <code>temp=0.7, top_p=0.8, top_k=20, presence_penalty=0.0</code>, <code>max_tokens=256</code>, <code>max_turns=35</code></td></tr>
              <tr><td><b>Stage 2 离线打分</b></td><td>HF Transformers, <code>scoring_temperature=1.0</code>（无截断）, <code>batch_size=1</code>（零 BF16 padding 抖动）, 目标切片 FP32 算精确全词表熵（单位 nats）</td></tr>
              <tr><td><b>Token 边界保护</b></td><td>直接沿用生成 <code>token_ids</code>，严格剥离 <code>Thought:</code> 与 <code>\nAction:</code> 模板字，强断言保证 Pass 1/2/3 Action token ID 逐位一致</td></tr>
              <tr><td><b>ScienceWorld 任务集</b></td><td>原版无简化 (<code>simplificationStr=""</code>), L1 <code>find-living-thing</code> / L2 <code>chemistry-mix-paint-secondary-color</code> / L3 <code>test-conductivity</code> × 20 var</td></tr>
              <tr><td><b>TextWorldExpress 任务集</b></td><td><code>gameFold="dev"</code>（保留 train/test 给未来 RL）, L1 <code>coin(7)</code> / L2 <code>mapreader(11)</code> / L3 <code>cookingworld(5,3)</code> × 20 seeds</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <div class="card">
        <h2 class="sec-title">E01 – E18 诊断清单完成进度</h2>
        <div class="table-wrap">
          <table>
            <thead><tr><th>编号</th><th>实验项目</th><th>状态</th><th>核心产出</th></tr></thead>
            <tbody>
              <tr><td><b>E01–E02</b></td><td>环境试水 + 120 局正式批量采集</td><td><span class="badge b-emerald">✅ 已完成</span></td><td>解析率 100%，零截断</td></tr>
              <tr><td><b>E03–E04</b></td><td>桥梁函数 I_t 计算与分层分布</td><td><span class="badge b-emerald">✅ 已完成</span></td><td>Fig01, Fig02 + 分解证明</td></tr>
              <tr><td><b>E05–E06</b></td><td>CoT 与 Action 全词表精确熵</td><td><span class="badge b-emerald">✅ 已完成</span></td><td>Fig03 + 13 万 token parquet</td></tr>
              <tr><td><b>E07–E08</b></td><td>Reward 覆盖率 + 未来 G3/G5</td><td><span class="badge b-emerald">✅ 已完成</span></td><td>含当前 & 严格未来两套</td></tr>
              <tr><td><b>E09</b></td><td>按 Episode 聚类相关性检验</td><td><span class="badge b-emerald">✅ 已完成</span></td><td>Fig05 + 1000次 Bootstrap CI</td></tr>
              <tr><td><b>E10</b></td><td>两指标联合 + 无泄露预测 CV</td><td><span class="badge b-emerald">✅ 已完成</span></td><td>Fig06 + 4象限拆解</td></tr>
              <tr><td><b>E11</b></td><td>零奖励五分类自动化解剖</td><td><span class="badge b-emerald">✅ 已完成</span></td><td>Fig04 + 跨 Bench 排序一致</td></tr>
              <tr><td><b>E12</b></td><td>统一交付打包与异常案例</td><td><span class="badge b-emerald">✅ 已完成</span></td><td>本交互式 HTML 报告 + JSONL</td></tr>
              <tr><td><b>E13</b></td><td>空 vs 中性无 CoT 基线稳健性</td><td><span class="badge b-emerald">✅ 已完成</span></td><td>Fig07 (符号一致 87–92%)</td></tr>
              <tr><td><b>E14</b></td><td>每 Bench 80 步人工语义盲标</td><td><span class="badge b-amber">⏳ 已导出盲标CSV</span></td><td>待团队人工填完 3 列 0/1</td></tr>
              <tr><td><b>E15</b></td><td>失败前 5 步长程演化曲线</td><td><span class="badge b-emerald">✅ 已完成</span></td><td>Fig08 (揭示低熵麻木趋势)</td></tr>
              <tr><td><b>E16</b></td><td>控制变量聚类稳健回归</td><td><span class="badge b-emerald">✅ 已完成</span></td><td>控制难度/步数/长度回归表</td></tr>
              <tr><td><b>E17</b></td><td>CoT 位置熵（开头/中部/结尾）</td><td><span class="badge b-emerald">✅ 已完成</span></td><td>Fig09 (发现跨Bench X型交叉)</td></tr>
              <tr><td><b>E18</b></td><td>State-OPD Self-Teacher Gap</td><td><span class="badge b-emerald">✅ 已完成</span></td><td>Fig10 (实锤65-68%符号反转)</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </section>
</main>

<!-- Fullscreen Lightbox & Token Tooltip -->
<div id="lightbox" class="lightbox" onclick="closeLightbox()"><img id="lightbox-img" src="" alt="Zoomed Figure" /></div>
<div id="tok-tip"></div>

<script>
const REPORT_DATA = {data_json};

const state = {{
  chartMode: "both",
  selectedEpId: REPORT_DATA.tw.episodes.find(e => e.id === "textworld_express__coin__s1000")?.id || REPORT_DATA.sw.episodes[0].id,
  highlightTurn: null,
  tokColorMode: "entropy",
}};

const ALL_EPISODES = [...REPORT_DATA.sw.episodes, ...REPORT_DATA.tw.episodes];
const EP_MAP = Object.fromEntries(ALL_EPISODES.map(e => [e.id, e]));

function esc(s) {{
  return String(s ?? "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}}

// ================= Tab Switching =================
document.querySelectorAll(".tab-btn").forEach(btn => {{
  btn.addEventListener("click", () => {{
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
    window.scrollTo({{ top: 0, behavior: "smooth" }});
  }});
}});

function openLightbox(src) {{
  document.getElementById("lightbox-img").src = src;
  document.getElementById("lightbox").classList.add("open");
}}
function closeLightbox() {{
  document.getElementById("lightbox").classList.remove("open");
}}
document.addEventListener("keydown", e => {{ if (e.key === "Escape") closeLightbox(); }});

function jumpToEpisode(epId, turnId = null) {{
  if (!EP_MAP[epId]) return;
  state.selectedEpId = epId;
  state.highlightTurn = turnId;
  // Reset filters so episode is visible
  document.getElementById("f-bench").value = "all";
  populateTaskFilter();
  document.getElementById("f-task").value = "all";
  document.getElementById("f-succ").value = "all";
  document.getElementById("f-rec").value = "all";
  document.getElementById("f-search").value = "";
  renderEpisodeList();
  renderEpisodeDetail();
  document.querySelector('[data-tab="tab-explorer"]').click();
  if (turnId !== null) {{
    setTimeout(() => {{
      const el = document.getElementById(`turn-card-${{turnId}}`);
      if (el) el.scrollIntoView({{ behavior: "smooth", block: "center" }});
    }}, 80);
  }}
}}

// ================= Tab 1: Overview =================
function renderOverview() {{
  const sw = REPORT_DATA.sw.overview;
  const tw = REPORT_DATA.tw.overview;

  document.getElementById("kpi-grid").innerHTML = `
    <div class="card">
      <div class="kpi-label">总诊断样本规模</div>
      <div class="kpi-val">${{sw.n_ep + tw.n_ep}} 局 · ${{(sw.n_turn + tw.n_turn).toLocaleString()}} 步</div>
      <div class="kpi-sub">SW: ${{sw.n_ep}}局/${{sw.n_turn}}步 (均${{sw.avg_turns}}步) · TW: ${{tw.n_ep}}局/${{tw.n_turn}}步 (均${{tw.avg_turns}}步)</div>
    </div>
    <div class="card">
      <div class="kpi-label">任务表现与奖励稀疏度</div>
      <div class="kpi-val">SW ${{sw.succ_pct}}% / TW ${{tw.succ_pct}}%</div>
      <div class="kpi-sub">0 奖励步占比：SW <b>${{sw.r_zero_pct}}%</b> · TW <b>${{tw.r_zero_pct}}%</b>（合法率 ${{sw.valid_pct}}% / ${{tw.valid_pct}}%）</div>
    </div>
    <div class="card">
      <div class="kpi-label">I_t ≈ No-CoT Surprisal</div>
      <div class="kpi-val">ρ = ${{sw.rho_surp_emp}} / ${{tw.rho_surp_emp}}</div>
      <div class="kpi-sub">Action 熵中位数 <b>0.000</b>，所有拿分步 (r&gt;0) I_t 中位数均为 <b>0.0000</b></div>
    </div>
    <div class="card">
      <div class="kpi-label">原公式 A=w·D 符号反转风险</div>
      <div class="kpi-val" style="color:var(--rose);">${{sw.dbl_neg_pct}}% / ${{tw.dbl_neg_pct}}%</div>
      <div class="kpi-sub">负向 CoT 中同时 D_self&lt;0 的 token 比例（误乘负权重将重奖错词）</div>
    </div>
  `;

  const findings = [
    {{
      cls: "rose",
      badge: '<span class="badge b-rose">❌ 证伪原假设：高 I_t ≠ 好 CoT</span>',
      title: "1. 桥梁函数 I_t 在两个 Bench、全部 6 个任务里均与即时奖励显著负相关，本质是“No-CoT 惊奇度”",
      body: "拆解公式后发现：模型写完 Thought 后，写 Action 几乎是 99% 确定性照抄答案（<code>log π_A(a|h,z)</code> 与 <code>H_A</code> 中位数均为 <b>0.0000</b>），导致 <code>I_t ≈ -log π_A(a|h)/|a|</code>（Spearman 高达 <b>0.976 / 0.973</b>）。拿分的临门一脚（如 <code>take coin</code>、<code>focus on foil</code>）太显而易见，不写 CoT 也会做，所以 <b>r&gt;0 步中位数 I_t = 0.0000</b>；而高 I_t（3–20）大量落在迷茫瞎逛（E11 <code>aimless_explore</code> 均值第一高）。E13 验证空/中性基线符号一致率高达 <b>92.2% / 87.1%</b>，证明这是定义本身特性，非格式噪音。",
      chips: [
        `SW I_t vs r_t: <b>-0.217</b>`,
        `TW I_t vs r_t: <b>-0.268</b>`,
        `6/6 任务内部全负: <b>-0.15 ~ -0.28</b>`,
        `E13 符号一致: <b>${{sw.e13_sign_pct}}% / ${{tw.e13_sign_pct}}%</b>`,
      ]
    }},
    {{
      cls: "rose",
      badge: '<span class="badge b-rose">❌ 证伪原表第 4 行 & E10 乘积增益</span>',
      title: "2. I_t &lt; 0 极稀少（仅 3%），且 I_t ≤ 0 组表现全面碾压 I_t &gt; 0；乘积项 I_t × r_t 无增量预测力",
      body: "超过 92% 的步子 <code>I_t &gt; 0</code>，真正 <code>I_t &lt; 0</code> 仅占 <b>2.7% (SW) / 3.4% (TW)</b> 且数值微弱（均值仅 -0.015 ~ -0.021），基本是普通步的微小数值波动。看 E10 2×2 分组：<code>I_t ≤ 0</code> 代表 CoT 与直觉意见一致的“稳步”，其动作合法率和成功率远超 <code>I_t &gt; 0</code>。无泄露 5-fold GroupKFold CV 显示加入 <code>I_t × r_t</code> 在 SW 上预测成功 AUROC 反而下降（0.621 → 0.596），TW 控制任务难度后 I_t 回归 p = 0.653 不显著。",
      chips: [
        `I_t &lt; 0 占比: <b>${{sw.I_emp_neg_pct}}% / ${{tw.I_emp_neg_pct}}%</b>`,
        `SW Low_HC &amp; I≤0 合法率: <b>84.2%</b> (vs I&gt;0 <b>40.7%</b>)`,
        `TW I≤0 成功率: <b>57–60%</b> (vs I&gt;0 <b>21–22%</b>)`,
      ]
    }},
    {{
      cls: "violet",
      badge: '<span class="badge b-violet">💡 本次最大新发现（跨 Bench 完全一致）</span>',
      title: "3. “上一步报错后的成功自救 (False → True)”才是 CoT 真正发力的黄金时刻！",
      body: "把上一步刚报错（<code>prev_valid = False</code>）之后的步子拆成“继续犯错 (F→F)”与“成功跳出自救 (F→T)”：两个游戏同时爆出惊人的同向大跳变——继续犯错时模型低熵麻木重复（SW <code>I_t=0.23, H_tail=0.140</code>），而成功自救时 <b>I_t 直接暴涨 7.3 倍（SW: 0.23 → 1.70）和 2.2 倍（TW: 0.87 → 1.92）</b>，同时 CoT 中后段熵明显抬升！说明 CoT 真正的因果价值不在顺风步，而在<b>卡壳报错后扭转动作方向</b>。",
      chips: [
        `SW 自救 I_t: <b>0.23 → 1.70 (7.3×)</b>`,
        `TW 自救 I_t: <b>0.87 → 1.92 (2.2×)</b>`,
        `SW 自救 H_tail: <b>0.140 → 0.197</b>`,
        `TW 自救 H_tail: <b>0.163 → 0.265</b>`,
      ]
    }},
    {{
      cls: "amber",
      badge: '<span class="badge b-amber">⚠️ 修正熵加权粒度（E05/E06/E15/E17）</span>',
      title: "4. Action 熵≈0 验证了“只看 CoT 熵”，但整段均值过粗——CoT 开头与中后段呈跨 Bench X 型交叉",
      body: "CoT 均值熵为 <b>0.153 / 0.163 nats</b>，而 Action 均值熵仅 <b>0.015 / 0.026 nats</b>，证实决策不确定性几乎全在 CoT。但全段均值 <code>H_C_mean</code> 控制任务难度后几乎不显著。按位置切开（Fig09）后两个游戏画出完全一样的 <b>X 型交叉</b>：非法动作在 <b>CoT 开头 (10%–20%)</b> 熵高达合法动作的 <b>2 倍</b>（<code>H_head vs valid: -0.227 / -0.293</code>，主要由报错后慌乱驱动），而中后段熵反而低于合法动作；E15 失败倒计时显示临近失败时熵略降（陷入低熵死循环）。",
      chips: [
        `CoT vs Action 熵: <b>0.16 vs 0.02 nats</b>`,
        `H_head vs valid: <b>-0.227 (SW) / -0.293 (TW)</b>`,
        `E15 失败末尾 H_C_p90: <b>下降（固执化）</b>`,
      ]
    }},
    {{
      cls: "rose",
      badge: '<span class="badge b-rose">❌ 实锤符号反转 + Self-OPD 失忆</span>',
      title: "5. 8B 删历史 CoT 做 Self-Teacher 98% 为负，且实锤原公式 A_C = w·D 会把近 7 成错词“负负得正”重奖",
      body: "同一 Qwen3-8B 去掉历史 CoT 后，失去前面推理上下文，98% 的 Turn 均值 <code>D_self &lt; 0</code>（均值 -1.12 / -1.55），步数越深越负（ρ = -0.35 / -0.28），控制步数和任务后对动作合法性甚至显著负相关。更致命的是，在负向 CoT（<code>A_proxy·I_t &lt; 0</code>）中，<b>68.4% (SW)</b> 与 <b>65.4% (TW)</b> 的 token 同时满足 <code>D_self &lt; 0</code>——乘以负权重将直接变成极大正奖励！",
      chips: [
        `D_self &lt; 0 占比: <b>98.4% (SW) / 97.9% (TW)</b>`,
        `D_self vs turn_id: <b>-0.349 / -0.276</b>`,
        `负负得正冲突比例: <b>${{sw.dbl_neg_pct}}% / ${{tw.dbl_neg_pct}}%</b>`,
      ]
    }},
  ];

  document.getElementById("findings-container").innerHTML = findings.map(f => `
    <div class="finding-card ${{f.cls}}">
      <div class="finding-head">
        <h3 class="finding-title">${{f.title}}</h3>
        ${{f.badge}}
      </div>
      <p class="finding-body">${{f.body}}</p>
      <div class="stat-strip">${{f.chips.map(c => `<span class="stat-chip">${{c}}</span>`).join("")}}</div>
    </div>
  `).join("");

  // Render Recovery Comparison SVG + Table
  renderRecoveryVisual();
}}

function renderRecoveryVisual() {{
  const swRec = REPORT_DATA.sw.recovery;
  const twRec = REPORT_DATA.tw.recovery;
  const swFF = swRec[0], swFT = swRec[1], twFF = twRec[0], twFT = twRec[1];

  const maxI = Math.max(swFF.I_mean, swFT.I_mean, twFF.I_mean, twFT.I_mean, 2.2);
  const maxH = Math.max(swFF.H_tail, swFT.H_tail, twFF.H_tail, twFT.H_tail, 0.30);

  function barChart(title, items, maxV, unit, xOff) {{
    const w = 260, h = 165, padB = 34, padT = 26, chartH = h - padB - padT;
    let s = `<g transform="translate(${{xOff}},0)">`;
    s += `<text x="${{w/2}}" y="15" text-anchor="middle" font-size="12" font-weight="700" fill="#0f172a">${{title}}</text>`;
    s += `<line x1="20" y1="${{h-padB}}" x2="${{w-10}}" y2="${{h-padB}}" stroke="#cbd5e1" stroke-width="1"/>`;
    const bw = 42, gap = 16;
    items.forEach((it, idx) => {{
      const bh = Math.max(2, (it.val / maxV) * chartH);
      const x = 32 + idx * (bw + gap);
      const y = h - padB - bh;
      s += `<rect x="${{x}}" y="${{y}}" width="${{bw}}" height="${{bh}}" rx="5" fill="${{it.color}}" opacity="0.9"/>`;
      s += `<text x="${{x+bw/2}}" y="${{y-5}}" text-anchor="middle" font-size="11" font-weight="700" fill="#0f172a">${{it.val.toFixed(2)}}</text>`;
      s += `<text x="${{x+bw/2}}" y="${{h-padB+14}}" text-anchor="middle" font-size="10.5" fill="#475569">${{it.lbl1}}</text>`;
      s += `<text x="${{x+bw/2}}" y="${{h-padB+26}}" text-anchor="middle" font-size="9.5" fill="#64748b">${{it.lbl2}}</text>`;
    }});
    s += `</g>`;
    return s;
  }}

  const svg = `
    <svg viewBox="0 0 560 175" style="width:100%;height:auto;background:#f8fafc;border-radius:8px;border:1px solid #e2e8f0;">
      ${{barChart("I_empty 均值（报错后继续错 vs 自救）", [
        {{ val: swFF.I_mean, lbl1: "SW F→F", lbl2: `n=${{swFF.n}}`, color: "#94a3b8" }},
        {{ val: swFT.I_mean, lbl1: "SW F→T", lbl2: `n=${{swFT.n}}`, color: "#7c3aed" }},
        {{ val: twFF.I_mean, lbl1: "TW F→F", lbl2: `n=${{twFF.n}}`, color: "#94a3b8" }},
        {{ val: twFT.I_mean, lbl1: "TW F→T", lbl2: `n=${{twFT.n}}`, color: "#059669" }},
      ], maxI, "", 10)}}
      ${{barChart("CoT 后段熵 H_tail (nats)", [
        {{ val: swFF.H_tail, lbl1: "SW F→F", lbl2: "麻木重复", color: "#94a3b8" }},
        {{ val: swFT.H_tail, lbl1: "SW F→T", lbl2: "探索纠错", color: "#7c3aed" }},
        {{ val: twFF.H_tail, lbl1: "TW F→F", lbl2: "麻木重复", color: "#94a3b8" }},
        {{ val: twFT.H_tail, lbl1: "TW F→T", lbl2: "探索纠错", color: "#059669" }},
      ], maxH, "nats", 290)}}
    </svg>
  `;
  document.getElementById("recovery-svg-box").innerHTML = svg;

  const rows = [
    ...swRec.map(r => ({{ bench: "SW", ...r }})),
    ...twRec.map(r => ({{ bench: "TW", ...r }})),
  ];
  document.getElementById("recovery-table-box").innerHTML = `
    <table>
      <thead><tr><th>Bench</th><th>状态转移</th><th class="num">n</th><th class="num">I_mean</th><th class="num">H_head</th><th class="num">H_tail</th><th class="num">G5_strict</th></tr></thead>
      <tbody>
        ${{rows.map(r => `
          <tr style="${{(!r.pv && r.av) ? 'background:#f5f3ff;font-weight:600;' : ''}}">
            <td>${{r.bench}}</td>
            <td>${{r.tag}}</td>
            <td class="num">${{r.n}}</td>
            <td class="num">${{r.I_mean.toFixed(3)}}</td>
            <td class="num">${{r.H_head.toFixed(3)}}</td>
            <td class="num">${{r.H_tail.toFixed(3)}}</td>
            <td class="num">${{r.G5s_mean.toFixed(3)}}</td>
          </tr>
        `).join("")}}
      </tbody>
    </table>
  `;
}}

// ================= Tab 2: Charts & Tables =================
function renderChartsTab() {{
  const cImg = document.getElementById("cross-overview-img");
  cImg.src = REPORT_DATA.cross_fig;
  cImg.onclick = () => openLightbox(REPORT_DATA.cross_fig);

  // Per-task breakdown table
  const allTasks = [...REPORT_DATA.sw.tasks, ...REPORT_DATA.tw.tasks];
  document.getElementById("task-summary-table").innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Bench</th><th>层级 / 任务</th><th class="num">成功率</th><th class="num">均分</th>
          <th class="num">合法率</th><th class="num">I中位</th><th class="num">ρ(I,r)</th><th class="num">ρ(H_head,valid)</th>
        </tr>
      </thead>
      <tbody>
        ${{allTasks.map(t => `
          <tr>
            <td><span class="badge ${{t.benchmark==='ScienceWorld'?'b-sky':'b-emerald'}}">${{t.benchmark==='ScienceWorld'?'SW':'TW'}}</span></td>
            <td><b>${{t.tier}}</b> <code>${{t.task_id}}</code></td>
            <td class="num">${{t.succ_pct}}%</td>
            <td class="num">${{t.score_mean.toFixed(3)}}</td>
            <td class="num">${{t.valid_pct}}%</td>
            <td class="num">${{t.I_med.toFixed(3)}}</td>
            <td class="num ${{t.rho_I_r < 0 ? 'neg' : 'pos'}}">${{t.rho_I_r ?? '-'}}</td>
            <td class="num ${{t.rho_Hhead_valid < 0 ? 'neg' : 'pos'}}">${{t.rho_Hhead_valid ?? '-'}}</td>
          </tr>
        `).join("")}}
      </tbody>
    </table>
  `;

  // E11 Comparison table
  const sw11 = Object.fromEntries(REPORT_DATA.sw.tables.e11_zero_reward.map(r => [r.zero_reward_category, r]));
  const tw11 = Object.fromEntries(REPORT_DATA.tw.tables.e11_zero_reward.map(r => [r.zero_reward_category, r]));
  const cats = [
    ["effective_setup", "有效铺垫 (effective_setup)"],
    ["aimless_explore", "盲目瞎逛 (aimless_explore)"],
    ["illegal_action", "非法动作 (illegal_action)"],
    ["repeat_noop", "原地重复 (repeat_noop)"],
  ];
  document.getElementById("e11-compare-table").innerHTML = `
    <table>
      <thead><tr><th>零奖励分类</th><th class="num">SW n</th><th class="num">SW I均值/中位</th><th class="num">SW H_C</th><th class="num">TW n</th><th class="num">TW I均值/中位</th><th class="num">TW H_C</th></tr></thead>
      <tbody>
        ${{cats.map(([k, lbl]) => {{
          const a = sw11[k] || {{}}, b = tw11[k] || {{}};
          return `<tr>
            <td><b>${{lbl}}</b></td>
            <td class="num">${{a.n_turns ?? 0}}</td>
            <td class="num">${{a.I_empty_mean != null ? a.I_empty_mean.toFixed(2)+' / '+a.I_empty_median.toFixed(3) : '-'}}</td>
            <td class="num">${{a.H_C_mean != null ? a.H_C_mean.toFixed(3) : '-'}}</td>
            <td class="num">${{b.n_turns ?? 0}}</td>
            <td class="num">${{b.I_empty_mean != null ? b.I_empty_mean.toFixed(2)+' / '+b.I_empty_median.toFixed(3) : '-'}}</td>
            <td class="num">${{b.H_C_mean != null ? b.H_C_mean.toFixed(3) : '-'}}</td>
          </tr>`;
        }}).join("")}}
      </tbody>
    </table>
  `;

  // E16 Key terms regression comparison
  const keyTerms = ["I_empty_norm", "H_C_mean", "H_C_p90", "action_valid_int"];
  const sw16 = REPORT_DATA.sw.tables.e16_controlled_regression.filter(r => keyTerms.includes(r.term));
  const tw16 = REPORT_DATA.tw.tables.e16_controlled_regression.filter(r => keyTerms.includes(r.term));
  const map16 = (arr) => Object.fromEntries(arr.map(r => [`${{r.outcome}}|${{r.term}}`, r]));
  const mSW = map16(sw16), mTW = map16(tw16);
  const pairs16 = [
    ["reward_raw", "I_empty_norm"],
    ["G3_strict", "I_empty_norm"],
    ["final_score_norm", "I_empty_norm"],
    ["reward_raw", "H_C_mean"],
    ["G3_strict", "H_C_p90"],
    ["final_score_norm", "H_C_mean"],
    ["reward_raw", "action_valid_int"],
  ];
  function fmtReg(r) {{
    if (!r) return "-";
    const sig = r.p_value < 0.01 ? "**" : (r.p_value < 0.05 ? "*" : "");
    const cls = r.p_value < 0.05 ? (r.coef < 0 ? "neg" : "pos") : "";
    return `<span class="${{cls}}">${{r.coef.toFixed(4)}}${{sig}}</span> <span style="color:#94a3b8;font-size:11px;">(p=${{r.p_value.toFixed(3)}})</span>`;
  }}
  document.getElementById("e16-compare-table").innerHTML = `
    <table>
      <thead><tr><th>因变量 (Outcome)</th><th>自变量 (Term)</th><th class="num">ScienceWorld</th><th class="num">TextWorldExpress</th></tr></thead>
      <tbody>
        ${{pairs16.map(([o, t]) => `
          <tr>
            <td><code>${{o}}</code></td>
            <td><code>${{t}}</code></td>
            <td class="num">${{fmtReg(mSW[`${{o}}|${{t}}`])}}</td>
            <td class="num">${{fmtReg(mTW[`${{o}}|${{t}}`])}}</td>
          </tr>
        `).join("")}}
      </tbody>
    </table>
  `;

  document.querySelectorAll("#chart-mode-btns .btn").forEach(btn => {{
    btn.addEventListener("click", () => {{
      document.querySelectorAll("#chart-mode-btns .btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.chartMode = btn.dataset.mode;
      renderFigGallery();
    }});
  }});
  renderFigGallery();
}}

function renderFigGallery() {{
  const mode = state.chartMode;
  const htmlParts = REPORT_DATA.fig_catalog.map(item => {{
    const swSrc = REPORT_DATA.sw.figures[item.key];
    const twSrc = REPORT_DATA.tw.figures[item.key];
    let body = "";
    if (mode === "both") {{
      body = `
        <div class="grid-2">
          <div>
            <div style="font-weight:700;font-size:12.5px;margin-bottom:6px;color:#0369a1;">ScienceWorld (N_ep=60, N_turn=1924)</div>
            <img class="fig-img" src="${{swSrc}}" onclick="openLightbox(this.src)" loading="lazy" />
            <div class="fig-caption"><b>SW 读图：</b>${{item.sw_note}}</div>
          </div>
          <div>
            <div style="font-weight:700;font-size:12.5px;margin-bottom:6px;color:#047857;">TextWorldExpress (N_ep=60, N_turn=1375)</div>
            <img class="fig-img" src="${{twSrc}}" onclick="openLightbox(this.src)" loading="lazy" />
            <div class="fig-caption"><b>TW 读图：</b>${{item.tw_note}}</div>
          </div>
        </div>
      `;
    }} else {{
      const isSw = mode === "sw";
      body = `
        <img class="fig-img" src="${{isSw ? swSrc : twSrc}}" onclick="openLightbox(this.src)" loading="lazy" />
        <div class="fig-caption"><b>${{isSw ? 'ScienceWorld' : 'TextWorldExpress'}} 读图：</b>${{isSw ? item.sw_note : item.tw_note}}</div>
      `;
    }}
    return `
      <div class="fig-card">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;flex-wrap:wrap;">
          <span class="badge b-slate">${{item.key}} · ${{item.e_tag}}</span>
          <h3 style="margin:0;font-size:15.5px;">${{item.title}}</h3>
        </div>
        ${{body}}
        <div class="fig-takeaway">📌 跨 Bench 结论：${{item.takeaway}}</div>
      </div>
    `;
  }}).join("");
  document.getElementById("fig-gallery").innerHTML = htmlParts;
}}

// ================= Token Heatmap Renderer =================
function tokenBgColor(tok, mode) {{
  if (mode === "entropy") {{
    const h = tok[1] || 0;
    if (h < 0.05) return "transparent";
    const t = Math.min(1, h / 1.5);
    // Warm amber -> rose gradient
    const r = 254;
    const g = Math.round(243 - t * 145);
    const b = Math.round(199 - t * 135);
    const a = 0.25 + t * 0.65;
    return `rgba(${{r}},${{g}},${{b}},${{a.toFixed(2)}})`;
  }} else {{
    const d = tok[3];
    if (d == null) return "transparent";
    if (d >= 0) {{
      const t = Math.min(1, d / 2.5);
      return `rgba(16,185,129,${{(0.12 + t * 0.55).toFixed(2)}})`;
    }} else {{
      const t = Math.min(1, (-d) / 3.5);
      return `rgba(239,68,68,${{(0.10 + t * 0.62).toFixed(2)}})`;
    }}
  }}
}}

function renderTokensHtml(toks, fallbackText, mode) {{
  if (!toks || !toks.length) return esc(fallbackText);
  return toks.map((tk, i) => {{
    const bg = tokenBgColor(tk, mode);
    const ds = tk[3] == null ? "N/A" : tk[3].toFixed(3);
    return `<span class="tok" style="background:${{bg}};" data-t="${{esc(tk[0])}}" data-h="${{tk[1].toFixed(3)}}" data-lp="${{tk[2].toFixed(3)}}" data-ds="${{ds}}">${{esc(tk[0])}}</span>`;
  }}).join("");
}}

const tipEl = document.getElementById("tok-tip");
document.addEventListener("mousemove", e => {{
  const el = e.target.closest(".tok");
  if (!el) {{ tipEl.style.display = "none"; return; }}
  tipEl.innerHTML =
    `Token: <b>${{esc(JSON.stringify(el.dataset.t))}}</b><br/>` +
    `H_C (精确熵): <b>${{el.dataset.h}}</b> nats<br/>` +
    `log π_S(z): <b>${{el.dataset.lp}}</b><br/>` +
    `D_self (T-S): <b>${{el.dataset.ds}}</b>`;
  tipEl.style.display = "block";
  const x = Math.min(window.innerWidth - 270, e.clientX + 14);
  const y = Math.min(window.innerHeight - 110, e.clientY + 14);
  tipEl.style.left = x + "px";
  tipEl.style.top = y + "px";
}});

// ================= Tab 3: Trajectory Explorer =================
function populateTaskFilter() {{
  const b = document.getElementById("f-bench").value;
  const sel = document.getElementById("f-task");
  const cur = sel.value;
  const tasks = [...new Set(ALL_EPISODES.filter(e => b === "all" || e.bench === b).map(e => e.task))];
  sel.innerHTML = `<option value="all">全部任务类别 (${{tasks.length}})</option>` +
    tasks.map(t => `<option value="${{t}}">${{t}}</option>`).join("");
  if (tasks.includes(cur)) sel.value = cur;
}}

function renderEpisodeList() {{
  const b = document.getElementById("f-bench").value;
  const t = document.getElementById("f-task").value;
  const s = document.getElementById("f-succ").value;
  const rec = document.getElementById("f-rec").value;
  const q = document.getElementById("f-search").value.trim().toLowerCase();

  const filtered = ALL_EPISODES.filter(e => {{
    if (b !== "all" && e.bench !== b) return false;
    if (t !== "all" && e.task !== t) return false;
    if (s === "succ" && !e.succ) return false;
    if (s === "fail" && e.succ) return false;
    if (rec === "has_rec" && e.n_rec === 0) return false;
    if (q) {{
      const hitId = e.id.toLowerCase().includes(q);
      const hitTurn = e.turns.some(tr => tr.act.toLowerCase().includes(q) || tr.cot.toLowerCase().includes(q));
      if (!hitId && !hitTurn) return false;
    }}
    return true;
  }});

  document.getElementById("ep-count-badge").textContent = `${{filtered.length}} / ${{ALL_EPISODES.length}}`;
  const listEl = document.getElementById("ep-list");
  listEl.innerHTML = filtered.map(e => `
    <div class="ep-item ${{e.id === state.selectedEpId ? 'active' : ''}}" onclick="selectEpisode('${{e.id}}')">
      <div class="ep-item-title">${{e.bench==='scienceworld'?'[SW]':'[TW]'}} ${{e.task}} #${{e.seed}}</div>
      <div class="ep-item-meta">
        <span class="badge ${{e.succ ? 'b-emerald' : 'b-rose'}}" style="padding:0 6px;font-size:11px;">${{e.succ ? '成功' : '失败'}} (${{e.score}})</span>
        <span>${{e.n_turns}}步</span>
        <span>非法${{e.inv}}</span>
        ${{e.n_rec > 0 ? `<span style="color:var(--violet);font-weight:600;">自救×${{e.n_rec}}</span>` : ''}}
      </div>
    </div>
  `).join("") || `<div style="padding:18px;text-align:center;color:var(--muted);">无匹配轨迹</div>`;
}}

function selectEpisode(epId) {{
  state.selectedEpId = epId;
  state.highlightTurn = null;
  renderEpisodeList();
  renderEpisodeDetail();
}}

function renderEpisodeTimelineSVG(ep) {{
  const ts = ep.turns;
  if (!ts.length) return "";
  const W = 880, H = 210, padL = 48, padR = 48, padT = 20, padB = 34;
  const cw = W - padL - padR, ch = H - padT - padB;
  const maxI = Math.max(2.0, ...ts.map(x => Math.max(0, x.ie ?? 0)));
  const maxH = Math.max(0.4, ...ts.map(x => x.hc ?? 0));
  const xPos = (i) => padL + (ts.length === 1 ? cw / 2 : (i / (ts.length - 1)) * cw);
  const yI = (v) => padT + ch - (Math.max(0, v ?? 0) / maxI) * ch;
  const yH = (v) => padT + ch - ((v ?? 0) / maxH) * ch;

  const pathI = ts.map((tr, i) => `${{i===0?'M':'L'}}${{xPos(i).toFixed(1)}},${{yI(tr.ie).toFixed(1)}}`).join(" ");
  const pathH = ts.map((tr, i) => `${{i===0?'M':'L'}}${{xPos(i).toFixed(1)}},${{yH(tr.hc).toFixed(1)}}`).join(" ");

  let markers = "";
  ts.forEach((tr, i) => {{
    const x = xPos(i), yi = yI(tr.ie), yh = yH(tr.hc);
    const isRec = (tr.pv === false && tr.valid === true);
    const isRew = tr.r > 0;
    const isInv = !tr.valid;
    if (isRec) {{
      markers += `<rect x="${{x-6}}" y="${{padT}}" width="12" height="${{ch}}" fill="rgba(124,58,237,0.13)"/>`;
    }}
    const fill = isRew ? "#059669" : (isRec ? "#7c3aed" : (isInv ? "#e11d48" : "#2563eb"));
    const rad = (isRew || isRec) ? 5.2 : 3.6;
    markers += `<circle cx="${{x.toFixed(1)}}" cy="${{yi.toFixed(1)}}" r="${{rad}}" fill="${{fill}}" stroke="#fff" stroke-width="1.2" style="cursor:pointer" onclick="scrollToTurn(${{tr.t}})"><title>Turn ${{tr.t}}: I_empty=${{tr.ie}}, H_C=${{tr.hc}}, r=${{tr.r}}, valid=${{tr.valid}}</title></circle>`;
    markers += `<circle cx="${{x.toFixed(1)}}" cy="${{yh.toFixed(1)}}" r="2.6" fill="#d97706" style="cursor:pointer" onclick="scrollToTurn(${{tr.t}})"><title>Turn ${{tr.t}}: H_C_mean=${{tr.hc}} nats</title></circle>`;
    if (i % Math.max(1, Math.ceil(ts.length / 18)) === 0 || i === ts.length - 1) {{
      markers += `<text x="${{x.toFixed(1)}}" y="${{H-10}}" text-anchor="middle" font-size="10" fill="#64748b">T${{tr.t}}</text>`;
    }}
  }});

  return `
    <svg viewBox="0 0 ${{W}} ${{H}}" style="width:100%;height:auto;background:#f8fafc;border-radius:8px;border:1px solid #e2e8f0;">
      <line x1="${{padL}}" y1="${{padT+ch}}" x2="${{W-padR}}" y2="${{padT+ch}}" stroke="#cbd5e1"/>
      <line x1="${{padL}}" y1="${{padT}}" x2="${{padL}}" y2="${{padT+ch}}" stroke="#93c5fd"/>
      <line x1="${{W-padR}}" y1="${{padT}}" x2="${{W-padR}}" y2="${{padT+ch}}" stroke="#fcd34d"/>
      <text x="${{padL-6}}" y="${{padT+8}}" text-anchor="end" font-size="10" fill="#2563eb">${{maxI.toFixed(1)}}</text>
      <text x="${{padL-6}}" y="${{padT+ch}}" text-anchor="end" font-size="10" fill="#2563eb">0</text>
      <text x="${{padL}}" y="12" font-size="10.5" font-weight="700" fill="#2563eb">← 左轴：I_empty (蓝线)</text>
      <text x="${{W-padR+6}}" y="${{padT+8}}" font-size="10" fill="#d97706">${{maxH.toFixed(2)}}</text>
      <text x="${{W-padR+6}}" y="${{padT+ch}}" font-size="10" fill="#d97706">0</text>
      <text x="${{W-padR}}" y="12" text-anchor="end" font-size="10.5" font-weight="700" fill="#d97706">右轴：CoT 熵 H_C (橙虚线) →</text>
      <path d="${{pathI}}" fill="none" stroke="#2563eb" stroke-width="2"/>
      <path d="${{pathH}}" fill="none" stroke="#d97706" stroke-width="1.8" stroke-dasharray="4 3"/>
      ${{markers}}
    </svg>
    <div class="legend-bar" style="margin-top:6px;">
      <span>图上圆点含义：</span>
      <span><span class="swatch" style="background:#059669;border-radius:50%;"></span> 拿分步 (r&gt;0)</span>
      <span><span class="swatch" style="background:#7c3aed;border-radius:50%;"></span> 报错后自救 (F→T，紫底柱)</span>
      <span><span class="swatch" style="background:#e11d48;border-radius:50%;"></span> 非法动作</span>
      <span><span class="swatch" style="background:#2563eb;border-radius:50%;"></span> 正常 0 分步</span>
    </div>
  `;
}}

function scrollToTurn(t) {{
  state.highlightTurn = t;
  document.querySelectorAll(".turn-card").forEach(el => el.classList.remove("highlight-turn"));
  const el = document.getElementById(`turn-card-${{t}}`);
  if (el) {{
    el.classList.add("highlight-turn");
    el.scrollIntoView({{ behavior: "smooth", block: "center" }});
  }}
}}

function renderTurnCard(tr, mode, idPrefix = "turn-card") {{
  const isRec = (tr.pv === false && tr.valid === true);
  const isRew = tr.r > 0;
  const isInv = !tr.valid;
  const cls = [
    "turn-card",
    isRec ? "recovery-turn" : (isRew ? "reward-turn" : (isInv ? "invalid-turn" : "")),
    (idPrefix === "turn-card" && state.highlightTurn === tr.t) ? "highlight-turn" : "",
  ].join(" ");

  const badges = [];
  if (isRew) badges.push(`<span class="badge b-emerald">🎯 r = +${{tr.r}}</span>`);
  else if (tr.r < 0) badges.push(`<span class="badge b-rose">r = ${{tr.r}}</span>`);
  else badges.push(`<span class="badge b-slate">r = 0 (${{tr.cat}})</span>`);
  if (isRec) badges.push(`<span class="badge b-violet">💡 报错自救 F→T</span>`);
  else if (tr.pv === false && !tr.valid) badges.push(`<span class="badge b-rose">连续报错 F→F</span>`);
  badges.push(tr.valid ? `<span class="badge b-emerald">合法动作</span>` : `<span class="badge b-rose">非法动作</span>`);

  return `
    <div id="${{idPrefix}}-${{tr.t}}" class="${{cls}}">
      <div class="turn-head">
        <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
          <strong style="font-size:14.5px;">Turn ${{tr.t}}</strong>
          ${{badges.join("")}}
        </div>
        <span style="font-size:12px;color:var(--muted);">累计归一化得分：<b>${{tr.sc}}</b> · 未来3步 G3_strict：<b>${{tr.g3}}</b></span>
      </div>

      <details class="obs-box" ${{tr.t === 0 || isRec || isInv ? '' : ''}}>
        <summary>Observation（点击展开/收起环境画面）</summary>
        <div style="margin-top:6px;">${{esc(tr.obs)}}</div>
      </details>

      <div class="cot-box">
        <span style="font-size:11.5px;font-weight:700;color:var(--muted);margin-right:6px;">Thought (CoT):</span>
        ${{renderTokensHtml(tr.toks, tr.cot, mode)}}
      </div>

      <div class="action-bar">
        <div>
          <span style="font-size:12px;color:var(--muted);margin-right:6px;">Action:</span>
          <span class="action-cmd">${{esc(tr.act)}}</span>
        </div>
        <span style="font-size:12px;color:var(--muted);">
          <code>log π(a|h,z)=${{tr.lpc}}</code> vs <code>log π(a|h)=${{tr.lpe}}</code>
        </span>
      </div>

      <div class="metric-pills">
        <span class="mpill"><b>I_empty</b>: ${{tr.ie}}</span>
        <span class="mpill"><b>I_neutral</b>: ${{tr.in}}</span>
        <span class="mpill"><b>H_C_mean</b>: ${{tr.hc}}</span>
        <span class="mpill"><b>H_head/tail</b>: ${{tr.hh}} / ${{tr.ht}}</span>
        <span class="mpill"><b>H_A</b>: ${{tr.ha}}</span>
        <span class="mpill"><b>D_self</b>: ${{tr.ds}} (负词${{tr.dnf != null ? Math.round(tr.dnf*100)+'%' : '-'}})</span>
      </div>
    </div>
  `;
}}

function renderTokenLegend(mode) {{
  if (mode === "entropy") {{
    return `<span>CoT 词底色 = <b>精确熵 H_C (nats)</b>（悬停看精确数值）：</span>
      <span><span class="swatch" style="background:transparent;"></span> &lt;0.05 (很自信)</span>
      <span><span class="swatch" style="background:rgba(254,200,150,0.45);"></span> ~0.5 (轻度犹豫)</span>
      <span><span class="swatch" style="background:rgba(254,110,75,0.85);"></span> ≥1.5 (高熵分支词)</span>`;
  }} else {{
    return `<span>CoT 词底色 = <b>Self-Teacher Gap D_self = log π_T - log π_S</b>：</span>
      <span><span class="swatch" style="background:rgba(16,185,129,0.55);"></span> D &gt; 0 (Teacher 更认可)</span>
      <span><span class="swatch" style="background:rgba(239,68,68,0.28);"></span> D &lt; 0 (轻度分歧)</span>
      <span><span class="swatch" style="background:rgba(239,68,68,0.70);"></span> D ≪ 0 (删历史 CoT 后严重陌生)</span>`;
  }}
}}

function renderEpisodeDetail() {{
  const ep = EP_MAP[state.selectedEpId];
  if (!ep) return;
  document.getElementById("ep-header-card").innerHTML = `
    <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;">
      <div>
        <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
          <span class="badge ${{ep.bench==='scienceworld'?'b-sky':'b-emerald'}}">${{ep.bench_label}}</span>
          <span class="badge b-slate">${{ep.tier}}</span>
          <span class="badge ${{ep.succ ? 'b-emerald' : 'b-rose'}}">${{ep.succ ? '✅ 通关成功' : '❌ 未通关 ('+ep.term+')'}}</span>
          ${{ep.n_rec > 0 ? `<span class="badge b-violet">💡 含 ${{ep.n_rec}} 次报错自救</span>` : ''}}
        </div>
        <h2 style="margin:8px 0 0;font-size:18px;"><code>${{ep.id}}</code></h2>
      </div>
      <div style="display:flex;gap:14px;font-size:13px;color:var(--muted);">
        <div>终局分：<b style="color:var(--text);font-size:16px;">${{ep.score}}</b></div>
        <div>总步数：<b style="color:var(--text);font-size:16px;">${{ep.n_turns}}</b></div>
        <div>非法步：<b style="color:var(--rose);font-size:16px;">${{ep.inv}}</b></div>
      </div>
    </div>
  `;
  document.getElementById("ep-timeline-svg").innerHTML = renderEpisodeTimelineSVG(ep);
  document.getElementById("tok-legend").innerHTML = renderTokenLegend(state.tokColorMode);
  document.getElementById("ep-turns-list").innerHTML = ep.turns.map(tr => renderTurnCard(tr, state.tokColorMode)).join("");
}}

// ================= Tab 4: Curated Cases =================
function renderCasesTab() {{
  const box = document.getElementById("cases-container");
  box.innerHTML = REPORT_DATA.curated_cases.map((c, idx) => {{
    const ep = EP_MAP[c.episode_id];
    if (!ep) return "";
    const mode = c.category === "sign_conflict" ? "dself" : "entropy";
    const turnsHtml = c.turns.map(tNum => {{
      const tr = ep.turns.find(x => x.t === tNum);
      return tr ? renderTurnCard(tr, mode, `case-${{idx}}`) : "";
    }}).join("");
    return `
      <div class="card" style="margin-bottom:18px;border-top:3px solid var(--${{c.color === 'orange' ? 'amber' : c.color}});">
        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;margin-bottom:10px;">
          <div>
            <span class="badge b-${{c.color}}">${{c.badge}}</span>
            <h3 style="margin:6px 0 0;font-size:16px;">${{c.title}}</h3>
          </div>
          <button class="btn primary" onclick="jumpToEpisode('${{c.episode_id}}', ${{c.turns[0]}})">在轨迹浏览器中查看完整该局 →</button>
        </div>
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 14px;margin-bottom:12px;font-size:13.5px;color:#334155;">
          <b>🔍 深度解读：</b>${{c.analysis}}
        </div>
        <div class="legend-bar" style="margin-bottom:8px;">${{renderTokenLegend(mode)}}</div>
        ${{turnsHtml}}
      </div>
    `;
  }}).join("");
}}

// ================= Init Listeners =================
["f-bench", "f-task", "f-succ", "f-rec"].forEach(id => {{
  document.getElementById(id).addEventListener("change", () => {{
    if (id === "f-bench") populateTaskFilter();
    renderEpisodeList();
  }});
}});
document.getElementById("f-search").addEventListener("input", renderEpisodeList);

document.querySelectorAll("#tok-color-mode .btn").forEach(btn => {{
  btn.addEventListener("click", () => {{
    document.querySelectorAll("#tok-color-mode .btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.tokColorMode = btn.dataset.cmode;
    renderEpisodeDetail();
  }});
}});

// Boot
renderOverview();
renderChartsTab();
populateTaskFilter();
renderEpisodeList();
renderEpisodeDetail();
renderCasesTab();
</script>
</body>
</html>
"""
    OUT_HTML.write_text(html_doc, encoding="utf-8")
    size_mb = OUT_HTML.stat().st_size / (1024 * 1024)
    print(f"[HTML Builder Complete] Wrote self-contained interactive report: {OUT_HTML} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()

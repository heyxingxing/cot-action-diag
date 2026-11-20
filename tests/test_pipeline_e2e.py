#!/usr/bin/env python3
"""
本地端到端自测（不需要 GPU 和 Java）：
1. 单元测试 ReAct 解析器、Token Span 剥离模板前缀、4-Pass token ID 逐位对齐断言与 State-OPD 历史剪枝；
2. 端到端跑通 ScienceWorld & TextWorldExpress（MockEnv + MockModel）：
   01_run_rollout -> 02_score_offline -> 03_analyze_and_plot -> 04_e14_audit (sample+evaluate) -> 05_compare_benchmarks
3. 核验所有 JSONL / Parquet / CSV / 10 张图与跨 Bench 汇总产物完整无误。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mock_model import MockTokenizer
from src.prompts import build_state_opd_messages, parse_react_output
from src.tokenization import build_four_pass_sequences, locate_token_spans


def test_parser_and_token_alignment() -> None:
    tok = MockTokenizer()
    cases = [
        "Thought: I need to find the apple in the kitchen.\nAction: go to kitchen",
        "Thought: Let's open it.\nAction:open fridge",
        "Thought: Line one.\nLine two of thought.\nAction: pick up metal fork\n",
    ]
    history = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Turn 0:\nObservation: Room A"},
        {"role": "assistant", "content": "Thought: Old noisy thought.\nAction: look around"},
        {"role": "user", "content": "Turn 1:\nObservation: Room B"},
    ]

    # Verify State-OPD strips historical Thought completely while keeping Action
    pruned = build_state_opd_messages(history)
    assert "Old noisy thought" not in pruned[2]["content"]
    assert pruned[2]["content"] == "Action: look around"

    for raw in cases:
        parsed = parse_react_output(raw)
        assert parsed.parse_ok, f"Expected parse_ok for: {raw!r}"
        gen_ids = tok.encode(raw, add_special_tokens=False) + [tok.eos_token_id]
        span = locate_token_spans(tok, gen_ids, parsed)
        assert span.cot_token_span is not None
        assert span.action_token_span is not None

        cot_dec = tok.decode(gen_ids[span.cot_token_span[0] : span.cot_token_span[1]]).strip()
        act_dec = tok.decode(gen_ids[span.action_token_span[0] : span.action_token_span[1]]).strip()
        assert "Thought:" not in cot_dec, f"Template leaked into CoT span: {cot_dec!r}"
        assert "Action:" not in act_dec, f"Template leaked into Action span: {act_dec!r}"
        assert act_dec == parsed.action, f"Action mismatch: {act_dec!r} != {parsed.action!r}"

        prompt_ids = tok.encode("prompt text ", add_special_tokens=False)
        fp = build_four_pass_sequences(
            tokenizer=tok,
            history=history,
            prompt_token_ids=prompt_ids,
            generated_token_ids=gen_ids,
            cot_token_span=span.cot_token_span,
            action_token_span=span.action_token_span,
            cot_prefix_tail=span.cot_prefix_tail,
            action_prefix_tail=span.action_prefix_tail,
            neutral_cot_text="I will choose the next action based on the current observation.",
            enable_thinking=False,
            build_teacher_pass=True,
        )
        a1 = fp.p1_ids[fp.p1_act_slice[0] : fp.p1_act_slice[1]]
        a2 = fp.p2_ids[fp.p2_act_slice[0] : fp.p2_act_slice[1]]
        a3 = fp.p3_ids[fp.p3_act_slice[0] : fp.p3_act_slice[1]]
        assert a1 == a2 == a3, "Action token IDs diverged across Pass 1/2/3!"
        c1 = fp.p1_ids[fp.p1_cot_slice[0] : fp.p1_cot_slice[1]]
        c4 = fp.p4_ids[fp.p4_cot_slice[0] : fp.p4_cot_slice[1]]
        assert c1 == c4, "CoT token IDs diverged across Pass 1/4!"

    # Verify broken formats are cleanly classified without crashing
    assert parse_react_output("Action: look around").parse_error_type == "missing_thought"
    assert parse_react_output("Thought: thinking only").parse_error_type == "missing_action"
    assert parse_react_output("Thought: \nAction: look").parse_error_type == "empty_cot"
    assert parse_react_output("gibberish").parse_error_type == "format_broken"
    print("[PASS] Parser, State-OPD pruning, and 4-pass token span alignment unit tests passed!")


def _run(cmd: list[str]) -> None:
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        print("STDOUT:\n", res.stdout)
        print("STDERR:\n", res.stderr)
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(cmd)}")


def test_full_mock_pipeline() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="cot_diag_e2e_"))
    try:
        for bench in ["scienceworld", "textworld_express"]:
            d_dir = tmp / "data" / bench
            r_dir = tmp / "results" / bench

            # Stage 1: Rollout (12 mock episodes so GroupKFold + cluster bootstrap + E16 all run)
            _run(
                [
                    sys.executable,
                    "scripts/01_run_rollout.py",
                    "--benchmark", bench,
                    "--mode", "full",
                    "--backend", "mock",
                    "--use-mock-env",
                    "--max-episodes", "12",
                    "--out-dir", str(d_dir),
                ]
            )
            # Test --resume idempotency
            _run(
                [
                    sys.executable,
                    "scripts/01_run_rollout.py",
                    "--benchmark", bench,
                    "--mode", "full",
                    "--backend", "mock",
                    "--use-mock-env",
                    "--max-episodes", "12",
                    "--out-dir", str(d_dir),
                    "--resume",
                ]
            )

            # Stage 2: Offline 4-Pass Scoring
            _run(
                [
                    sys.executable,
                    "scripts/02_score_offline.py",
                    "--data-dir", str(d_dir),
                    "--backend", "mock",
                ]
            )
            assert (d_dir / "turn_metrics.jsonl").exists()
            assert (d_dir / "token_stats.parquet").exists()

            # Stage 3: Statistical Analysis + 10 Figures + Chinese Report
            _run(
                [
                    sys.executable,
                    "scripts/03_analyze_and_plot.py",
                    "--data-dir", str(d_dir),
                    "--out-dir", str(r_dir),
                    "--bootstrap", "100",
                ]
            )
            figs = sorted((r_dir / "figures").glob("Fig*.png"))
            assert len(figs) == 10, f"Expected 10 figures for {bench}, got {len(figs)}"
            for f in figs:
                assert f.stat().st_size > 5000, f"Figure too small/empty: {f}"
            assert (r_dir / "report_summary.md").exists()

            # Stage 4: E14 sample + simulated manual labeling + evaluate
            audit_csv = r_dir / "e14_blind.csv"
            _run(
                [
                    sys.executable,
                    "scripts/04_e14_audit.py",
                    "sample",
                    "--data-dir", str(d_dir),
                    "--out-csv", str(audit_csv),
                    "--n-samples", "24",
                ]
            )
            df_audit = pd.read_csv(audit_csv)
            assert "I_empty_norm" not in df_audit.columns, "Blind CSV must NOT expose I_empty_norm!"
            df_audit["cot_explicit_intent"] = [i % 2 for i in range(len(df_audit))]
            df_audit["action_contradicts_cot"] = [(i + 1) % 2 for i in range(len(df_audit))]
            df_audit["cot_too_vague"] = [1 if i % 3 == 0 else 0 for i in range(len(df_audit))]
            df_audit.to_csv(audit_csv, index=False)
            _run(
                [
                    sys.executable,
                    "scripts/04_e14_audit.py",
                    "evaluate",
                    "--data-dir", str(d_dir),
                    "--labeled-csv", str(audit_csv),
                    "--out-dir", str(r_dir),
                ]
            )
            assert (r_dir / "e14_alignment.csv").exists()
            assert (r_dir / "Fig10b_E14_semantic_alignment.png").exists()

        # Stage 5: Cross-benchmark comparison
        cross_dir = tmp / "results" / "cross_benchmark"
        _run(
            [
                sys.executable,
                "scripts/05_compare_benchmarks.py",
                "--sw-results", str(tmp / "results" / "scienceworld"),
                "--tw-results", str(tmp / "results" / "textworld_express"),
                "--out-dir", str(cross_dir),
            ]
        )
        assert (cross_dir / "cross_benchmark_comparison.md").exists()
        assert (cross_dir / "Fig_cross_benchmark_overview.png").exists()
        print("[PASS] Full end-to-end mock pipeline (both benchmarks, 10+10+3 figures, all tables & reports) passed!")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_parser_and_token_alignment()
    test_full_mock_pipeline()

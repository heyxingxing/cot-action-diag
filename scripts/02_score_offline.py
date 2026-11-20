#!/usr/bin/env python3
"""
Stage 2: 离线 4-Pass Teacher-Forcing 打分脚本 (一次性覆盖 E03/E05/E06/E08/E11/E13/E17/E18)

核心优化与防坑设计：
1. 使用 HF Transformers (temperature=1.0, 无 top-p/top-k 截断) 算真实全词表分布；
2. 先对 logits 做 slice，只把目标 CoT / Action token 对应的行转 float32 算精确香农熵（nats），
   峰值显存降 60 倍，80G A800 轻松 batch_size=4~8；
3. 绝不落盘全词表 logits，只存逐 token 标量 (token_stats.parquet, 仅 5-15 MB)；
4. 强断言校验 Pass 1/2/3 的 Action token IDs 逐位完全相同，Pass 1/4 的 CoT token IDs 逐位完全相同；
5. 自动计算 E08 (G3/G5 含当前与严格未来两套)、E10 去中心化代理优势、E11 零奖励 5 分类与 E17 分段熵。
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_protocol
from src.io_utils import load_jsonl, write_jsonl
from src.tokenization import FourPassInputs, build_four_pass_sequences


class HFOfflineScorer:
    """Memory-efficient batched HF teacher-forcing scorer computing exact full-vocab entropy."""

    def __init__(self, model_id: str, revision: Optional[str], cfg: Dict[str, Any]) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        ocfg = cfg["offline_scoring"]
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.pad_id = int(self.tokenizer.pad_token_id)

        dtype = torch.bfloat16 if ocfg["dtype"] == "bfloat16" else torch.float16
        attn_impl = ocfg.get("attn_implementation", "flash_attention_2")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                dtype=dtype,
                attn_implementation=attn_impl,
                device_map="auto",
            ).eval()
        except Exception as e:
            print(f"[Info] attn_implementation='{attn_impl}' unavailable ({e}); falling back to 'sdpa'.")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                dtype=dtype,
                attn_implementation="sdpa",
                device_map="auto",
            ).eval()

    def score_batch(
        self,
        items: List[Tuple[List[int], List[Tuple[int, int]]]],
    ) -> List[List[Tuple[List[float], List[float]]]]:
        """
        items: list of (input_ids, list_of_target_slices [(start, end), ...])
        Right-pads sequences so token indices remain identical!
        Computes float32 log_softmax & exact entropy ONLY on the requested target slices.
        Returns for each item: list of (token_logps, token_entropies_nats) per target slice.
        """
        import torch

        if not items:
            return []

        max_len = max(len(seq) for seq, _ in items)
        bsz = len(items)
        device = next(self.model.parameters()).device

        input_tensor = torch.full((bsz, max_len), self.pad_id, dtype=torch.long, device=device)
        attn_mask = torch.zeros((bsz, max_len), dtype=torch.long, device=device)
        for b, (seq, _) in enumerate(items):
            L = len(seq)
            input_tensor[b, :L] = torch.tensor(seq, dtype=torch.long, device=device)
            attn_mask[b, :L] = 1

        batch_out: List[List[Tuple[List[float], List[float]]]] = []
        with torch.no_grad():
            # Run only the decoder and apply lm_head on the target rows: full [B, L, V] logits
            # would cost ~15 GB at bs=4 x 12K tokens for rows we never read.
            decoder = self.model.get_decoder()
            lm_head = self.model.get_output_embeddings()
            hidden = decoder(input_ids=input_tensor, attention_mask=attn_mask, use_cache=False).last_hidden_state

            for b, (seq, slices) in enumerate(items):
                item_res: List[Tuple[List[float], List[float]]] = []
                for start, end in slices:
                    if start <= 0 or end <= start or end > len(seq):
                        item_res.append(([], []))
                        continue
                    # Causal LM shift: hidden[pos - 1] predicts input_ids[pos]
                    item_res.append(
                        self._token_stats(lm_head(hidden[b, start - 1 : end - 1, :]), input_tensor[b, start:end])
                    )
                batch_out.append(item_res)
            del hidden
        return batch_out

    @staticmethod
    def _token_stats(logits_rows: Any, target_ids: Any) -> Tuple[List[float], List[float]]:
        """fp32 log-softmax over the full vocab -> (target-token logprobs, exact entropies in nats)."""
        import torch

        log_probs = torch.log_softmax(logits_rows.float(), dim=-1)
        ent = -(log_probs.exp() * log_probs).sum(dim=-1)
        tok_lp = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        return (
            [round(float(x), 6) for x in tok_lp.cpu().tolist()],
            [round(float(max(0.0, x)), 6) for x in ent.cpu().tolist()],
        )

    def score_shared_prefix(
        self,
        prefix_ids: List[int],
        suffixes: List[Tuple[List[int], List[Tuple[int, int]]]],
    ) -> List[List[Tuple[List[float], List[float]]]]:
        """
        Score several continuations of ONE prompt: the prefix is encoded once into a KV cache and
        each suffix runs on top of it, with the cache cropped back to the prefix in between. This
        removes ~2/3 of the compute of Passes 1-3, which differ only after the shared prompt.
        Slices are absolute positions in prefix+suffix and must lie inside the suffix.
        """
        import torch

        device = next(self.model.parameters()).device
        decoder = self.model.get_decoder()
        lm_head = self.model.get_output_embeddings()
        p_len = len(prefix_ids)
        out: List[List[Tuple[List[float], List[float]]]] = []
        with torch.no_grad():
            pre = decoder(input_ids=torch.tensor([prefix_ids], device=device), use_cache=True)
            cache = pre.past_key_values
            last_prefix_h = pre.last_hidden_state[:, -1:, :]
            del pre
            for suffix_ids, slices in suffixes:
                suf = torch.tensor([suffix_ids], device=device)
                h_suf = decoder(input_ids=suf, past_key_values=cache, use_cache=True).last_hidden_state
                cache.crop(p_len)
                # Row k of h predicts the token at absolute position p_len + k
                h = torch.cat([last_prefix_h, h_suf], dim=1)[0]
                item_res: List[Tuple[List[float], List[float]]] = []
                for start, end in slices:
                    if start < p_len or end <= start or end > p_len + len(suffix_ids):
                        item_res.append(([], []))
                        continue
                    item_res.append(self._token_stats(lm_head(h[start - p_len : end - p_len]), suf[0, start - p_len : end - p_len]))
                out.append(item_res)
            del cache
        return out

    def close(self) -> None:
        import torch

        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class MockOfflineScorer:
    """Fast CPU mock scorer for local pipeline verification."""

    def __init__(self) -> None:
        from src.mock_model import MockTokenizer

        self.tokenizer = MockTokenizer()

    def score_batch(
        self,
        items: List[Tuple[List[int], List[Tuple[int, int]]]],
        pass_tag: str = "p1",
    ) -> List[List[Tuple[List[float], List[float]]]]:
        from src.mock_model import mock_score_sequence

        out: List[List[Tuple[List[float], List[float]]]] = []
        for seq, slices in items:
            res = [mock_score_sequence(seq, sl, pass_tag=pass_tag) for sl in slices]
            out.append(res)
        return out

    def close(self) -> None:
        return None


def _run_bucketed_pass(
    scorer: Any,
    requests: List[Tuple[int, List[int], List[Tuple[int, int]]]],
    batch_size: int,
    desc: str,
    is_mock: bool = False,
    pass_tag: str = "p1",
) -> Dict[int, List[Tuple[List[float], List[float]]]]:
    """Sort requests by sequence length to minimize padding waste, run batched forward passes."""
    ordered = sorted(requests, key=lambda x: len(x[1]))
    results: Dict[int, List[Tuple[List[float], List[float]]]] = {}
    for i in tqdm(range(0, len(ordered), batch_size), desc=desc):
        chunk = ordered[i : i + batch_size]
        payload = [(seq, slices) for _, seq, slices in chunk]
        if is_mock:
            scored = scorer.score_batch(payload, pass_tag=pass_tag)
        else:
            scored = scorer.score_batch(payload)
        for (turn_idx, _, _), res in zip(chunk, scored):
            results[turn_idx] = res
    return results


def _run_shared_prefix_passes(
    scorer: Any,
    turns: List[Dict[str, Any]],
    p1_reqs: List[Tuple[int, List[int], List[Tuple[int, int]]]],
    p2_reqs: List[Tuple[int, List[int], List[Tuple[int, int]]]],
    p3_reqs: List[Tuple[int, List[int], List[Tuple[int, int]]]],
) -> Tuple[Dict[int, Any], Dict[int, Any], Dict[int, Any]]:
    """Passes 1-3 of a turn all start with the same prompt_token_ids, so encode it once per turn."""
    per_turn: Dict[int, List[Tuple[str, List[int], List[Tuple[int, int]]]]] = {}
    for tag, reqs in (("p1", p1_reqs), ("p2", p2_reqs), ("p3", p3_reqs)):
        for idx, seq, slices in reqs:
            per_turn.setdefault(idx, []).append((tag, seq, slices))

    results: Dict[str, Dict[int, Any]] = {"p1": {}, "p2": {}, "p3": {}}
    for idx in tqdm(sorted(per_turn), desc="Passes 1-3 (shared prompt KV: H_C, H_A, I_empty, I_neutral)"):
        prefix = list(turns[idx]["prompt_token_ids"])
        p_len = len(prefix)
        items = per_turn[idx]
        for _, seq, _ in items:
            assert seq[:p_len] == prefix, "Pass 1-3 sequences must start with the turn's prompt_token_ids"
        scored = scorer.score_shared_prefix(prefix, [(seq[p_len:], slices) for _, seq, slices in items])
        for (tag, _, _), res in zip(items, scored):
            results[tag][idx] = res
    return results["p1"], results["p2"], results["p3"]


def _safe_mean(arr: List[float]) -> Optional[float]:
    return float(np.mean(arr)) if arr else None


def _safe_quantile(arr: List[float], q: float) -> Optional[float]:
    return float(np.quantile(arr, q)) if arr else None


def _safe_std(arr: List[float]) -> Optional[float]:
    return float(np.std(arr)) if arr else None


def _safe_corr(a: List[float], b: List[float]) -> Optional[float]:
    if len(a) < 3 or len(a) != len(b):
        return None
    sa, sb = np.std(a), np.std(b)
    if sa < 1e-12 or sb < 1e-12:
        return None
    val = float(np.corrcoef(a, b)[0, 1])
    return None if math.isnan(val) else val


def _segment_entropies(ent: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Compute E17 head/mid/tail mean entropy and relative position of max-entropy token."""
    n = len(ent)
    if n == 0:
        return None, None, None, None
    c1 = max(1, n // 3)
    c2 = max(c1 + 1, (2 * n) // 3) if n >= 3 else n
    head = float(np.mean(ent[:c1]))
    mid = float(np.mean(ent[c1:c2])) if c2 > c1 else head
    tail = float(np.mean(ent[c2:])) if n > c2 else mid
    peak_idx = int(np.argmax(ent))
    peak_rel = float(peak_idx / max(1, n - 1))
    return head, mid, tail, peak_rel


def verify_span_alignment(tokenizer: Any, turns: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
    """
    Decode every stored CoT/Action token span and require it to equal the parsed text exactly,
    so a span that leaks "Thought:"/"Action:", newlines or EOS is caught. Misaligned spans are
    set to None in place (excluded from scoring) instead of silently producing wrong metrics.
    """
    n_checked = 0
    mismatches: List[str] = []
    for t in turns:
        gen = t["generated_token_ids"]
        for span_key, text_key in (("cot_token_span", "cot"), ("action_token_span", "action")):
            span = t.get(span_key)
            if not span:
                continue
            n_checked += 1
            decoded = tokenizer.decode(gen[span[0] : span[1]], skip_special_tokens=False).strip()
            expected = (t.get(text_key) or "").strip()
            if decoded != expected:
                mismatches.append(
                    f"{t['episode_id']} t={t['turn_id']} {text_key}: decoded={decoded!r} parsed={expected!r}"
                )
                t[span_key] = None
    return n_checked, mismatches


def compute_episode_level_targets_and_e11(
    turns: List[Dict[str, Any]],
    episodes_by_id: Dict[str, Dict[str, Any]],
    metrics_rows: List[Dict[str, Any]],
) -> None:
    """
    In-place enrichment of metrics_rows with:
      - E08: G3_incl, G5_incl, G3_strict, G5_strict, final_score_norm, terminal_success
      - E10: task-centered A_turn_proxy & A_ep_proxy (zero-mean within each task_id)
      - E11: mutually exclusive 5-way zero_reward_category
    """
    ep_to_indices: Dict[str, List[int]] = {}
    for idx, t in enumerate(turns):
        ep_to_indices.setdefault(t["episode_id"], []).append(idx)

    # 1. E08 future returns + E11 zero-reward categorization per episode
    for ep_id, idxs in ep_to_indices.items():
        idxs.sort(key=lambda k: turns[k]["turn_id"])
        ep_meta = episodes_by_id.get(ep_id, {})
        ep_succ = bool(ep_meta.get("terminal_success", False))
        ep_final_norm = float(ep_meta.get("final_score_norm", 0.0))
        rewards = [float(turns[k]["reward_raw"]) for k in idxs]
        T = len(idxs)
        seen_obs_action = set()

        for pos, k in enumerate(idxs):
            t = turns[k]
            m = metrics_rows[k]

            g3_incl = float(sum(rewards[pos : min(T, pos + 3)]))
            g5_incl = float(sum(rewards[pos : min(T, pos + 5)]))
            g3_strict = float(sum(rewards[pos + 1 : min(T, pos + 4)]))
            g5_strict = float(sum(rewards[pos + 1 : min(T, pos + 6)]))

            m["G3_incl"] = round(g3_incl, 6)
            m["G5_incl"] = round(g5_incl, 6)
            m["G3_strict"] = round(g3_strict, 6)
            m["G5_strict"] = round(g5_strict, 6)
            m["final_score_norm"] = round(ep_final_norm, 6)
            m["terminal_success"] = ep_succ

            # E11 5-way mutually exclusive categorization for reward_raw == 0
            r_t = float(t["reward_raw"])
            obs_act_key = (t["observation"].strip(), (t["action"] or "").strip().lower())
            if r_t != 0.0:
                cat = "nonzero_reward"
            elif not t["parse_ok"]:
                cat = "parse_fail"
            elif not t["action_valid"]:
                cat = "illegal_action"
            elif (t["next_observation"].strip() == t["observation"].strip()) or (obs_act_key in seen_obs_action):
                cat = "repeat_noop"
            elif (g5_strict > 0.0) or ep_succ:
                cat = "effective_setup"
            else:
                cat = "aimless_explore"
            seen_obs_action.add(obs_act_key)
            m["zero_reward_category"] = cat

    # 2. Task-centered proxy advantages for E10 quadrant analysis (has both positive & negative values)
    by_task: Dict[str, List[int]] = {}
    for idx, t in enumerate(turns):
        by_task.setdefault(t["task_id"], []).append(idx)

    for task_id, idxs in by_task.items():
        g3_vals = np.array([metrics_rows[k]["G3_incl"] for k in idxs], dtype=float)
        mu_g3 = float(np.mean(g3_vals))
        sd_g3 = float(np.std(g3_vals))
        ep_vals = np.array([metrics_rows[k]["final_score_norm"] for k in idxs], dtype=float)
        mu_ep = float(np.mean(ep_vals))

        for k in idxs:
            metrics_rows[k]["A_turn_proxy"] = round(
                float((metrics_rows[k]["G3_incl"] - mu_g3) / (sd_g3 + 1e-6)), 6
            )
            metrics_rows[k]["A_ep_proxy"] = round(float(metrics_rows[k]["final_score_norm"] - mu_ep), 6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: Offline 4-Pass HF Teacher-Forcing Scorer.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "protocol.yaml"))
    parser.add_argument("--data-dir", required=True, help="Directory containing episodes.jsonl and turns.jsonl.")
    parser.add_argument("--backend", default="hf", choices=["hf", "mock"])
    parser.add_argument("--batch-size", type=int, default=None, help="Override offline_scoring.batch_size_turns.")
    parser.add_argument("--skip-teacher", action="store_true", help="Skip Pass 4 (E18 Teacher Gap).")
    parser.add_argument("--run-strong-teacher", action="store_true", help="Also run Pass 4b with strong teacher (e.g. Qwen3-32B).")
    parser.add_argument(
        "--no-prefix-sharing",
        action="store_true",
        help="Score Passes 1-3 as independent full sequences instead of sharing the prompt KV cache (for cross-checking).",
    )
    args = parser.parse_args()

    cfg = load_protocol(args.config)
    data_dir = Path(args.data_dir)
    episodes_path = data_dir / "episodes.jsonl"
    turns_path = data_dir / "turns.jsonl"
    if not turns_path.exists():
        raise FileNotFoundError(f"Missing turns.jsonl in {data_dir}. Run 01_run_rollout.py first.")

    episodes = load_jsonl(episodes_path)
    turns = load_jsonl(turns_path)
    episodes_by_id = {e["episode_id"]: e for e in episodes}
    print(f"[Stage 2] Loaded {len(episodes)} episodes, {len(turns)} turns from {data_dir}")

    ocfg = cfg["offline_scoring"]
    mcfg = cfg["model"]
    batch_size = int(args.batch_size or ocfg["batch_size_turns"])
    neutral_cot = str(ocfg["neutral_cot_text"])
    enable_thinking = bool(mcfg["enable_thinking"])
    run_self_teacher = (not args.skip_teacher) and bool(ocfg["teacher"]["enable_self_teacher"])
    run_strong_teacher = (not args.skip_teacher) and (
        args.run_strong_teacher or bool(ocfg["teacher"]["enable_strong_teacher"])
    )

    is_mock = (args.backend == "mock")
    if is_mock:
        scorer: Any = MockOfflineScorer()
        # Re-hydrate MockTokenizer vocab by encoding all saved text deterministically if needed
        # (Use stored token_ids directly; decode works after warming up mock vocab)
        for t in turns:
            scorer.tokenizer.encode(t["raw_output"], add_special_tokens=False)
            scorer.tokenizer.encode(t["cot"], add_special_tokens=False)
            scorer.tokenizer.encode(t["action"], add_special_tokens=False)
    else:
        scorer = HFOfflineScorer(mcfg["hf_id"], mcfg.get("revision"), cfg)

    # The mock tokenizer's vocab is rebuilt per process, so decoding stored ids is only meaningful for HF
    alignment_ok = True
    mismatches: List[str] = []
    if is_mock:
        print("[Stage 2] Span decode check skipped for mock backend (non-persistent mock vocab).")
    else:
        n_checked, mismatches = verify_span_alignment(scorer.tokenizer, turns)
        alignment_ok = not mismatches
        print(f"[Stage 2] Span decode check: {n_checked - len(mismatches)}/{n_checked} spans decode exactly to parsed text.")
        for line in mismatches[:5]:
            print(f"  [Misaligned, excluded] {line}")

    # Build 4-pass sequences with strict bit-for-bit token alignment assertions
    four_pass_list: List[FourPassInputs] = []
    p1_reqs: List[Tuple[int, List[int], List[Tuple[int, int]]]] = []
    p2_reqs: List[Tuple[int, List[int], List[Tuple[int, int]]]] = []
    p3_reqs: List[Tuple[int, List[int], List[Tuple[int, int]]]] = []
    p4_reqs: List[Tuple[int, List[int], List[Tuple[int, int]]]] = []

    for idx, t in enumerate(turns):
        cot_span = tuple(t["cot_token_span"]) if t.get("cot_token_span") else None
        act_span = tuple(t["action_token_span"]) if t.get("action_token_span") else None
        fp = build_four_pass_sequences(
            tokenizer=scorer.tokenizer,
            history=t["history"],
            prompt_token_ids=t["prompt_token_ids"],
            generated_token_ids=t["generated_token_ids"],
            cot_token_span=cot_span,
            action_token_span=act_span,
            cot_prefix_tail=t.get("cot_prefix_tail", "Thought:"),
            action_prefix_tail=t.get("action_prefix_tail", "Action:"),
            neutral_cot_text=neutral_cot,
            enable_thinking=enable_thinking,
            build_teacher_pass=(run_self_teacher or run_strong_teacher),
        )
        four_pass_list.append(fp)

        p1_slices: List[Tuple[int, int]] = []
        if fp.p1_cot_slice:
            p1_slices.append(fp.p1_cot_slice)
        if fp.p1_act_slice:
            p1_slices.append(fp.p1_act_slice)
        if p1_slices:
            p1_reqs.append((idx, fp.p1_ids, p1_slices))

        if fp.p2_ids is not None and fp.p2_act_slice is not None:
            p2_reqs.append((idx, fp.p2_ids, [fp.p2_act_slice]))
        if fp.p3_ids is not None and fp.p3_act_slice is not None:
            p3_reqs.append((idx, fp.p3_ids, [fp.p3_act_slice]))
        if fp.p4_ids is not None and fp.p4_cot_slice is not None:
            p4_reqs.append((idx, fp.p4_ids, [fp.p4_cot_slice]))

    if alignment_ok:
        print("[Stage 2] Alignment assertions passed!")
    else:
        print(f"[Stage 2] ALIGNMENT CHECK FAILED: {len(mismatches)} misaligned spans were excluded (see above).")
    print(
        f"[Stage 2] Queued Pass1={len(p1_reqs)}, "
        f"Pass2={len(p2_reqs)}, Pass3={len(p3_reqs)}, Pass4={len(p4_reqs)}"
    )

    if is_mock or args.no_prefix_sharing:
        p1_res = _run_bucketed_pass(scorer, p1_reqs, batch_size, "Pass 1 (With-CoT: H_C, H_A, logp_A)", is_mock, "p1")
        p2_res = _run_bucketed_pass(scorer, p2_reqs, batch_size, "Pass 2 (Empty No-CoT: I_empty)", is_mock, "p2")
        p3_res = _run_bucketed_pass(scorer, p3_reqs, batch_size, "Pass 3 (Neutral No-CoT: I_neutral)", is_mock, "p3")
    else:
        p1_res, p2_res, p3_res = _run_shared_prefix_passes(scorer, turns, p1_reqs, p2_reqs, p3_reqs)

    p4_self_res: Dict[int, List[Tuple[List[float], List[float]]]] = {}
    if run_self_teacher and p4_reqs:
        p4_self_res = _run_bucketed_pass(
            scorer, p4_reqs, batch_size, "Pass 4a (State-OPD Self-Teacher: D_self)", is_mock, "p4"
        )

    # Free 8B before optionally loading 32B strong teacher on single 80GB A800
    tokenizer_ref = scorer.tokenizer
    if not is_mock:
        scorer.close()

    p4_strong_res: Dict[int, List[Tuple[List[float], List[float]]]] = {}
    if run_strong_teacher and p4_reqs:
        strong_id = str(ocfg["teacher"]["strong_teacher_hf_id"])
        strong_bs = int(ocfg["teacher"].get("strong_teacher_batch_size", 1))
        print(f"[Stage 2] Loading Strong Teacher '{strong_id}' (batch_size={strong_bs})...")
        if is_mock:
            p4_strong_res = _run_bucketed_pass(
                scorer, p4_reqs, strong_bs, f"Pass 4b ({strong_id})", True, "p4"
            )
        else:
            strong_scorer = HFOfflineScorer(strong_id, None, cfg)
            p4_strong_res = _run_bucketed_pass(
                strong_scorer, p4_reqs, strong_bs, f"Pass 4b ({strong_id})", False, "p4"
            )
            strong_scorer.close()

    # Assemble turn_metrics.jsonl and token_stats.parquet
    metrics_rows: List[Dict[str, Any]] = []
    token_rows: List[Dict[str, Any]] = []

    for idx, t in enumerate(turns):
        fp = four_pass_list[idx]
        cot_lps: List[float] = []
        cot_ents: List[float] = []
        act_lps_p1: List[float] = []
        act_ents_p1: List[float] = []

        if idx in p1_res:
            cursor = 0
            if fp.p1_cot_slice:
                cot_lps, cot_ents = p1_res[idx][cursor]
                cursor += 1
            if fp.p1_act_slice:
                act_lps_p1, act_ents_p1 = p1_res[idx][cursor]

        act_lps_p2 = p2_res[idx][0][0] if idx in p2_res else []
        act_lps_p3 = p3_res[idx][0][0] if idx in p3_res else []
        cot_lps_t_self = p4_self_res[idx][0][0] if idx in p4_self_res else []
        cot_lps_t_strong = p4_strong_res[idx][0][0] if idx in p4_strong_res else []

        # Bridge function I_t (E03 & E13)
        act_len = len(act_lps_p1)
        logp_a_cot = float(sum(act_lps_p1)) if act_len > 0 else None
        logp_a_empty = float(sum(act_lps_p2)) if len(act_lps_p2) == act_len and act_len > 0 else None
        logp_a_neutral = float(sum(act_lps_p3)) if len(act_lps_p3) == act_len and act_len > 0 else None

        I_empty_raw = (logp_a_cot - logp_a_empty) if (logp_a_cot is not None and logp_a_empty is not None) else None
        I_empty_norm = (I_empty_raw / act_len) if I_empty_raw is not None else None
        I_neutral_raw = (
            (logp_a_cot - logp_a_neutral) if (logp_a_cot is not None and logp_a_neutral is not None) else None
        )
        I_neutral_norm = (I_neutral_raw / act_len) if I_neutral_raw is not None else None
        I_fmt_delta = (
            (I_empty_norm - I_neutral_norm) if (I_empty_norm is not None and I_neutral_norm is not None) else None
        )

        # CoT & Action exact entropy stats (E05, E06, E17)
        h_head, h_mid, h_tail, h_peak_rel = _segment_entropies(cot_ents)
        cot_ids = (
            t["generated_token_ids"][t["cot_token_span"][0] : t["cot_token_span"][1]]
            if t.get("cot_token_span")
            else []
        )
        act_ids = (
            t["generated_token_ids"][t["action_token_span"][0] : t["action_token_span"][1]]
            if t.get("action_token_span")
            else []
        )

        top3_tokens: List[Dict[str, Any]] = []
        if cot_ents and len(cot_ents) == len(cot_ids):
            order = np.argsort(cot_ents)[::-1][:3]
            for pos in order:
                pos_i = int(pos)
                top3_tokens.append(
                    {
                        "pos": pos_i,
                        "rel_pos": round(pos_i / max(1, len(cot_ents) - 1), 3),
                        "token_id": int(cot_ids[pos_i]),
                        "token_text": tokenizer_ref.decode([cot_ids[pos_i]], skip_special_tokens=False),
                        "entropy": round(float(cot_ents[pos_i]), 4),
                    }
                )

        # Teacher Gap D_{t,i} = log pi_T(z_{t,i} | s_t) - log pi_S(z_{t,i} | h_t)
        d_self = (
            [float(lt - ls) for lt, ls in zip(cot_lps_t_self, cot_lps)]
            if len(cot_lps_t_self) == len(cot_lps) and cot_lps
            else []
        )
        d_strong = (
            [float(lt - ls) for lt, ls in zip(cot_lps_t_strong, cot_lps)]
            if len(cot_lps_t_strong) == len(cot_lps) and cot_lps
            else []
        )

        # Token-level rows for parquet (compact scalars only — NEVER full logits)
        for pos_i in range(len(cot_lps)):
            tid = int(cot_ids[pos_i]) if pos_i < len(cot_ids) else -1
            token_rows.append(
                {
                    "episode_id": t["episode_id"],
                    "turn_id": int(t["turn_id"]),
                    "segment": "cot",
                    "token_pos": pos_i,
                    "token_id": tid,
                    "token_text": tokenizer_ref.decode([tid], skip_special_tokens=False) if tid >= 0 else "",
                    "logp_student": cot_lps[pos_i],
                    "entropy_nats": cot_ents[pos_i],
                    "logp_teacher_self": cot_lps_t_self[pos_i] if pos_i < len(cot_lps_t_self) else np.nan,
                    "logp_teacher_strong": cot_lps_t_strong[pos_i] if pos_i < len(cot_lps_t_strong) else np.nan,
                }
            )
        for pos_j in range(len(act_lps_p1)):
            tid = int(act_ids[pos_j]) if pos_j < len(act_ids) else -1
            token_rows.append(
                {
                    "episode_id": t["episode_id"],
                    "turn_id": int(t["turn_id"]),
                    "segment": "action",
                    "token_pos": pos_j,
                    "token_id": tid,
                    "token_text": tokenizer_ref.decode([tid], skip_special_tokens=False) if tid >= 0 else "",
                    "logp_student": act_lps_p1[pos_j],
                    "entropy_nats": act_ents_p1[pos_j],
                    "logp_teacher_self": np.nan,
                    "logp_teacher_strong": np.nan,
                }
            )

        row: Dict[str, Any] = {
            "benchmark": t["benchmark"],
            "task_id": t["task_id"],
            "difficulty_tier": t["difficulty_tier"],
            "seed": t["seed"],
            "episode_id": t["episode_id"],
            "turn_id": t["turn_id"],
            "parse_ok": t["parse_ok"],
            "action_valid": t["action_valid"],
            "reward_raw": t["reward_raw"],
            "score": t["score"],
            "score_norm": t["score_norm"],
            # Bridge function
            "action_token_len": act_len,
            "logp_a_with_cot": round(logp_a_cot, 6) if logp_a_cot is not None else None,
            "logp_a_empty": round(logp_a_empty, 6) if logp_a_empty is not None else None,
            "logp_a_neutral": round(logp_a_neutral, 6) if logp_a_neutral is not None else None,
            "I_empty_norm": round(I_empty_norm, 6) if I_empty_norm is not None else None,
            "I_empty_raw": round(I_empty_raw, 6) if I_empty_raw is not None else None,
            "I_neutral_norm": round(I_neutral_norm, 6) if I_neutral_norm is not None else None,
            "I_neutral_raw": round(I_neutral_raw, 6) if I_neutral_raw is not None else None,
            "I_fmt_delta": round(I_fmt_delta, 6) if I_fmt_delta is not None else None,
            # CoT exact entropy (E05 & E17)
            "cot_token_len": len(cot_ents),
            "entropy_is_approx": False,
            "H_C_mean": round(_safe_mean(cot_ents), 6) if cot_ents else None,
            "H_C_p50": round(_safe_quantile(cot_ents, 0.50), 6) if cot_ents else None,
            "H_C_p90": round(_safe_quantile(cot_ents, 0.90), 6) if cot_ents else None,
            "H_C_max": round(float(max(cot_ents)), 6) if cot_ents else None,
            "H_C_std": round(_safe_std(cot_ents), 6) if cot_ents else None,
            "H_C_head_mean": round(h_head, 6) if h_head is not None else None,
            "H_C_mid_mean": round(h_mid, 6) if h_mid is not None else None,
            "H_C_tail_mean": round(h_tail, 6) if h_tail is not None else None,
            "H_C_peak_pos_rel": round(h_peak_rel, 4) if h_peak_rel is not None else None,
            "top3_high_entropy_tokens": top3_tokens,
            # Action exact entropy (E06)
            "H_A_mean": round(_safe_mean(act_ents_p1), 6) if act_ents_p1 else None,
            "H_A_max": round(float(max(act_ents_p1)), 6) if act_ents_p1 else None,
            # Teacher Gap (E18)
            "D_self_mean": round(_safe_mean(d_self), 6) if d_self else None,
            "D_self_p10": round(_safe_quantile(d_self, 0.10), 6) if d_self else None,
            "D_self_p50": round(_safe_quantile(d_self, 0.50), 6) if d_self else None,
            "D_self_p90": round(_safe_quantile(d_self, 0.90), 6) if d_self else None,
            "D_self_neg_frac": round(float(np.mean([x < 0 for x in d_self])), 4) if d_self else None,
            "corr_D_HC_within_turn": round(_safe_corr(d_self, cot_ents), 4)
            if _safe_corr(d_self, cot_ents) is not None
            else None,
            "D_strong_mean": round(_safe_mean(d_strong), 6) if d_strong else None,
            "D_strong_neg_frac": round(float(np.mean([x < 0 for x in d_strong])), 4) if d_strong else None,
        }
        metrics_rows.append(row)

    # Enrich with E08 future returns, E10 task-centered proxy advantages, and E11 5-way zero-reward labels
    compute_episode_level_targets_and_e11(turns, episodes_by_id, metrics_rows)

    metrics_path = data_dir / "turn_metrics.jsonl"
    token_path = data_dir / "token_stats.parquet"
    write_jsonl(metrics_path, metrics_rows)
    pd.DataFrame(token_rows).to_parquet(token_path, index=False)

    # Quick sanity printout
    i_emp = [r["I_empty_norm"] for r in metrics_rows if r["I_empty_norm"] is not None]
    i_neu = [r["I_neutral_norm"] for r in metrics_rows if r["I_neutral_norm"] is not None]
    hc_m = [r["H_C_mean"] for r in metrics_rows if r["H_C_mean"] is not None]
    print("\n" + "=" * 72)
    print(f"[Stage 2 Complete] Saved:\n  - {metrics_path} ({len(metrics_rows)} turns)\n  - {token_path} ({len(token_rows)} tokens)")
    if i_emp:
        print(
            f"  I_empty_norm:   mean={np.mean(i_emp):.4f}, P50={np.median(i_emp):.4f}, "
            f"pos_frac={np.mean([x > 0 for x in i_emp])*100:.1f}%"
        )
    if i_neu:
        print(
            f"  I_neutral_norm: mean={np.mean(i_neu):.4f}, P50={np.median(i_neu):.4f}, "
            f"pos_frac={np.mean([x > 0 for x in i_neu])*100:.1f}%"
        )
    if hc_m:
        print(f"  H_C_mean:       mean={np.mean(hc_m):.4f} nats, P50={np.median(hc_m):.4f}, P90={np.quantile(hc_m, 0.9):.4f}")
    finite_cols = ["I_empty_norm", "I_neutral_norm", "H_C_mean", "H_A_mean", "D_self_mean"]
    n_nonfinite = sum(
        1 for r in metrics_rows for c in finite_cols if r.get(c) is not None and not math.isfinite(r[c])
    )
    print(f"  Non-finite (NaN/Inf) values in {finite_cols}: {n_nonfinite}   [Gate == 0]")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()

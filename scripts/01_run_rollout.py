#!/usr/bin/env python3
"""
Stage 1: 在线玩游戏采集脚本 (覆盖 E01 smoke-test & E02 批量采集)

支持两个 Benchmark（scienceworld / textworld_express），三种推理后端（vllm / hf / mock）：
- 默认 vllm：开启 prefix caching + 官方 Qwen3-8B Non-Thinking 采样参数，直接拿精确 generated_token_ids
- 支持 --resume：每局结束立刻落盘 flush，断线重连不重跑
- 跑完自动打印 E01 四条验收红线自检报告
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# vLLM 0.25.x touches CUDA in the parent yet defaults to forking EngineCore, which then fails with
# "Cannot re-initialize CUDA in forked subprocess"; spawn is safe because main() is __main__-guarded.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import EpisodeSpec, get_episode_specs, load_protocol
from src.envs import create_env
from src.io_utils import (
    append_jsonl,
    ensure_dir,
    get_completed_episode_ids,
    load_jsonl,
    resolve_hf_revision,
)
from src.prompts import (
    build_system_prompt,
    format_assistant_turn,
    format_user_turn,
    parse_react_output,
)
from src.tokenization import (
    locate_token_spans,
    truncate_messages_to_budget,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


class VLLMGenerator:
    """vLLM rollout wrapper returning exact raw text + exact generated_token_ids."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        mcfg = cfg["model"]
        rcfg = cfg["rollout"]
        self.model_id = mcfg["hf_id"]
        self.revision = resolve_hf_revision(self.model_id, mcfg.get("revision"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=mcfg.get("revision"))
        self.llm = LLM(
            model=self.model_id,
            revision=mcfg.get("revision"),
            dtype=mcfg["dtype"],
            tensor_parallel_size=int(rcfg["tensor_parallel_size"]),
            gpu_memory_utilization=float(rcfg["gpu_memory_utilization"]),
            max_model_len=int(mcfg["max_context_tokens"]),
            enable_prefix_caching=bool(rcfg.get("enable_prefix_caching", True)),
            seed=int(cfg["experiment"]["seed"]),
        )
        self.sampling_params = SamplingParams(
            temperature=float(rcfg["temperature"]),
            top_p=float(rcfg["top_p"]),
            top_k=int(rcfg["top_k"]),
            min_p=float(rcfg.get("min_p", 0.0)),
            presence_penalty=float(rcfg.get("presence_penalty", 0.0)),
            repetition_penalty=float(rcfg.get("repetition_penalty", 1.0)),
            max_tokens=int(rcfg["max_tokens_per_turn"]),
            stop=list(rcfg["stop_sequences"]),
            skip_special_tokens=False,
        )

    def generate_batch(
        self,
        prompt_token_ids_list: List[List[int]],
        turn_hints: List[Tuple[int, int]],
    ) -> List[Tuple[str, List[int]]]:
        # Pass exact prompt_token_ids to vLLM so tokenization matches Stage 2 100%
        inputs = [{"prompt_token_ids": ids} for ids in prompt_token_ids_list]
        outputs = self.llm.generate(inputs, self.sampling_params, use_tqdm=False)
        results: List[Tuple[str, List[int]]] = []
        for out in outputs:
            cand = out.outputs[0]
            tok_ids = [int(t) for t in cand.token_ids]
            raw_text = self.tokenizer.decode(tok_ids, skip_special_tokens=True)
            results.append((raw_text, tok_ids))
        return results


class HFGenerator:
    """Fallback HF transformers rollout wrapper (works without vLLM)."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        mcfg = cfg["model"]
        rcfg = cfg["rollout"]
        self.model_id = mcfg["hf_id"]
        self.revision = resolve_hf_revision(self.model_id, mcfg.get("revision"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=mcfg.get("revision"))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        dtype = torch.bfloat16 if mcfg["dtype"] == "bfloat16" else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=mcfg.get("revision"),
            dtype=dtype,
            device_map="auto",
        ).eval()
        self.rcfg = rcfg

    def generate_batch(
        self,
        prompt_token_ids_list: List[List[int]],
        turn_hints: List[Tuple[int, int]],
    ) -> List[Tuple[str, List[int]]]:
        import torch

        rcfg = self.rcfg
        stop_strs = list(rcfg["stop_sequences"])
        results: List[Tuple[str, List[int]]] = []
        for prompt_ids in prompt_token_ids_list:
            inp = torch.tensor([prompt_ids], dtype=torch.long, device=self.model.device)
            with torch.no_grad():
                gen = self.model.generate(
                    input_ids=inp,
                    do_sample=True,
                    temperature=float(rcfg["temperature"]),
                    top_p=float(rcfg["top_p"]),
                    top_k=int(rcfg["top_k"]),
                    max_new_tokens=int(rcfg["max_tokens_per_turn"]),
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            gen_ids = [int(t) for t in gen[0, len(prompt_ids) :].tolist()]
            raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            # Enforce stop sequences manually for HF backend
            for s in stop_strs:
                if s and s in raw_text:
                    idx = raw_text.index(s)
                    raw_text = raw_text[:idx]
                    gen_ids = self.tokenizer.encode(raw_text, add_special_tokens=False)
            results.append((raw_text, gen_ids))
        return results


class MockGenerator:
    """Fast local CPU mock generator for testing the entire pipeline end-to-end."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        from src.mock_model import MockRolloutBackend, MockTokenizer

        self.model_id = cfg["model"]["hf_id"] + " (MOCK)"
        self.revision = "mock_v1"
        self.tokenizer = MockTokenizer()
        self._backend = MockRolloutBackend(self.tokenizer, seed=int(cfg["experiment"]["seed"]))

    def generate_batch(
        self,
        prompt_token_ids_list: List[List[int]],
        turn_hints: List[Tuple[int, int]],
    ) -> List[Tuple[str, List[int]]]:
        texts = [self.tokenizer.decode(ids, skip_special_tokens=False) for ids in prompt_token_ids_list]
        return self._backend.generate_batch(texts, turn_hints)


def _start_episode(spec: EpisodeSpec, env: Any) -> Dict[str, Any]:
    reset_res = env.reset(spec)
    sys_prompt = build_system_prompt(spec.benchmark, reset_res.task_description, spec.task_id)
    return {
        "spec": spec,
        "env": env,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": format_user_turn(0, reset_res.observation)},
        ],
        "obs": reset_res.observation,
        "turn_id": 0,
        "turns_records": [],
        "total_reward": 0.0,
        "last_score": reset_res.score,
        "last_score_norm": reset_res.score_norm,
        "terminal_success": False,
        "termination_reason": "max_turns",
        "parse_fail_count": 0,
        "consec_parse_fail": 0,
        "invalid_action_count": 0,
        "truncated_turns_count": 0,
    }


def _episode_record(st: Dict[str, Any], generator: Any, enable_thinking: bool) -> Dict[str, Any]:
    spec = st["spec"]
    return {
        "benchmark": spec.benchmark,
        "task_id": spec.task_id,
        "difficulty_tier": spec.difficulty_tier,
        "seed": spec.seed,
        "episode_id": spec.episode_id,
        "model_hf_id": generator.model_id,
        "model_revision": generator.revision,
        "enable_thinking": enable_thinking,
        "num_turns": len(st["turns_records"]),
        "total_reward": round(float(st["total_reward"]), 6),
        "final_score": round(float(st["last_score"]), 6),
        "final_score_norm": round(float(st["last_score_norm"]), 6),
        "terminal_success": bool(st["terminal_success"]),
        "termination_reason": st["termination_reason"],
        "parse_fail_count": int(st["parse_fail_count"]),
        "invalid_action_count": int(st["invalid_action_count"]),
        "truncated_turns_count": int(st["truncated_turns_count"]),
    }


def run_rollout_pool(
    specs: List[EpisodeSpec],
    envs: List[Any],
    generator: Any,
    cfg: Dict[str, Any],
    on_episode_done: Callable[[Dict[str, Any], List[Dict[str, Any]]], None],
) -> None:
    """
    Keep every env slot busy: all active episodes share one batched generate call per round,
    env reset/step run concurrently in threads (each env owns its own JVM), and a finished
    episode is flushed immediately and its slot refilled with the next pending spec.
    """
    mcfg = cfg["model"]
    rcfg = cfg["rollout"]
    max_turns = int(rcfg["max_turns_per_episode"])
    max_prompt_tokens = int(mcfg["max_context_tokens"]) - int(rcfg["max_tokens_per_turn"]) - 64
    enable_thinking = bool(mcfg["enable_thinking"])
    max_consec_parse_fail = int(rcfg.get("max_consecutive_parse_fails", 3))

    pending = deque(specs)
    slots: List[Optional[Dict[str, Any]]] = [None] * len(envs)

    with ThreadPoolExecutor(max_workers=len(envs)) as pool:

        def _refill(slot_ids: List[int]) -> None:
            jobs = {i: pool.submit(_start_episode, pending.popleft(), envs[i]) for i in slot_ids if pending}
            for i, fut in jobs.items():
                slots[i] = fut.result()

        _refill(list(range(len(envs))))
        while True:
            active = [i for i, st in enumerate(slots) if st is not None]
            if not active:
                break

            prompt_ids_batch: List[List[int]] = []
            trimmed_msgs_batch: List[List[Dict[str, str]]] = []
            truncated_batch: List[bool] = []
            turn_hints: List[Tuple[int, int]] = []
            for i in active:
                st = slots[i]
                trimmed_msgs, p_ids, was_trunc = truncate_messages_to_budget(
                    st["messages"],
                    generator.tokenizer,
                    max_prompt_tokens=max_prompt_tokens,
                    enable_thinking=enable_thinking,
                )
                prompt_ids_batch.append(p_ids)
                trimmed_msgs_batch.append(trimmed_msgs)
                truncated_batch.append(was_trunc)
                turn_hints.append((st["spec"].seed, st["turn_id"]))

            gen_batch = generator.generate_batch(prompt_ids_batch, turn_hints)
            parsed_batch = [parse_react_output(raw_output) for raw_output, _ in gen_batch]
            # Step every env concurrently (empty string when no action could be extracted)
            step_futs = [
                pool.submit(slots[i]["env"].step, parsed.action if parsed.action else "")
                for i, parsed in zip(active, parsed_batch)
            ]

            finished: List[int] = []
            for i, p_ids, trimmed_msgs, was_trunc, (raw_output, gen_ids), parsed, fut in zip(
                active, prompt_ids_batch, trimmed_msgs_batch, truncated_batch, gen_batch, parsed_batch, step_futs
            ):
                st = slots[i]
                spec: EpisodeSpec = st["spec"]
                turn_id = st["turn_id"]
                span_info = locate_token_spans(generator.tokenizer, gen_ids, parsed)
                step_res = fut.result()

                if was_trunc:
                    st["truncated_turns_count"] += 1
                if parsed.parse_ok:
                    st["consec_parse_fail"] = 0
                else:
                    st["parse_fail_count"] += 1
                    st["consec_parse_fail"] += 1
                if not step_res.action_valid:
                    st["invalid_action_count"] += 1

                st["total_reward"] += step_res.reward_raw
                st["last_score"] = step_res.score
                st["last_score_norm"] = step_res.score_norm

                # Decide episode termination
                done = bool(step_res.done)
                if step_res.terminal_success:
                    st["terminal_success"] = True
                    st["termination_reason"] = "success"
                    done = True
                elif step_res.terminal_failure and done:
                    st["termination_reason"] = "failure"
                elif st["consec_parse_fail"] >= max_consec_parse_fail:
                    done = True
                    st["termination_reason"] = "repeated_parse_fail"
                elif turn_id + 1 >= max_turns:
                    done = True
                    st["termination_reason"] = "max_turns"

                st["turns_records"].append(
                    {
                        "benchmark": spec.benchmark,
                        "task_id": spec.task_id,
                        "difficulty_tier": spec.difficulty_tier,
                        "seed": spec.seed,
                        "episode_id": spec.episode_id,
                        "turn_id": turn_id,
                        "history": trimmed_msgs,
                        "observation": st["obs"],
                        "next_observation": step_res.observation,
                        "raw_output": raw_output,
                        "cot": parsed.cot,
                        "action": parsed.action,
                        "parse_ok": parsed.parse_ok,
                        "parse_error_type": parsed.parse_error_type,
                        "action_valid": step_res.action_valid,
                        "action_in_valid_list": step_res.action_in_valid_list,
                        "reward_raw": step_res.reward_raw,
                        "reward_aux": step_res.reward_aux,
                        "reward_source": step_res.reward_source,
                        "score": step_res.score,
                        "score_norm": step_res.score_norm,
                        "done": done,
                        "truncated": was_trunc,
                        "prompt_token_ids": p_ids,
                        "generated_token_ids": gen_ids,
                        "cot_token_span": list(span_info.cot_token_span) if span_info.cot_token_span else None,
                        "action_token_span": list(span_info.action_token_span) if span_info.action_token_span else None,
                        "cot_prefix_tail": span_info.cot_prefix_tail,
                        "action_prefix_tail": span_info.action_prefix_tail,
                        "messages_history_len": len(trimmed_msgs),
                    }
                )

                # Advance dialogue history faithfully
                assistant_msg = (
                    format_assistant_turn(parsed.cot, parsed.action) if parsed.parse_ok else raw_output.strip()
                )
                st["messages"].append({"role": "assistant", "content": assistant_msg})
                st["messages"].append({"role": "user", "content": format_user_turn(turn_id + 1, step_res.observation)})
                st["obs"] = step_res.observation
                st["turn_id"] += 1

                if done:
                    on_episode_done(_episode_record(st, generator, enable_thinking), st["turns_records"])
                    slots[i] = None
                    finished.append(i)

            _refill(finished)


def print_rollout_summary(episodes_path: Path, turns_path: Path, mode: str) -> None:
    eps = load_jsonl(episodes_path)
    turns = load_jsonl(turns_path)
    if not eps or not turns:
        return
    n_ep = len(eps)
    n_turn = len(turns)
    parse_rate = sum(1 for t in turns if t["parse_ok"]) / n_turn
    valid_rate = sum(1 for t in turns if t["action_valid"]) / n_turn
    succ_rate = sum(1 for e in eps if e["terminal_success"]) / n_ep
    pos_score_rate = sum(1 for e in eps if e["final_score_norm"] > 0) / n_ep
    avg_turns = np.mean([e["num_turns"] for e in eps])
    avg_score = np.mean([e["final_score_norm"] for e in eps])

    print("\n" + "=" * 72)
    print(f"Stage 1 Rollout Summary ({mode.upper()}) -> {episodes_path.parent}")
    print("=" * 72)
    print(f"  Episodes:           {n_ep}")
    print(f"  Total Turns:        {n_turn} (avg {avg_turns:.1f} turns/ep)")
    print(f"  Parse OK Rate:      {parse_rate*100:.1f}%   [Gate >= 90%]")
    print(f"  Action Valid Rate:  {valid_rate*100:.1f}%   [Gate >= 60%]")
    print(f"  Success Rate:       {succ_rate*100:.1f}%   | Positive-Score Episodes: {pos_score_rate*100:.1f}%")
    print(f"  Mean Final Score:   {avg_score:.3f} (normalized 0-1)")
    reasons: Dict[str, int] = {}
    for e in eps:
        reasons[e["termination_reason"]] = reasons.get(e["termination_reason"], 0) + 1
    print(f"  Termination:        {reasons}")
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for e in eps:
        by_task.setdefault(e["task_id"], []).append(e)
    for tid, items in by_task.items():
        sr = np.mean([x["terminal_success"] for x in items]) * 100
        sc = np.mean([x["final_score_norm"] for x in items])
        print(f"    - {tid:22s}: n={len(items):2d}, success={sr:5.1f}%, mean_score_norm={sc:.3f}")

    # Inspect 2 sample parsed spans so user can verify Action: prefix was stripped
    print("\n  Sample parsed turns (verify CoT/Action clean separation):")
    shown = 0
    for t in turns:
        if t["parse_ok"] and t["cot_token_span"] and t["action_token_span"]:
            print(f"    [{t['episode_id']} t={t['turn_id']}]")
            print(f"      CoT    (span={t['cot_token_span']}, tail={t['cot_prefix_tail']!r}): {t['cot'][:90]}")
            print(f"      Action (span={t['action_token_span']}, tail={t['action_prefix_tail']!r}): {t['action']!r}")
            shown += 1
            if shown >= 2:
                break
    print("=" * 72 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: Run E01 smoke-test or E02 full rollout.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "protocol.yaml"))
    parser.add_argument("--benchmark", required=True, choices=["scienceworld", "textworld_express"])
    parser.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    parser.add_argument("--backend", default=None, choices=["vllm", "hf", "mock"])
    parser.add_argument("--use-mock-env", action="store_true", help="Use local MockTextEnv instead of real game.")
    parser.add_argument("--num-parallel-envs", type=int, default=None, help="Override rollout.num_parallel_envs.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional cap for debugging.")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: data/{benchmark}_{mode}).")
    parser.add_argument("--resume", action="store_true", help="Skip already completed episodes in episodes.jsonl.")
    args = parser.parse_args()

    cfg = load_protocol(args.config)
    seed_everything(int(cfg["experiment"]["seed"]))

    backend = args.backend or cfg["rollout"]["backend"]
    use_mock_env = args.use_mock_env or (backend == "mock")
    specs = get_episode_specs(cfg, args.benchmark, mode=args.mode, max_episodes=args.max_episodes)

    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "data" / f"{args.benchmark}_{args.mode}")
    ensure_dir(out_dir)
    episodes_path = out_dir / "episodes.jsonl"
    turns_path = out_dir / "turns.jsonl"

    if args.resume:
        done_ids = get_completed_episode_ids(episodes_path)
        before = len(specs)
        specs = [s for s in specs if s.episode_id not in done_ids]
        print(f"[Resume] Found {len(done_ids)} completed episodes; remaining {len(specs)}/{before}.")
    else:
        for p in (episodes_path, turns_path):
            if p.exists():
                p.unlink()

    if not specs:
        print("All requested episodes are already completed.")
        print_rollout_summary(episodes_path, turns_path, args.mode)
        return

    print(f"[Stage 1] Initializing backend='{backend}' for {args.benchmark} ({args.mode}, {len(specs)} episodes)...")
    if backend == "vllm":
        generator: Any = VLLMGenerator(cfg)
    elif backend == "hf":
        generator = HFGenerator(cfg)
    else:
        generator = MockGenerator(cfg)

    n_parallel = int(args.num_parallel_envs or cfg["rollout"].get("num_parallel_envs", 1))
    n_parallel = max(1, min(n_parallel, len(specs)))
    step_limit = int(cfg["benchmarks"][args.benchmark].get("env_step_limit", cfg["rollout"]["max_turns_per_episode"]))

    # Each env launches its own JVM; start them concurrently
    with ThreadPoolExecutor(max_workers=n_parallel) as pool:
        envs = list(
            pool.map(lambda _: create_env(args.benchmark, env_step_limit=step_limit, use_mock=use_mock_env), range(n_parallel))
        )
    print(f"[Stage 1] Running {len(specs)} episodes on {n_parallel} concurrent env slots.")
    pbar = tqdm(total=len(specs), desc=f"Rollout {args.benchmark} ({args.mode})")

    def _flush(ep_rec: Dict[str, Any], turn_recs: List[Dict[str, Any]]) -> None:
        append_jsonl(episodes_path, [ep_rec])
        append_jsonl(turns_path, turn_recs)
        pbar.update(1)

    try:
        run_rollout_pool(specs, envs, generator, cfg, _flush)
    finally:
        pbar.close()
        for env in envs:
            env.close()

    print_rollout_summary(episodes_path, turns_path, args.mode)


if __name__ == "__main__":
    main()

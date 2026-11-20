"""Lightweight deterministic MockTokenizer & MockScorer for local CPU end-to-end verification."""

from __future__ import annotations

import hashlib
import math
import random
import re
from typing import Dict, List, Tuple


class MockTokenizer:
    """Simple reversible regex tokenizer mimicking BPE leading-space tokens."""

    def __init__(self) -> None:
        self._str_to_id: Dict[str, int] = {"<pad>": 0, "<eos>": 1}
        self._id_to_str: Dict[int, str] = {0: "<pad>", 1: "<eos>"}
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.all_special_ids = [0, 1]

    def _get_id(self, tok: str) -> int:
        if tok not in self._str_to_id:
            tid = len(self._str_to_id)
            self._str_to_id[tok] = tid
            self._id_to_str[tid] = tok
        return self._str_to_id[tok]

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        # Split so leading single space attaches to following word (just like Qwen BPE!)
        pieces = re.findall(r" ?[A-Za-z0-9_]+| ?[^\sA-Za-z0-9_]|\n|[ \t]+", text)
        return [self._get_id(p) for p in pieces]

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        out = []
        for tid in ids:
            if skip_special_tokens and tid in self.all_special_ids:
                continue
            out.append(self._id_to_str.get(int(tid), ""))
        return "".join(out)

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str:
        parts = [f"<|{m['role']}|>\n{m['content']}\n" for m in messages]
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        text = "".join(parts)
        return self.encode(text) if tokenize else text


class MockRolloutBackend:
    """Produces varied ReAct traces (success, failure, loops, invalid actions, parse errors)."""

    def __init__(self, tokenizer: MockTokenizer, seed: int = 42) -> None:
        self.tokenizer = tokenizer
        self.rng = random.Random(seed)
        self._plan = [
            ("I should first head to the kitchen to locate the container.", "go to kitchen"),
            ("Now I see the fridge, so I will open it to check inside.", "open fridge"),
            ("The target object is inside the fridge; I should pick it up.", "take target object"),
            ("Next I need to prepare and use the target object.", "use target object"),
            ("Everything is ready; I can now finish the task.", "finish task"),
        ]

    def generate_batch(self, prompt_texts: List[str], turn_hints: List[Tuple[int, int]]) -> List[Tuple[str, List[int]]]:
        results: List[Tuple[str, List[int]]] = []
        for ptext, (ep_seed, turn_id) in zip(prompt_texts, turn_hints):
            local_rng = random.Random(ep_seed * 1009 + turn_id * 37)
            # Half episodes follow the plan with occasional slips; half struggle/loop/fail
            good_ep = (ep_seed % 2 == 0)
            r = local_rng.random()
            if r < 0.04:
                raw = "Let me check the room first without formatting."
            elif good_ep and turn_id < len(self._plan):
                if r < 0.15:
                    raw = "Thought: Maybe I should check my items before moving on.\nAction: inventory"
                else:
                    cot, act = self._plan[turn_id]
                    raw = f"Thought: {cot}\nAction: {act}"
            elif not good_ep:
                if r < 0.25:
                    raw = "Thought: I am not sure what is here, let me try a random command.\nAction: teleport mars"
                elif r < 0.65:
                    raw = "Thought: Let me look around again to see if anything changed.\nAction: look around"
                elif turn_id == 0:
                    raw = f"Thought: {self._plan[0][0]}\nAction: {self._plan[0][1]}"
                elif turn_id == 1:
                    raw = f"Thought: {self._plan[1][0]}\nAction: {self._plan[1][1]}"
                else:
                    raw = "Thought: I will inspect my inventory to plan the next step.\nAction: inventory"
            else:
                raw = "Thought: Let me finish up the remaining step.\nAction: finish task"

            tok_ids = self.tokenizer.encode(raw, add_special_tokens=False) + [self.tokenizer.eos_token_id]
            results.append((raw, tok_ids))
        return results


def mock_score_sequence(
    input_ids: List[int],
    target_slice: Tuple[int, int],
    pass_tag: str,
) -> Tuple[List[float], List[float]]:
    """
    Deterministic mock log-probs and entropies (nats) for a target token slice.
    Injects mild realistic correlation so E03-E18 plots show non-degenerate structure.
    """
    start, end = target_slice
    logps: List[float] = []
    entropies: List[float] = []
    for pos in range(start, end):
        ctx_hash = hashlib.md5(
            f"{pass_tag}:{pos}:{input_ids[max(0, pos-4):pos+1]}".encode("utf-8")
        ).hexdigest()
        u = int(ctx_hash[:8], 16) / 0xFFFFFFFF
        v = int(ctx_hash[8:16], 16) / 0xFFFFFFFF
        # Pass 1 slightly higher logp on average than Pass 2/3, with realistic negative tail
        base_shift = -0.7 if pass_tag == "p1" else (-1.1 if pass_tag == "p2" else -0.95)
        if pass_tag == "p4":
            base_shift = -0.85
        lp = base_shift - 1.6 * u
        ent = 0.25 + 2.2 * (v ** 1.3) + (0.4 * (pos - start) / max(1, end - start))
        logps.append(round(float(lp), 6))
        entropies.append(round(float(max(1e-4, ent)), 6))
    return logps, entropies

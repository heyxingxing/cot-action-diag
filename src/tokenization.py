"""Exact token-span alignment & 4-pass teacher-forcing sequence builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.prompts import ParsedReactOutput, build_state_opd_messages


def apply_chat_template_safe(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    enable_thinking: bool = False,
    add_generation_prompt: bool = True,
) -> str:
    """
    Render chat template with Qwen3 enable_thinking toggle when supported,
    falling back gracefully for tokenizers without that kwarg (e.g. MockTokenizer).
    """
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
    # Minimal deterministic fallback for simple mock tokenizers
    parts = [f"<|{m['role']}|>\n{m['content']}\n" for m in messages]
    if add_generation_prompt:
        parts.append("<|assistant|>\n")
    return "".join(parts)


def encode_no_special(tokenizer: Any, text: str) -> List[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def truncate_messages_to_budget(
    messages: List[Dict[str, str]],
    tokenizer: Any,
    max_prompt_tokens: int,
    enable_thinking: bool = False,
) -> Tuple[List[Dict[str, str]], List[int], bool]:
    """
    Ensure prompt fits in max_prompt_tokens by dropping the oldest intermediate
    (assistant, user) pairs while ALWAYS keeping:
      - messages[0] (system prompt + task goal)
      - messages[1] (Turn 0 initial observation)
      - messages[-1] (current turn observation)
    Returns (trimmed_messages, prompt_token_ids, was_truncated).
    """
    cur = list(messages)
    prompt_text = apply_chat_template_safe(tokenizer, cur, enable_thinking=enable_thinking, add_generation_prompt=True)
    prompt_ids = encode_no_special(tokenizer, prompt_text)
    truncated = False

    # Drop oldest intermediate (assistant, user) pair at indices (2, 3) until under budget
    while len(prompt_ids) > max_prompt_tokens and len(cur) > 4:
        cur = cur[:2] + cur[4:]
        truncated = True
        prompt_text = apply_chat_template_safe(
            tokenizer, cur, enable_thinking=enable_thinking, add_generation_prompt=True
        )
        prompt_ids = encode_no_special(tokenizer, prompt_text)

    return cur, prompt_ids, truncated


def _token_char_offsets_from_ids(tokenizer: Any, token_ids: List[int]) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Compute exact character (start, end) offsets for each token in token_ids by
    decoding prefixes incrementally. 100% faithful to the actual generated token IDs
    even when re-tokenizing raw text might differ due to BPE merges.
    """
    offsets: List[Tuple[int, int]] = []
    prev_text = ""
    for i in range(1, len(token_ids) + 1):
        cur_text = tokenizer.decode(token_ids[:i], skip_special_tokens=False)
        offsets.append((len(prev_text), len(cur_text)))
        prev_text = cur_text
    return prev_text, offsets


def _strip_special_and_trailing_ws_tokens(
    tokenizer: Any,
    token_ids: List[int],
    start: int,
    end: int,
) -> Tuple[int, int]:
    """Trim trailing EOS/special tokens or pure-newline tokens from [start, end)."""
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    while end > start:
        tid = token_ids[end - 1]
        if tid in special_ids:
            end -= 1
            continue
        tok_str = tokenizer.decode([tid], skip_special_tokens=False)
        if tok_str in ("\n", "\r\n", "\r", ""):
            end -= 1
            continue
        break
    return start, end


@dataclass
class TokenSpanInfo:
    cot_token_span: Optional[Tuple[int, int]]       # [start, end) in generated_token_ids
    action_token_span: Optional[Tuple[int, int]]    # [start, end) in generated_token_ids
    cot_prefix_tail: str                            # exact suffix before cot_token_ids (e.g. "Thought:" or "Thought: ")
    action_prefix_tail: str                         # exact suffix before action_token_ids (e.g. "Action:" or "Action: ")


def locate_token_spans(
    tokenizer: Any,
    generated_token_ids: List[int],
    parsed: ParsedReactOutput,
) -> TokenSpanInfo:
    """
    Locate exact [start, end) token indices in generated_token_ids for:
      - CoT content (excluding 'Thought:')
      - Action content (excluding '\\nAction:' and trailing EOS/newlines)

    Also records the exact prefix tail ('Action:' vs 'Action: ') immediately preceding
    action_token_span so Pass 2 (empty CoT) and Pass 3 (neutral CoT) concatenate the
    EXACT SAME action_token_ids with zero BPE mismatch!
    """
    if not generated_token_ids:
        return TokenSpanInfo(None, None, "Thought:", "Action:")

    full_text, offsets = _token_char_offsets_from_ids(tokenizer, generated_token_ids)

    def _find_span(
        label_end: Optional[int],
        char_span: Optional[Tuple[int, int]],
        label_name: str,
    ) -> Tuple[Optional[Tuple[int, int]], str]:
        if label_end is None or char_span is None:
            return None, f"{label_name}:"
        c_start, c_end = char_span
        if c_end <= c_start:
            return None, f"{label_name}:"

        # Start at first token whose char_start >= label_end (strictly after 'Action:' / 'Thought:')
        # and that overlaps or precedes content.
        tok_start: Optional[int] = None
        for idx, (ts, te) in enumerate(offsets):
            if te <= label_end:
                continue
            if ts < label_end:
                # Token straddles ':' (e.g. ': '), skip it so we don't score ':'
                continue
            tok_start = idx
            break

        if tok_start is None:
            # Edge case: content is inside the token straddling label_end; fallback to token covering c_start
            for idx, (ts, te) in enumerate(offsets):
                if te > c_start:
                    tok_start = idx
                    break
        if tok_start is None:
            return None, f"{label_name}:"

        # End at first token whose char_start >= c_end
        tok_end = tok_start
        for idx in range(tok_start, len(offsets)):
            ts, te = offsets[idx]
            if ts >= c_end:
                break
            tok_end = idx + 1

        tok_start, tok_end = _strip_special_and_trailing_ws_tokens(
            tokenizer, generated_token_ids, tok_start, tok_end
        )
        if tok_end <= tok_start:
            return None, f"{label_name}:"

        # Exact gap between label_end and offsets[tok_start][0] (typically "" when BPE attaches leading space to word)
        gap = full_text[label_end : offsets[tok_start][0]]
        prefix_tail = f"{label_name}:{gap}"
        return (tok_start, tok_end), prefix_tail

    cot_span, cot_tail = _find_span(parsed.thought_label_end, parsed.cot_char_span, "Thought")
    act_span, act_tail = _find_span(parsed.action_label_end, parsed.action_char_span, "Action")

    return TokenSpanInfo(
        cot_token_span=cot_span,
        action_token_span=act_span,
        cot_prefix_tail=cot_tail,
        action_prefix_tail=act_tail,
    )


@dataclass
class FourPassInputs:
    # Pass 1: full prompt + generated_token_ids (scores BOTH CoT & Action)
    p1_ids: List[int]
    p1_cot_slice: Optional[Tuple[int, int]]
    p1_act_slice: Optional[Tuple[int, int]]
    # Pass 2: full prompt + empty Thought + EXACT same action_token_ids
    p2_ids: Optional[List[int]]
    p2_act_slice: Optional[Tuple[int, int]]
    # Pass 3: full prompt + neutral Thought + EXACT same action_token_ids
    p3_ids: Optional[List[int]]
    p3_act_slice: Optional[Tuple[int, int]]
    # Pass 4: State-OPD pruned prompt (no historical CoT) + Thought prefix + EXACT same cot_token_ids
    p4_ids: Optional[List[int]]
    p4_cot_slice: Optional[Tuple[int, int]]


def build_four_pass_sequences(
    tokenizer: Any,
    history: List[Dict[str, str]],
    prompt_token_ids: List[int],
    generated_token_ids: List[int],
    cot_token_span: Optional[Tuple[int, int]],
    action_token_span: Optional[Tuple[int, int]],
    cot_prefix_tail: str,
    action_prefix_tail: str,
    neutral_cot_text: str,
    enable_thinking: bool = False,
    build_teacher_pass: bool = True,
) -> FourPassInputs:
    """
    Build exact token ID sequences for all 4 offline teacher-forcing passes, with hard
    assertions guaranteeing bit-for-bit identical action_token_ids across Pass 1/2/3 and
    identical cot_token_ids across Pass 1/4.
    """
    p_len = len(prompt_token_ids)
    p1_ids = list(prompt_token_ids) + list(generated_token_ids)

    p1_cot_slice = (p_len + cot_token_span[0], p_len + cot_token_span[1]) if cot_token_span else None
    p1_act_slice = (p_len + action_token_span[0], p_len + action_token_span[1]) if action_token_span else None

    p2_ids, p2_act_slice = None, None
    p3_ids, p3_act_slice = None, None

    if action_token_span is not None:
        a0, a1 = action_token_span
        act_ids = list(generated_token_ids[a0:a1])
        if act_ids:
            # Pass 2: Empty CoT baseline -> "Thought:\n" + action_prefix_tail + act_ids
            p2_prefix = encode_no_special(tokenizer, f"Thought:\n{action_prefix_tail}")
            p2_ids = list(prompt_token_ids) + p2_prefix + act_ids
            p2_start = p_len + len(p2_prefix)
            p2_act_slice = (p2_start, p2_start + len(act_ids))

            # Pass 3: Neutral CoT baseline -> "Thought: <neutral>\n" + action_prefix_tail + act_ids
            p3_prefix = encode_no_special(tokenizer, f"Thought: {neutral_cot_text.strip()}\n{action_prefix_tail}")
            p3_ids = list(prompt_token_ids) + p3_prefix + act_ids
            p3_start = p_len + len(p3_prefix)
            p3_act_slice = (p3_start, p3_start + len(act_ids))

            # Hard integrity assertions: Pass 1, 2, 3 MUST score the exact same action tokens!
            assert p1_ids[p1_act_slice[0] : p1_act_slice[1]] == act_ids
            assert p2_ids[p2_act_slice[0] : p2_act_slice[1]] == act_ids
            assert p3_ids[p3_act_slice[0] : p3_act_slice[1]] == act_ids

    p4_ids, p4_cot_slice = None, None
    if build_teacher_pass and cot_token_span is not None:
        c0, c1 = cot_token_span
        cot_ids = list(generated_token_ids[c0:c1])
        if cot_ids:
            pruned_history = build_state_opd_messages(history)
            pruned_prompt_text = apply_chat_template_safe(
                tokenizer, pruned_history, enable_thinking=enable_thinking, add_generation_prompt=True
            )
            pruned_prompt_ids = encode_no_special(tokenizer, pruned_prompt_text)
            p4_prefix = encode_no_special(tokenizer, cot_prefix_tail)
            p4_ids = pruned_prompt_ids + p4_prefix + cot_ids
            p4_start = len(pruned_prompt_ids) + len(p4_prefix)
            p4_cot_slice = (p4_start, p4_start + len(cot_ids))

            assert p1_ids[p1_cot_slice[0] : p1_cot_slice[1]] == cot_ids
            assert p4_ids[p4_cot_slice[0] : p4_cot_slice[1]] == cot_ids

    return FourPassInputs(
        p1_ids=p1_ids,
        p1_cot_slice=p1_cot_slice,
        p1_act_slice=p1_act_slice,
        p2_ids=p2_ids,
        p2_act_slice=p2_act_slice,
        p3_ids=p3_ids,
        p3_act_slice=p3_act_slice,
        p4_ids=p4_ids,
        p4_cot_slice=p4_cot_slice,
    )

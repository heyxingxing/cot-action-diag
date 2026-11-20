"""Shared ReAct prompts, strict output parser, and State-OPD history pruning."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


SHARED_SYSTEM_HEADER = """You are an interactive agent solving a multi-step task in a text environment.
At every turn, you MUST respond in the exact two-line format below:
Thought: <concise reasoning in 1-3 sentences about the current state and next step>
Action: <exactly one valid environment command>

Rules:
- Always output "Thought:" first, followed by a newline and "Action:".
- Output ONLY one action per turn. Do not add extra explanations after the action.
- Do not invent objects or commands that are not allowed by the environment."""


# Interface manual only (command forms, their preconditions and effects), verified against the
# environments' own templates and responses. Per the protocol, no task strategies or demonstrations.
BENCHMARK_RULES: Dict[str, str] = {
    "scienceworld": """Environment: ScienceWorld
Available commands (use exactly these forms; OBJ and LOC must be names exactly as they appear in observations):
- look around | look at OBJ | look in OBJ | inventory | task
- go to LOC: LOC must be a room directly connected to your current room, and the door to it must be open (use "open door to LOC" first)
- open OBJ | close OBJ: doors ("open door to LOC") and openable objects such as containers or appliances
- pick up OBJ: move OBJ into your inventory
- put down OBJ: move OBJ from your inventory to the current room
- move OBJ to OBJ: place the first object into or onto the second one (e.g., a container)
- pour OBJ in OBJ | mix OBJ | dunk OBJ in OBJ | eat OBJ | flush OBJ
- activate OBJ | deactivate OBJ | use OBJ on OBJ | read OBJ
- connect OBJ to OBJ | disconnect OBJ: for electrical components; name each part (e.g., a terminal) exactly as shown in observations
- wait: wait 10 time steps | wait1: wait 1 time step
- focus on OBJ: declares OBJ as the task target; focusing on a wrong object immediately ends the task as a failure
Note: "go to" only accepts room names, never objects. There is no "take" or "put ... in" command; use "pick up" and "move ... to".
Note: If an object is inside a container or several objects share a name, use the full name shown in the observation (e.g., "OBJ in CONTAINER").""",
    "textworld_express": """Environment: TextWorldExpress
Available commands in this game (use exactly these forms; no other commands exist; OBJ must be an object name exactly as it appears in observations; DIR is north, south, east or west):""",
}

# TextWorldExpress games expose different command sets (taken from each game's validActions).
TWX_GAME_COMMANDS: Dict[str, str] = {
    "coin": """- look around | inventory
- move DIR: a closed door blocks movement; open it first with "open door to DIR"
- open door to DIR | close door to DIR: only doors can be opened or closed in this game
- take OBJ""",
    "mapreader": """- look around | inventory | task
- move DIR
- take OBJ | put OBJ in OBJ
- read OBJ""",
    "cookingworld": """- look around | inventory | examine OBJ | read OBJ
- move DIR: a closed door blocks movement; open it first with "open door to DIR"
- open OBJ | close OBJ: doors ("open door to DIR") and containers such as a fridge or a drawer
- take OBJ | put OBJ in OBJ: "take" has no "from" clause
- chop OBJ | slice OBJ | dice OBJ: requires a knife in your inventory
- cook OBJ in stove | cook OBJ in oven | cook OBJ in barbeque
- prepare meal | eat meal""",
}


def build_system_prompt(benchmark: str, task_description: str, task_id: str = "") -> str:
    """Build identical ReAct system prompt + benchmark-specific command rules & goal."""
    rules = BENCHMARK_RULES.get(
        benchmark,
        f"Environment: {benchmark}\nUse only valid commands supported by the environment.",
    )
    if benchmark == "textworld_express":
        rules = f"{rules}\n" + TWX_GAME_COMMANDS.get(task_id, "\n".join(TWX_GAME_COMMANDS.values()))
    task_desc = (task_description or "").strip()
    return f"{SHARED_SYSTEM_HEADER}\n\n{rules}\n\nTask Goal:\n{task_desc}"


def format_user_turn(turn_id: int, observation: str) -> str:
    return f"Turn {turn_id}:\nObservation: {(observation or '').strip()}"


def format_assistant_turn(cot: str, action: str) -> str:
    return f"Thought: {cot.strip()}\nAction: {action.strip()}"


@dataclass
class ParsedReactOutput:
    parse_ok: bool
    parse_error_type: str               # "" when ok, otherwise specific reason
    cot: str
    action: str
    thought_label_end: Optional[int]    # char index immediately after "Thought:"
    cot_char_span: Optional[Tuple[int, int]]
    action_label_end: Optional[int]     # char index immediately after "Action:"
    action_char_span: Optional[Tuple[int, int]]


_REACT_FULL_RE = re.compile(
    r"(?P<thought_label>Thought:)[ \t]*(?P<cot>.*?)\r?\n[ \t]*(?P<action_label>Action:)[ \t]*(?P<action>[^\r\n]+)",
    re.DOTALL,
)
_ACTION_ONLY_RE = re.compile(r"(?P<action_label>Action:)[ \t]*(?P<action>[^\r\n]+)")
_THOUGHT_ONLY_RE = re.compile(r"(?P<thought_label>Thought:)[ \t]*(?P<cot>.*)", re.DOTALL)


def parse_react_output(raw_output: str) -> ParsedReactOutput:
    """Strictly parse 'Thought: ...\\nAction: ...' while recording exact character offsets."""
    text = raw_output or ""
    m = _REACT_FULL_RE.search(text)
    if m:
        cot_raw = m.group("cot")
        act_raw = m.group("action")
        cot_stripped = cot_raw.strip()
        act_stripped = act_raw.strip()

        # Trim trailing/leading whitespace from char spans while keeping inner offsets exact
        cot_start = m.start("cot") + (len(cot_raw) - len(cot_raw.lstrip()))
        cot_end = cot_start + len(cot_stripped)
        act_start = m.start("action") + (len(act_raw) - len(act_raw.lstrip()))
        act_end = act_start + len(act_stripped)

        if not cot_stripped:
            return ParsedReactOutput(
                parse_ok=False,
                parse_error_type="empty_cot",
                cot="",
                action=act_stripped,
                thought_label_end=m.end("thought_label"),
                cot_char_span=None,
                action_label_end=m.end("action_label"),
                action_char_span=(act_start, act_end) if act_stripped else None,
            )
        if not act_stripped:
            return ParsedReactOutput(
                parse_ok=False,
                parse_error_type="empty_action",
                cot=cot_stripped,
                action="",
                thought_label_end=m.end("thought_label"),
                cot_char_span=(cot_start, cot_end),
                action_label_end=m.end("action_label"),
                action_char_span=None,
            )
        return ParsedReactOutput(
            parse_ok=True,
            parse_error_type="",
            cot=cot_stripped,
            action=act_stripped,
            thought_label_end=m.end("thought_label"),
            cot_char_span=(cot_start, cot_end),
            action_label_end=m.end("action_label"),
            action_char_span=(act_start, act_end),
        )

    # Fallback diagnostics (never silently dropped; logged with exact failure reason)
    m_act = _ACTION_ONLY_RE.search(text)
    m_tho = _THOUGHT_ONLY_RE.search(text)
    if m_act and not m_tho:
        act_raw = m_act.group("action")
        act_stripped = act_raw.strip()
        act_start = m_act.start("action") + (len(act_raw) - len(act_raw.lstrip()))
        act_end = act_start + len(act_stripped)
        return ParsedReactOutput(
            parse_ok=False,
            parse_error_type="missing_thought",
            cot="",
            action=act_stripped,
            thought_label_end=None,
            cot_char_span=None,
            action_label_end=m_act.end("action_label"),
            action_char_span=(act_start, act_end) if act_stripped else None,
        )
    if m_tho and not m_act:
        cot_raw = m_tho.group("cot")
        cot_stripped = cot_raw.strip()
        cot_start = m_tho.start("cot") + (len(cot_raw) - len(cot_raw.lstrip()))
        cot_end = cot_start + len(cot_stripped)
        return ParsedReactOutput(
            parse_ok=False,
            parse_error_type="missing_action",
            cot=cot_stripped,
            action="",
            thought_label_end=m_tho.end("thought_label"),
            cot_char_span=(cot_start, cot_end) if cot_stripped else None,
            action_label_end=None,
            action_char_span=None,
        )
    return ParsedReactOutput(
        parse_ok=False,
        parse_error_type="format_broken",
        cot="",
        action="",
        thought_label_end=None,
        cot_char_span=None,
        action_label_end=None,
        action_char_span=None,
    )


def build_state_opd_messages(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Construct State-OPD teacher context s_t from student history h_t:
    Keep system prompt (task goal + rules), all historical/current user observations,
    and historical Actions ONLY — stripping all historical free-form Thought/CoT text.
    """
    pruned: List[Dict[str, str]] = []
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        if role == "assistant":
            parsed = parse_react_output(content)
            if parsed.action:
                pruned.append({"role": "assistant", "content": f"Action: {parsed.action}"})
            else:
                pruned.append({"role": "assistant", "content": "Action: [invalid]"})
        else:
            pruned.append({"role": role, "content": content})
    return pruned

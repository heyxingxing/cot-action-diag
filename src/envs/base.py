"""Abstract environment interface + realistic MockTextEnv for local E2E testing."""

from __future__ import annotations

import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from src.config import EpisodeSpec


_ARTICLE_RE = re.compile(r"\b(?:the|a|an)\b")


def normalize_action(action: str) -> str:
    """Canonical form for valid-list matching: the game parsers accept articles
    ("go to the kitchen") that the valid-action lists never contain."""
    return " ".join(_ARTICLE_RE.sub(" ", (action or "").lower()).split())


@dataclass
class ResetResult:
    observation: str
    task_description: str
    valid_actions: List[str]
    score: float
    score_norm: float
    reward_aux: float
    reward_source: str


@dataclass
class StepResult:
    observation: str
    reward_raw: float
    reward_aux: float
    reward_source: str
    score: float
    score_norm: float
    done: bool
    action_valid: bool
    terminal_success: bool
    terminal_failure: bool
    valid_actions: List[str]
    action_in_valid_list: bool = False   # strict exact-list match, kept alongside action_valid for analysis


class BaseTextEnv(ABC):
    """Thin unified adapter implemented identically for ScienceWorld & TextWorldExpress."""

    @abstractmethod
    def reset(self, spec: EpisodeSpec) -> ResetResult:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: str) -> StepResult:
        raise NotImplementedError

    def close(self) -> None:
        return None


class MockTextEnv(BaseTextEnv):
    """
    Deterministic multi-step synthetic text environment used for local dry-runs and unit tests.
    Mimics sparse subgoal rewards (mostly 0, +0.3/+0.4/+0.3 milestones), invalid actions,
    and occasional fatal failures so every E01-E18 plot & metric is exercised.
    """

    def __init__(self, benchmark: str = "scienceworld", env_step_limit: int = 35):
        self.benchmark = benchmark
        self.env_step_limit = env_step_limit
        self._step = 0
        self._stage = 0
        self._score = 0.0
        self._rng = random.Random(0)
        self._scale = 100.0 if benchmark == "scienceworld" else 1.0
        self._reward_source = (
            "scienceworld_subgoal_delta" if benchmark == "scienceworld" else "textworld_express_delta"
        )
        self._valid = [
            "look around",
            "inventory",
            "go to kitchen",
            "open fridge",
            "take target object",
            "use target object",
            "finish task",
        ]

    def reset(self, spec: EpisodeSpec) -> ResetResult:
        self._step = 0
        self._stage = 0
        self._score = 0.0
        self._rng = random.Random(spec.seed + (1000 if spec.benchmark == "textworld_express" else 0))
        obs = f"You are in the hallway. Task: {spec.task_id} (tier {spec.difficulty_tier}, seed {spec.seed})."
        desc = f"Complete the {spec.task_id} goal by locating and processing the target object."
        return ResetResult(
            observation=obs,
            task_description=desc,
            valid_actions=list(self._valid),
            score=0.0,
            score_norm=0.0,
            reward_aux=0.0,
            reward_source=self._reward_source,
        )

    def step(self, action: str) -> StepResult:
        self._step += 1
        act = (action or "").strip().lower()
        valid_set = {a.lower() for a in self._valid}
        action_valid = act in valid_set
        reward_norm = 0.0
        done = False
        success = False
        failure = False

        if not action_valid:
            obs = f"No known action matches '{action}'."
        elif act == "look around":
            obs = f"You look around at stage {self._stage}. The kitchen door is open."
        elif act == "go to kitchen" and self._stage == 0:
            self._stage = 1
            obs = "You enter the kitchen and see a fridge."
        elif act == "open fridge" and self._stage == 1:
            self._stage = 2
            reward_norm = 0.3
            obs = "You open the fridge. Inside is the target object."
        elif act == "take target object" and self._stage == 2:
            self._stage = 3
            reward_norm = 0.3
            obs = "You pick up the target object."
        elif act == "use target object" and self._stage == 3:
            self._stage = 4
            obs = "You prepare the target object."
        elif act == "finish task" and self._stage >= 4:
            self._stage = 5
            reward_norm = 0.4
            done = True
            success = True
            obs = "Task completed successfully!"
        elif act == "inventory":
            obs = f"Inventory check (stage={self._stage})."
        else:
            obs = f"You try '{act}', but nothing changes at stage {self._stage}."

        self._score = min(1.0, self._score + reward_norm)
        if not done and self._step >= self.env_step_limit:
            done = True
            failure = not success

        reward_raw = reward_norm * self._scale
        score_raw = self._score * self._scale
        return StepResult(
            observation=obs,
            reward_raw=reward_raw,
            reward_aux=score_raw,
            reward_source=self._reward_source,
            score=score_raw,
            score_norm=self._score,
            done=done,
            action_valid=action_valid,
            terminal_success=success,
            terminal_failure=failure,
            valid_actions=list(self._valid),
            action_in_valid_list=action_valid,
        )

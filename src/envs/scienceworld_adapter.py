"""ScienceWorld environment adapter (reuses a single Py4J JVM instance safely)."""

from __future__ import annotations

from typing import Any, List, Set

from src.config import EpisodeSpec
from src.envs.base import BaseTextEnv, ResetResult, StepResult, normalize_action

# Rejection / failed-precondition responses, extracted from the scienceworld 1.2.3 jar. The exact
# valid-combination list misses many surface forms the parser accepts (e.g. "pick up pea plant in
# flower pot 9", "put X in Y"), so validity means "the environment executed it" instead.
_SW_FAILURE_PATTERNS = (
    "no known action matches",
    "ambiguous request",
    "not clear how",
    "i'm not sure how",
    "i'm not sure what that means",
    "isn't open",
    "is not open",
    "is not openable",
    "is not closeable",
    "is not moveable",
    "is not movable",
    "is not a container",
    "is not edible",
    "is not flammable",
    "is not currently passable",
    "is not something that can be",
    "can't be moved there",
    "can't be poured out",
    "can't be connected to itself",
    "you can't",
    "doesn't appear to go anywhere",
    "appears broken",
    "already appears to be in the inventory",
    "so there's nothing to pour",
    "doesn't appear to have a temperature",
    "error:",
)


class ScienceWorldAdapter(BaseTextEnv):
    """
    Wrapper over AllenAI ScienceWorldEnv.
    Handles both snake_case and camelCase Py4J method names across package versions,
    and reuses a single ScienceWorldEnv JVM instance across episodes via env.load().
    """

    REWARD_SOURCE = "scienceworld_subgoal_delta"

    def __init__(self, env_step_limit: int = 35):
        from scienceworld import ScienceWorldEnv

        self.env_step_limit = env_step_limit
        self._env = ScienceWorldEnv("", envStepLimit=env_step_limit)
        self._valid_actions: List[str] = []
        self._valid_set: Set[str] = set()
        self._prev_score: float = 0.0

    def _call_compat(self, *names: str, default: Any = None) -> Any:
        for name in names:
            fn = getattr(self._env, name, None)
            if callable(fn):
                return fn()
        return default

    def _query_valid_actions(self) -> List[str]:
        raw = self._call_compat(
            "get_valid_action_object_combinations",
            "getValidActionObjectCombinations",
            default=[],
        )
        return [str(x).strip() for x in (raw or []) if str(x).strip()]

    def reset(self, spec: EpisodeSpec) -> ResetResult:
        self._env.load(
            spec.task_id,
            spec.seed,
            spec.simplification_str,
        )
        res = self._env.reset()
        if isinstance(res, tuple) and len(res) >= 2:
            obs, info = res[0], res[1]
        else:
            obs, info = res, {}

        task_desc = self._call_compat("get_task_description", "getTaskDescription", default="") or ""
        score = float((info or {}).get("score", 0.0))
        self._prev_score = score
        self._valid_actions = self._query_valid_actions()
        self._valid_set = {normalize_action(a) for a in self._valid_actions}

        obs_str = str(obs or "").strip()
        return ResetResult(
            observation=obs_str,
            task_description=str(task_desc).strip(),
            valid_actions=list(self._valid_actions),
            score=score,
            score_norm=max(score, 0.0) / 100.0,
            reward_aux=score,
            reward_source=self.REWARD_SOURCE,
        )

    def step(self, action: str) -> StepResult:
        act_str = (action or "").strip()
        obs, reward, done, info = self._env.step(act_str)
        obs_str = str(obs or "").strip()
        info = info or {}

        score = float(info.get("score", self._prev_score))
        reward_raw = float(reward)
        self._prev_score = score

        obs_lower = obs_str.lower()
        failed = any(p in obs_lower for p in _SW_FAILURE_PATTERNS)
        in_valid_list = normalize_action(act_str) in self._valid_set
        action_valid = bool(act_str) and not failed

        terminal_success = bool(score >= 100.0)
        # Only a genuine task failure (score -100); hitting envStepLimit is reported as max_turns upstream
        terminal_failure = bool(score < 0.0)

        self._valid_actions = self._query_valid_actions() if not done else []
        self._valid_set = {normalize_action(a) for a in self._valid_actions}

        return StepResult(
            observation=obs_str,
            reward_raw=reward_raw,
            reward_aux=score,
            reward_source=self.REWARD_SOURCE,
            score=score,
            score_norm=max(score, 0.0) / 100.0,
            done=bool(done),
            action_valid=action_valid,
            terminal_success=terminal_success,
            terminal_failure=terminal_failure,
            valid_actions=list(self._valid_actions),
            action_in_valid_list=in_valid_list,
        )

    def close(self) -> None:
        close_fn = getattr(self._env, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass

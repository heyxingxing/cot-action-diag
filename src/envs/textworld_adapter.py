"""TextWorldExpress environment adapter (reuses a single Py4J JVM instance safely)."""

from __future__ import annotations

from typing import List, Set

from src.config import EpisodeSpec
from src.envs.base import BaseTextEnv, ResetResult, StepResult, normalize_action


class TextWorldExpressAdapter(BaseTextEnv):
    """
    Wrapper over TextWorldExpressEnv supporting coin / mapreader / cookingworld.
    Uses gameFold='dev' by default so train/test folds remain clean for future RL.
    """

    REWARD_SOURCE = "textworld_express_delta"

    def __init__(self, env_step_limit: int = 35):
        from textworld_express import TextWorldExpressEnv

        self.env_step_limit = env_step_limit
        self._env = TextWorldExpressEnv(envStepLimit=env_step_limit)
        self._valid_actions: List[str] = []
        self._valid_set: Set[str] = set()

    def reset(self, spec: EpisodeSpec) -> ResetResult:
        game_params = spec.game_params or ""
        self._env.load(gameName=spec.task_id, gameParams=game_params)
        obs, infos = self._env.reset(
            seed=spec.seed,
            gameFold=spec.game_fold,
            generateGoldPath=False,
        )
        infos = infos or {}
        self._valid_actions = [str(a).strip() for a in (infos.get("validActions") or []) if str(a).strip()]
        self._valid_set = {normalize_action(a) for a in self._valid_actions}

        task_desc = str(infos.get("taskDescription", "")).strip()
        obs_str = str(obs or "").strip()
        if not task_desc:
            task_desc = obs_str.splitlines()[0] if obs_str else f"Complete the {spec.task_id} task."

        score = float(infos.get("score", 0.0))
        score_raw = float(infos.get("scoreRaw", score))

        return ResetResult(
            observation=obs_str,
            task_description=task_desc,
            valid_actions=list(self._valid_actions),
            score=score,
            score_norm=max(min(score, 1.0), 0.0),
            reward_aux=score_raw,
            reward_source=self.REWARD_SOURCE,
        )

    def step(self, action: str) -> StepResult:
        # TWX's parser is case-sensitive: observations say "To the North ..." so the model often writes
        # "open door to North", yet some object names keep capitals ("dirty purple T-shirt"). Send the
        # valid action's exact spelling on a case-insensitive match, otherwise the lowercased action.
        act_str = " ".join((action or "").lower().split())
        act_str = {a.lower(): a for a in self._valid_actions}.get(act_str, act_str)
        obs, reward, done, infos = self._env.step(act_str)
        obs_str = str(obs or "").strip()
        infos = infos or {}

        obs_lower = obs_str.lower()
        parser_rejected = (
            "not a valid action" in obs_lower
            or "don't understand" in obs_lower
            or "i'm not sure what you mean" in obs_lower
            or "cannot perform" in obs_lower
        )
        in_valid_list = normalize_action(act_str) in self._valid_set
        if self._valid_set:
            action_valid = in_valid_list and (not parser_rejected)
        else:
            action_valid = bool(act_str) and (not parser_rejected)

        score = float(infos.get("score", 0.0))
        score_raw = float(infos.get("scoreRaw", score))
        reward_raw = float(reward)
        terminal_success = bool(infos.get("tasksuccess", False))
        # Only a genuine task failure; hitting envStepLimit is reported as max_turns upstream
        terminal_failure = bool(infos.get("taskfailure", False))

        self._valid_actions = [str(a).strip() for a in (infos.get("validActions") or []) if str(a).strip()]
        self._valid_set = {normalize_action(a) for a in self._valid_actions}

        return StepResult(
            observation=obs_str,
            reward_raw=reward_raw,
            reward_aux=score_raw,
            reward_source=self.REWARD_SOURCE,
            score=score,
            score_norm=max(min(score, 1.0), 0.0),
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

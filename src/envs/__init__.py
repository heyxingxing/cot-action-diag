"""Environment factory for ScienceWorld, TextWorldExpress, and local MockTextEnv."""

from __future__ import annotations

from src.envs.base import BaseTextEnv, MockTextEnv


def create_env(benchmark: str, env_step_limit: int = 35, use_mock: bool = False) -> BaseTextEnv:
    if use_mock:
        return MockTextEnv(benchmark=benchmark, env_step_limit=env_step_limit)
    if benchmark == "scienceworld":
        from src.envs.scienceworld_adapter import ScienceWorldAdapter

        return ScienceWorldAdapter(env_step_limit=env_step_limit)
    if benchmark == "textworld_express":
        from src.envs.textworld_adapter import TextWorldExpressAdapter

        return TextWorldExpressAdapter(env_step_limit=env_step_limit)
    raise ValueError(f"Unsupported benchmark: {benchmark}")

"""Configuration loading and episode schedule helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class EpisodeSpec:
    benchmark: str
    task_id: str
    difficulty_tier: str
    seed: int
    episode_id: str
    game_params: Optional[str] = None
    simplification_str: str = ""
    game_fold: str = "dev"
    env_step_limit: int = 35


def load_protocol(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Protocol config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _validate_protocol(cfg)
    return cfg


def _validate_protocol(cfg: Dict[str, Any]) -> None:
    required_top = ["experiment", "model", "rollout", "offline_scoring", "metrics", "benchmarks"]
    for key in required_top:
        if key not in cfg:
            raise ValueError(f"Missing top-level key '{key}' in protocol.yaml")
    if float(cfg["rollout"]["temperature"]) <= 0.0:
        raise ValueError(
            "rollout.temperature must be > 0 for Qwen3-8B (greedy decoding degrades quality and loops)."
        )
    if float(cfg["offline_scoring"]["scoring_temperature"]) != 1.0:
        raise ValueError("offline_scoring.scoring_temperature must be strictly 1.0 for unbiased logp/entropy.")


def get_episode_specs(
    cfg: Dict[str, Any],
    benchmark: str,
    mode: str = "full",
    max_episodes: Optional[int] = None,
) -> List[EpisodeSpec]:
    """Build ordered list of EpisodeSpec for 'smoke' (E01) or 'full' (E02)."""
    if benchmark not in cfg["benchmarks"]:
        raise ValueError(f"Unknown benchmark '{benchmark}'. Choose from: {list(cfg['benchmarks'].keys())}")
    bcfg = cfg["benchmarks"][benchmark]
    step_limit = int(bcfg.get("env_step_limit", cfg["rollout"]["max_turns_per_episode"]))
    simpl = str(bcfg.get("simplification_str", ""))
    fold = str(bcfg.get("game_fold", "dev"))

    specs: List[EpisodeSpec] = []
    if mode == "smoke":
        for item in bcfg["smoke_episodes"]:
            task_id = str(item["task_id"])
            seed = int(item["seed"])
            specs.append(
                EpisodeSpec(
                    benchmark=benchmark,
                    task_id=task_id,
                    difficulty_tier=str(item["difficulty_tier"]),
                    seed=seed,
                    episode_id=f"{benchmark}__{task_id}__s{seed}",
                    game_params=item.get("game_params"),
                    simplification_str=simpl,
                    game_fold=fold,
                    env_step_limit=step_limit,
                )
            )
    elif mode == "full":
        for task in bcfg["tasks"]:
            task_id = str(task["task_id"])
            tier = str(task["difficulty_tier"])
            gparams = task.get("game_params")
            for seed in task["seeds"]:
                seed_i = int(seed)
                specs.append(
                    EpisodeSpec(
                        benchmark=benchmark,
                        task_id=task_id,
                        difficulty_tier=tier,
                        seed=seed_i,
                        episode_id=f"{benchmark}__{task_id}__s{seed_i}",
                        game_params=gparams,
                        simplification_str=simpl,
                        game_fold=fold,
                        env_step_limit=step_limit,
                    )
                )
    else:
        raise ValueError(f"Unknown mode '{mode}'. Expected 'smoke' or 'full'.")

    if max_episodes is not None:
        specs = specs[:max_episodes]
    return specs

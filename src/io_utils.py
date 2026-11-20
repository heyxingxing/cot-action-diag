"""JSONL I/O utilities, resume helpers, and HuggingFace revision resolver."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def get_completed_episode_ids(episodes_jsonl: str | Path) -> Set[str]:
    return {str(r["episode_id"]) for r in load_jsonl(episodes_jsonl) if "episode_id" in r}


def resolve_hf_revision(model_id: str, configured_revision: Optional[str]) -> str:
    """Return pinned commit hash if available, otherwise try local HF cache metadata."""
    if configured_revision:
        return str(configured_revision)
    try:
        from huggingface_hub import model_info

        info = model_info(model_id)
        if getattr(info, "sha", None):
            return str(info.sha)
    except Exception:
        pass
    return "local_or_unresolved"

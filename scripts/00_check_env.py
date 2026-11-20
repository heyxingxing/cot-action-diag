#!/usr/bin/env python3
"""
Step 0 服务器一键体检脚本：
在跑 E01 之前检查 GPU、Java、PyTorch、transformers>=4.51、vLLM、ScienceWorld、TextWorldExpress。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ok(msg: str) -> None:
    print(f"[OK]   {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check GPU, Java, Python deps, and both benchmarks.")
    parser.add_argument("--check-hf-model", action="store_true", help="Also verify loading Qwen/Qwen3-8B tokenizer.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "protocol.yaml"))
    args = parser.parse_args()

    print("=" * 72)
    print("CoT -> Action Diagnostic Environment Check (Qwen/Qwen3-8B)")
    print("=" * 72)
    _ok(f"Python: {sys.version.split()[0]}")

    # 1. Java (required by ScienceWorld and TextWorldExpress)
    java_bin = shutil.which("java")
    if java_bin:
        out = subprocess.run([java_bin, "-version"], capture_output=True, text=True)
        first_line = (out.stderr or out.stdout).strip().splitlines()[0] if (out.stderr or out.stdout) else "unknown"
        _ok(f"Java: {first_line}")
    else:
        _fail("Java not found! Install via: sudo apt-get install -y openjdk-11-jdk")

    # 2. PyTorch & GPU
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            _ok(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda} | GPU 0: {name} ({vram_gb:.1f} GB)")
            if vram_gb < 38:
                _warn("GPU VRAM < 40GB: Qwen3-8B full-vocab offline scoring needs batch_size_turns=1~2.")
        else:
            _warn(f"PyTorch {torch.__version__} | No CUDA GPU detected (OK for local CPU analysis/mock tests).")
    except Exception as e:
        _fail(f"PyTorch import error: {e}")

    # 3. transformers >= 4.51.0
    try:
        import transformers
        from packaging.version import Version

        ver = transformers.__version__
        if Version(ver) >= Version("4.51.0"):
            _ok(f"transformers {ver} (>= 4.51.0, Qwen3 native support ready)")
        else:
            _fail(f"transformers {ver} is too old for Qwen3! Upgrade to >= 4.51.0")
    except Exception as e:
        _fail(f"transformers check failed: {e}")

    # 4. vLLM & FlashAttention
    try:
        import vllm

        _ok(f"vLLM {vllm.__version__}")
    except Exception as e:
        _warn(f"vLLM not available ({e}); Stage 1 can still fallback to --backend hf (slower).")

    try:
        import flash_attn

        _ok(f"flash_attn {flash_attn.__version__}")
    except Exception:
        _warn("flash_attn not installed; Stage 2 will automatically fallback to PyTorch SDPA.")

    # 5. Data & stats packages
    for pkg in ["yaml", "pandas", "pyarrow", "scipy", "sklearn", "matplotlib", "seaborn"]:
        try:
            __import__(pkg)
            _ok(f"Package '{pkg}' installed")
        except Exception as e:
            _fail(f"Missing package '{pkg}': {e}")

    # 6. Benchmark smoke import + reset
    try:
        from scienceworld import ScienceWorldEnv

        env = ScienceWorldEnv("", envStepLimit=5)
        env.load("find-living-thing", 0, "")
        env.reset()
        if hasattr(env, "close"):
            env.close()
        _ok("ScienceWorld (scienceworld) reset smoke test passed!")
    except Exception as e:
        _warn(f"ScienceWorld check failed: {e}")

    try:
        from textworld_express import TextWorldExpressEnv

        env = TextWorldExpressEnv(envStepLimit=5)
        env.load(gameName="coin", gameParams="numLocations=5,includeDoors=1")
        obs, infos = env.reset(seed=1000, gameFold="dev", generateGoldPath=False)
        if hasattr(env, "close"):
            env.close()
        _ok(f"TextWorldExpress (textworld-express) reset smoke test passed! ({len(infos.get('validActions', []))} valid actions)")
    except Exception as e:
        _warn(f"TextWorldExpress check failed: {e}")

    # 7. Optional HF Qwen3-8B tokenizer check
    if args.check_hf_model:
        try:
            from transformers import AutoTokenizer
            from src.config import load_protocol

            cfg = load_protocol(args.config)
            model_id = cfg["model"]["hf_id"]
            tok = AutoTokenizer.from_pretrained(model_id)
            rendered = tok.apply_chat_template(
                [{"role": "user", "content": "hi"}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            _ok(f"Tokenizer '{model_id}' loaded; vocab={len(tok)}; enable_thinking=False template OK ({len(rendered)} chars)")
        except Exception as e:
            _fail(f"Tokenizer check failed: {e}")


if __name__ == "__main__":
    main()

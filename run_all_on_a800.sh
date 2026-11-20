#!/usr/bin/env bash
# 一键在单张 A800 (80GB) 上依次刷完 ScienceWorld 和 TextWorldExpress 的正式诊断实验 (E02-E18)
set -euo pipefail

cd "$(dirname "$0")"

# 自动激活服务器预装的 py312 环境（若存在）
if [ -f "/usr/local/miniconda3/bin/activate" ]; then
  # shellcheck disable=SC1091
  source /usr/local/miniconda3/bin/activate py312
fi

# 若本地共享盘已有 Qwen3-8B 权重，自动用 protocol_a800.yaml 跳过下载
CONFIG_PATH="configs/protocol.yaml"
if [ -d "/model/ModelScope/Qwen/Qwen3-8B" ] && [ -f "configs/protocol_a800.yaml" ]; then
  CONFIG_PATH="configs/protocol_a800.yaml"
fi
echo "[Info] Using config: ${CONFIG_PATH}"

echo "========================================================================"
echo "[1/7] ScienceWorld: Stage 1 Rollout (60 episodes, vLLM + prefix caching)"
echo "========================================================================"
python scripts/01_run_rollout.py \
  --config "${CONFIG_PATH}" \
  --benchmark scienceworld \
  --mode full \
  --resume

echo "========================================================================"
echo "[2/7] ScienceWorld: Stage 2 Offline 4-Pass Scoring (HF exact entropy)"
echo "========================================================================"
python scripts/02_score_offline.py \
  --config "${CONFIG_PATH}" \
  --data-dir data/scienceworld_full

echo "========================================================================"
echo "[3/7] ScienceWorld: Stage 3 Analysis & 10 Figures"
echo "========================================================================"
python scripts/03_analyze_and_plot.py \
  --config "${CONFIG_PATH}" \
  --data-dir data/scienceworld_full \
  --out-dir results/scienceworld_full

echo "========================================================================"
echo "[4/7] TextWorldExpress: Stage 1 Rollout (60 episodes, vLLM)"
echo "========================================================================"
python scripts/01_run_rollout.py \
  --config "${CONFIG_PATH}" \
  --benchmark textworld_express \
  --mode full \
  --resume

echo "========================================================================"
echo "[5/7] TextWorldExpress: Stage 2 Offline 4-Pass Scoring (HF exact entropy)"
echo "========================================================================"
python scripts/02_score_offline.py \
  --config "${CONFIG_PATH}" \
  --data-dir data/textworld_express_full

echo "========================================================================"
echo "[6/7] TextWorldExpress: Stage 3 Analysis & 10 Figures"
echo "========================================================================"
python scripts/03_analyze_and_plot.py \
  --config "${CONFIG_PATH}" \
  --data-dir data/textworld_express_full \
  --out-dir results/textworld_express_full

echo "========================================================================"
echo "[7/7] Cross-Benchmark Side-by-Side Summary + Export E14 Blind CSVs"
echo "========================================================================"
python scripts/05_compare_benchmarks.py \
  --sw-results results/scienceworld_full \
  --tw-results results/textworld_express_full \
  --out-dir results/cross_benchmark

python scripts/04_e14_audit.py sample \
  --data-dir data/scienceworld_full \
  --out-csv results/scienceworld_full/e14_blind_to_label.csv \
  --n-samples 80

python scripts/04_e14_audit.py sample \
  --data-dir data/textworld_express_full \
  --out-csv results/textworld_express_full/e14_blind_to_label.csv \
  --n-samples 80

echo "ALL DONE! Results are in results/scienceworld_full, results/textworld_express_full, and results/cross_benchmark."

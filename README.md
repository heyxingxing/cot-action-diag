# `cot_action_diag` — Qwen/Qwen3-8B 双 Benchmark 一键诊断实验框架

一套代码同时跑完 **ScienceWorld** 和 **TextWorldExpress** 的全部诊断实验（**E01–E18**），专为 **1× A800（80GB）** 设计。

---

## 仓库导览（队友先看这里）

**建议阅读顺序**
1. [`PROGRESS.md`](PROGRESS.md)：当前进度、配置、全量结果、按图结论、对 w = f(C, H) 的初步回答、已知局限和待办。**最新结论以这里为准。**
2. `results/experiment_report.html`：交互式总报告（单文件，下载后用浏览器打开）。部分解读有过度推断，核对结果见 `PROGRESS.md` §8。
3. `results/{scienceworld,textworld_express}_full/figures/`、`tables/`：两个 Bench 各 10 张图和统计表；`results/cross_benchmark/`：跨 Bench 对照。
4. [`docs/experiment_plan_qwen3_8b.md`](docs/experiment_plan_qwen3_8b.md)：实验手册（E01–E18 定义、指标公式、零样本规则）；`docs/cot_action_overview_same_tasks_v2.pdf`：诊断任务设计总览。

**目录**

| 路径 | 内容 |
| --- | --- |
| `src/` | Prompt 构造、tokenization 对齐、两个环境的适配器 |
| `scripts/00-06` | 环境体检 → rollout（vLLM）→ 离线 4-Pass 打分（HF）→ 分析画图 → E14 盲标 → 跨 Bench 对照 → HTML 报告 |
| `configs/` | `protocol.yaml`（通用）、`protocol_a800.yaml`（服务器本地模型路径） |
| `data/*_full/` | 全量数据：`episodes.jsonl`（每局）、`turns.jsonl`（每步原始文本和 token ids）、`turn_metrics.jsonl`（每步指标）、`token_stats.parquet`（逐 token logp / 熵 / D） |
| `data/*_smoke*/` | E01 试水：`_smoke` 为 v2（接口说明书 prompt），`_smoke_v1` 为旧 prompt |
| `results/` | 图、表、报告 |
| `logs/` | 全量运行和 E01 v2 的终端日志 |
| `tests/` | 本地无 GPU 端到端自测 |

**读数据的注意事项**
- 大多数分析只需要 `turn_metrics.jsonl`（pandas `read_json(..., lines=True)` 即可）。
- `token_stats.parquet` 由 pyarrow 26 写出，pyarrow 19 读会报 `Repetition level histogram size mismatch`，请用较新版本的 pyarrow。`logp_teacher_strong` 列全为 NaN（32B Teacher 没跑）。
- `results/` 里所有图表都包含“打转步”（重复上一步的不合法动作），去掉后的对照数字见 `PROGRESS.md` §5.4。

---

## 零、当前进度与交接速查（给接手执行的 Agent）

- **GPU 服务器**：免密 SSH 别名 **`ssh a800`**（地址不入库，向项目负责人索取；1× NVIDIA A800-SXM4-80GB，16C/236GB RAM，170GB 可用盘）
- **远程代码路径**：`/root/cot_action_diag/`（与本目录保持同步）
- **Python 环境**：`source /usr/local/miniconda3/bin/activate py312`（Python 3.12.11, `torch 2.11.0+cu130`, `vllm 0.25.1`, `transformers 5.14.1`, `openjdk-11-jdk`, `scienceworld 1.2.3`, `textworld-express 1.1.0` —— **全部已装好且 `00_check_env.py` 全绿通过！不要重装 `torch`/`vllm`/`transformers`**）
- **本地共享盘预下载模型**：`/model/ModelScope/Qwen/Qwen3-8B`（及 `/model/ModelScope/Qwen/Qwen3-32B`），**运行命令时请传 `--config configs/protocol_a800.yaml`**（零下载耗时）
- **当前状态（2026-11-20）**：E01 试水和两个 Bench 的全量实验都已跑完，`data/`、`results/` 已拉回本地。配置改动、核心结果、结论和待办见 **[`PROGRESS.md`](PROGRESS.md)**。

---

## 一、上传到 A800 服务器后的极速上手（照抄命令就行）

### Step 0：装环境 & 一键体检
```bash
# 1. 装 Java（ScienceWorld 和 TextWorldExpress 底层都要用到！）
sudo apt-get update && sudo apt-get install -y openjdk-11-jdk

# 2. 建环境并装依赖
conda create -n cot_diag python=3.10 -y
conda activate cot_diag
pip install "torch>=2.4.0" --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install flash-attn --no-build-isolation   # 若编译慢可跳过，代码会自动用 PyTorch SDPA

# 3. 体检（顺便测一下 HuggingFace 能否拉到 Qwen/Qwen3-8B）
python scripts/00_check_env.py --check-hf-model
```
*(若国内服务器下载慢，可先 `export HF_ENDPOINT=https://hf-mirror.com`)*

---

### Step 1：先跑 E01 试水（各 5 局，10 分钟内验完 4 条过关门槛）
```bash
source /usr/local/miniconda3/bin/activate py312
cd /root/cot_action_diag

# ScienceWorld 5 局试水 + 离线打分自检
python scripts/01_run_rollout.py --config configs/protocol_a800.yaml --benchmark scienceworld --mode smoke
python scripts/02_score_offline.py --config configs/protocol_a800.yaml --data-dir data/scienceworld_smoke

# TextWorldExpress 5 局试水 + 离线打分自检
python scripts/01_run_rollout.py --config configs/protocol_a800.yaml --benchmark textworld_express --mode smoke
python scripts/02_score_offline.py --config configs/protocol_a800.yaml --data-dir data/textworld_express_smoke
```
**检查终端打印的 Summary**：
- `Parse OK Rate >= 90%`、`Action Valid Rate >= 60%`
- 3 类任务不是全 0 也不是全 100（若某类任务太难或太易，改 `configs/protocol_a800.yaml` 里的任务名或 `numLocations` 后重跑试水）
- Stage 2 打印 `Alignment assertions passed!` 且 `I_empty_norm / H_C_mean` 正常无 NaN。

---

### Step 2：正式梭哈两个 Bench（各 60 局采集 + 离线 4-Pass 打分 + 自动出 10 张图）
直接跑现成的一键脚本（支持断线 `--resume` 续跑，两套总 GPU 耗时约 **1.5–2 小时**）：
```bash
bash run_all_on_a800.sh
```
或者手动逐条运行：
```bash
# === Benchmark A: ScienceWorld (60 局) ===
python scripts/01_run_rollout.py --benchmark scienceworld --mode full --resume
python scripts/02_score_offline.py --data-dir data/scienceworld_full
python scripts/03_analyze_and_plot.py \
  --data-dir data/scienceworld_full \
  --out-dir results/scienceworld_full

# === Benchmark B: TextWorldExpress (60 局) ===
python scripts/01_run_rollout.py --benchmark textworld_express --mode full --resume
python scripts/02_score_offline.py --data-dir data/textworld_express_full
python scripts/03_analyze_and_plot.py \
  --data-dir data/textworld_express_full \
  --out-dir results/textworld_express_full

# === 跨 Benchmark 一键汇总对照 ===
python scripts/05_compare_benchmarks.py \
  --sw-results results/scienceworld_full \
  --tw-results results/textworld_express_full \
  --out-dir results/cross_benchmark
```
*(可选：如果想把 E18 顺手再补一个 `Qwen/Qwen3-32B` 强 Teacher，给 `02_score_offline.py` 加上 `--run-strong-teacher` 即可，单张 80GB A800 可以直接跑。)*

---

### Step 3：跑完马上关 A800 省钱！拿数据回本地做 E14 人工抽检
把 `data/` 和 `results/`（总共不到 100 MB）打包下载回本地电脑，关掉租的卡：
```bash
# 1. 导出盲标 CSV（隐藏了 I_t 和熵，防止先入为主），每边 80 条
python scripts/04_e14_audit.py sample \
  --data-dir data/scienceworld_full \
  --out-csv results/scienceworld_full/e14_blind_to_label.csv --n-samples 80

python scripts/04_e14_audit.py sample \
  --data-dir data/textworld_express_full \
  --out-csv results/textworld_express_full/e14_blind_to_label.csv --n-samples 80

# 2. 用 Excel/Numbers 填完 3 个 0/1 列后，一键算 AUROC 和画图：
python scripts/04_e14_audit.py evaluate \
  --data-dir data/scienceworld_full \
  --labeled-csv results/scienceworld_full/e14_blind_to_label.csv \
  --out-dir results/scienceworld_full
```

---

## 二、本地无 GPU 端到端自测（发到服务器前先验证代码零 bug）
```bash
python tests/test_pipeline_e2e.py
```

---

## 三、产出文件一览
- `data/{benchmark}_full/`：`episodes.jsonl`、`turns.jsonl`、`turn_metrics.jsonl`、`token_stats.parquet`
- `results/{benchmark}_full/figures/`：`Fig01_E04_bridge_dist.png` ~ `Fig10_E18_teacher_gap.png`（10 张 300 DPI 标准图）
- `results/{benchmark}_full/tables/`：`summary_stats.csv`、`e09_correlations.csv`、`e10_quadrants.csv`、`e10_leak_free_cv.csv`、`e11_zero_reward.csv`、`e16_controlled_regression.csv`
- `results/{benchmark}_full/report_summary.md`：自动填好数字和异常案例的中文总结报告
- `results/cross_benchmark/`：`cross_benchmark_comparison.md` + `Fig_cross_benchmark_overview.png`

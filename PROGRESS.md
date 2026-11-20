# 进度记录（截至 2026-11-20）

**一句话状态**：两个 Bench（ScienceWorld / TextWorldExpress）各 60 局全量实验已在 a800 跑完（exit 0），`data/` 和 `results/` 已拉回本地。对“w = f(C, H) 怎么设计”已有初步回答（§7）。交互式报告已重写为 `index.html`，在线地址 https://heyxingxing.github.io/cot-action-diag/ 。还没做的：E14 人工盲标；是否把“去掉打转步”的对照并进分析脚本，尚未决定。

> 数值来源说明：标“表”的数字直接取自 `results/*/tables/*.csv` 或 `report_summary.md`；标“图”的是看图读出的趋势，没有单独算数；§7 的分组、报错后转移表（含聚类 bootstrap CI）和 token 级统计已写进 `scripts/06_build_html_report.py`，运行时打印并写入报告；§5.4 去掉打转步后的相关系数和 §8 的部分核对数字仍来自**未入库的临时脚本**，复现时需按文中定义重算。

---

## 1. 运行环境与操作要点

- **服务器**：`ssh a800`（地址不入库，向项目负责人索取），1× A800-80GB，16 核，236GB 内存。
- **环境**：`source /usr/local/miniconda3/bin/activate py312`。**不要重装 torch / vllm / transformers**（torch 2.11.0+cu130、vllm 0.25.1、transformers 5.14.1）。
- **代码**：远程 `/root/cot_action_diag/`，与本地本目录同步。
- **模型**：`/model/ModelScope/Qwen/Qwen3-8B`。所有命令带 `--config configs/protocol_a800.yaml`（`run_all_on_a800.sh` 会自动选这个配置）。
- **操作上的坑**：
  - 服务器没有 rsync 和 tmux。同步文件用 tar 走 ssh；长任务用 `nohup setsid ... &`。
  - 通过 `ssh a800 'python3 -c "..."'` 传带 f-string 的 Python 会被引号拆坏。把脚本写成文件，再 `ssh a800 'python3 - 参数' < 脚本文件`。
  - macOS 本地没有 `timeout` 命令。
  - ScienceWorld 要取 gold path，必须在 `load()` 时传 `generateGoldPath=True`。

## 2. 实验配置（`configs/protocol_a800.yaml`）

| 项 | 设置 |
| --- | --- |
| 模型 | Qwen3-8B，bf16，`enable_thinking: false`，显式 ReAct（`Thought:` / `Action:`） |
| Prompt | 零样本：只给“接口说明书”（命令格式、前置条件、效果），不放示范轨迹，不附每步合法动作列表 |
| 采样 | temperature 0.7，top_p 0.8，top_k 20，每步最多 256 token |
| 每局 | 最多 35 步；上下文上限 12288 token（实际最长 prompt 约 4.9K，没有截断） |
| 并发 | `num_parallel_envs: 60`（一个 Bench 的 60 局同时跑），vLLM 用 spawn 方式启动 worker，开 prefix caching |
| 离线打分 | HF 4-Pass。Pass 1-3 共享 prompt 的 KV cache，Pass 4（self-teacher）bs=1；熵用 fp32 在全词表上精确计算 |
| SW 任务 | find-living-thing (L1)、chemistry-mix-paint-secondary-color (L2)、test-conductivity (L3)，每类 seed 0-19 |
| TWX 任务 | coin (L1, `numLocations=7,includeDoors=1`)、mapreader (L2, `numLocations=11,includeDoors=1`)、cookingworld (L3, `numLocations=5,numIngredients=3,includeDoors=1`)，dev fold，每类 seed 1000-1019 |
| 未开启 | Qwen3-32B strong teacher（`--run-strong-teacher`）没跑 |

## 3. 本轮对代码和方案的改动

- **`src/prompts.py`**：`BENCHMARK_RULES` 改为纯接口说明书；新增 `TWX_GAME_COMMANDS`，按 coin / mapreader / cookingworld 分别列出该游戏的命令。命令都以环境自带模板和实测回显为准。删掉了环境里根本不存在的 fry / roast / grill。
  - 不写示范轨迹的原因：方案第 176 行禁止“给某一边偷加解题技巧或示例”，加了就算作弊。
- **`scripts/01_run_rollout.py`**：调用 `build_system_prompt` 时把 `spec.task_id` 传进去，TWX 才能按游戏选命令表。
- **`src/envs/textworld_adapter.py`**：TWX 的解析器区分大小写，模型常写 `open door to North`，而有些物体名本身带大写（`dirty purple T-shirt`）。现在先在合法动作里做不区分大小写的匹配，命中就发送合法动作的原始拼写，否则发送小写后的动作。
- **`src/envs/scienceworld_adapter.py`**（更早一轮的修复）：SW 的 `action_valid` 改为“执行后没有出现失败回显”（`_SW_FAILURE_PATTERNS`）；原来的严格列表匹配保留在 `action_in_valid_list`。
- **两个 config**（`protocol.yaml` 和 `protocol_a800.yaml` 改动相同）：
  - SW 的 L2 任务从 melt 换成 chemistry-mix-paint-secondary-color。melt 的 gold path 中位数是 34 步，20 个 variation 里只有 4 个不超过 25 步，35 步上限内几乎做不完。
  - mapreader 改为 11 个地点。7 个地点生成不了默认的 `maxDistanceApart=4`（dev seed 1000 会直接崩），11 个地点 20 个 dev seed 都能生成，gold path 7-13 步。
- **`docs/experiment_plan_qwen3_8b.md`**（实验手册）：第 176-178 行写明零样本规则和两边的命令表；L2 任务替换及原因；SW 和 TWX 的 `action_valid` 口径；mapreader 参数；task_id 示例。
- **没改**：02 / 03 / 04 / 05 脚本和测试。本地 `python3 tests/test_pipeline_e2e.py` 通过。

## 4. E01 试水（每个 Bench 5 局）

| 版本 | parse | action_valid | 重复上一步动作 | 成功 |
| --- | --- | --- | --- | --- |
| SW v1（旧 prompt） | 100% | 50.3% | 19.4% | 0/5 |
| SW v2（接口说明书） | 100% | 28.0% | 52.6% | 0/5 |
| TWX v1 | 100% | 52.1% | 16.2% | 3/5 |
| TWX v2 | 100% | 53.1% | 37.1% | 2/5 |

- 新 prompt 没有提高合法率，反而让“失败后原样重试”变多了。
- 硬门槛 `action_valid >= 60%` 两边都没达到。用户决定照常跑全量。

## 5. 全量结果

### 5.1 基础体检（表）

| 指标 | ScienceWorld | TextWorldExpress |
| --- | --- | --- |
| 局数 / 总步数 | 60 / 1924（平均 32.1 步） | 60 / 1375（平均 22.9 步） |
| parse_ok | 100% | 100% |
| action_valid | 42.5% | 59.9% |
| 终局成功率 | 13.3% | 50.0% |
| 平均终局得分（归一化） | 0.425 | 0.544 |
| 结束原因 | 成功 8 / 失败 4 / 步数用完 48 | 成功 30 / 失败 2 / 步数用完 28 |
| 单步奖励 正 / 零 / 负 | 7.8% / 92.0% / 0.2% | 5.2% / 94.8% / 0% |
| 数据对齐 / NaN | 3848/3848 通过，NaN 0 | 2750/2750 通过，NaN 0 |
| 截断 | 生成没碰到 256 上限，上下文没截断 | 同左 |

### 5.2 按任务

| Bench | 任务 | 成功 | 平均得分 |
| --- | --- | --- | --- |
| SW | find-living-thing | 8/20 | 0.62 |
| SW | chemistry-mix-paint-secondary-color | 0/20 | 0.17 |
| SW | test-conductivity | 0/20 | 0.48 |
| TWX | coin | 10/20 | 0.50 |
| TWX | mapreader | 20/20（饱和） | 1.00 |
| TWX | cookingworld | 0/20 | 0.13 |

所有用到“终局成功”的分析，信息几乎全来自 SW find-living-thing 和 TWX coin。

### 5.3 核心指标分布（表）

| 指标 | SW 均值 / P50 / P90 | TWX 均值 / P50 / P90 |
| --- | --- | --- |
| I_empty_norm | 0.840 / 0.005 / 2.94 | 1.290 / 0.027 / 4.53 |
| I_neutral_norm | 0.940 / 0.146 / 2.84 | 1.573 / 0.308 / 4.94 |
| I_fmt_delta | -0.101 / -0.016 / 0.38 | -0.283 / 0.000 / 0.59 |
| H_C_mean（nats） | 0.153 / 0.128 / 0.31 | 0.163 / 0.139 / 0.32 |
| H_A_mean（nats） | 0.015 / 0 / 0.04（57% 的步为 0） | 0.026 / 0 / 0.08（54% 的步为 0） |
| D_self_mean | -1.12 / -1.04 / -0.36（98% 为负） | -1.55 / -1.38 / -0.41（98% 为负） |
| CoT 长度（token） | 29.9 / 29 / 40 | 29.5 / 29 / 40 |

### 5.4 “打转步”分层

- **定义**：这一步的动作和上一步完全相同，而且上一步不合法。
- 占比：SW 33.5%（645 步），TWX 14.6%（201 步）。打转步全部不合法，I_empty 中位数约 0（0.002 / 0.011）。
- 去掉打转步后，合法率 SW 64.0%、TWX 70.1%。
- 去掉打转步后，主要负相关不变，甚至更强：

| Spearman | SW 全部 | SW 去打转 | TWX 全部 | TWX 去打转 |
| --- | --- | --- | --- | --- |
| I_empty vs reward | -0.217 | -0.288 | -0.268 | -0.281 |
| I_neutral vs reward | -0.231 | -0.252 | -0.293 | -0.303 |
| I_neutral vs action_valid | -0.251 | -0.278 | -0.296 | -0.291 |
| I_neutral vs G5 | -0.140 | -0.169 | -0.215 | -0.215 |
| H_C vs reward | -0.064 | -0.125 | -0.181 | -0.197 |
| H_A vs G5 | -0.167 | -0.235 | -0.077 | -0.111 |

注意：`results/` 里所有图和表都是包含打转步的版本。

## 6. 按图得出的结论

- **Fig01（I 的分布）**：I 大量堆在 0 附近，拖一条长尾（P50 0.005 / 0.027，P90 2.9 / 4.5）。大部分步里 Thought 不改变动作，只有少数步起决定作用。
- **Fig02（分层看 I）**（图）：开局前 5 步 I 最高，随步数衰减到接近 0；动作越短 I 越高；mapreader 的 I 几乎全是 0。
- **Fig03（熵）**：动作熵一半以上的步正好是 0，Thought 熵也很低（均值约 0.15 nats）。Thought 一写完，动作基本就确定了。
- **Fig04 / E11（零奖励步分类，表）**：I_empty 中位数在“瞎逛”里最高（SW 0.126 / TWX 0.569），在“有效铺垫”（0.0008 / 0.0003）和“重复无效”（0.002 / 0.004）里接近 0。H_C 在“有效铺垫”里最低（0.131 / 0.140）。
- **Fig05 / E09（聚类 bootstrap 相关，表）**：
  - 两个 Bench、两种基线都显著的只有：**I 与单步奖励负相关**（SW -0.217 / -0.231，TWX -0.268 / -0.293）。
  - I_neutral 和所有结果都显著负相关。
  - SW 的 I_empty 和 G3 / G5 / success / valid 的相关都不显著（valid 是 -0.060，CI [-0.130, 0.004]）。TWX 的这几项都显著为负。
  - H_C_mean 在 SW 里全部不显著，在 TWX 里是弱负相关（-0.12 ~ -0.18）。
  - H_A 和各项结果大多显著负相关；例外是 TWX 的 H_A 与 action_valid（+0.008，不显著）。
- **E16（控制步数、长度、合法性后的回归，表）**：I_empty 对单步奖励仍然显著为负（SW p=0.004，TWX p=0.014）；对 G3 不显著（SW p=0.32，TWX p=0.079）；H_C_mean 在所有结果上都不显著。
- **Fig06 / E10（四象限与预测力，表）**：
  - I ≤ 0 的步只占 6.2%（SW）和 7.4%（TWX），但未来奖励最好。SW 里“低 H_C 且 I ≤ 0”的 G3 是 13.86（n=57，合法率 84%），“低 H_C 且 I > 0”只有 2.61。TWX 的 I ≤ 0 两格 G3 分别是 0.130 和 0.183，I > 0 两格约 0.05。
  - 约 80% 的步落在“A < 0 且 I > 0”象限（图）。也就是说，按 A×I 做 credit assignment 的话，大部分 Thought 会被判为该罚。
  - 预测 G3：所有特征组合的 CV R² 约为 0（SW -0.012，TWX 0.001）。
  - 预测终局成功的 AUROC：SW 的 I 是 0.463（不如随机），r 是 0.621；TWX 的 I 是 0.667，r 是 0.574。
- **Fig07 / E13（换基线，表）**：空 Thought 和中性 Thought 两种基线结果一致。Spearman 0.777 / 0.849，正负号一致率 92.2% / 87.1%。格式效应很小（I_fmt_delta 均值 -0.10 / -0.28）。
- **Fig08 / E15（失败前 5 步）**（图）：没有预警信号。熵和 I 在失败前基本平坦，置信区间很宽。
- **Fig09 / E17（Thought 内部位置熵）**：
  - 这里 head 指 CoT 前 1/3 的 token。不合法步的 head 熵明显更高：SW 0.206 vs 0.137，TWX 0.237 vs 0.133。
  - head 熵和 action_valid 的 Spearman 是 -0.227 / -0.293，是所有熵指标里最强的；整段平均只有 -0.038 / -0.149。
  - 按 1/3 分段，mid 和 tail 没有这个差异（SW tail 甚至是合法步更高：0.165 vs 0.147）；按十分位看，TWX 在 60%-90% 处不合法步略低，最后 10% 又更高。对整段取平均会把 head 的信号稀释掉。
  - **混杂（2026-11-19 复核）**：head 熵和“上一步是否报错”的相关（-0.307 / -0.366）比它和本步合法性的相关还强。按上一步是否合法分层后，head 熵和本步合法性的相关只剩 -0.08 ~ -0.18；TWX 中上一步报错的那些步甚至是 +0.03。所以 head 熵高主要说明上一步刚报过错，它单独预测本步对错的能力很弱。
- **Fig10 / E18（Teacher gap）**：D_self 几乎全为负，0 附近的一小撮是第 0 步（那时还没有旧 Thought）。在 C_proxy < 0 的步里（SW 1542 步 / TWX 1139 步），D_self < 0 的 token 平均占 68.4% / 65.4%。这些 token 如果直接拿负优势去乘 D，就会出现“负负得正”。**但要注意**：每一步 token 级 D_self 的中位数只有约 -0.004（一半以上的步，中位数 token 的 |D| < 0.01）；而且把所有步算进来，D<0 的 token 平均也占 65% / 64%，并不是 C<0 的步才有。所以“约 2/3”里大部分是接近 0 的微小负值，量级大的负 D 集中在少数 token 上（每步的 P10 约为 -3 ~ -4）。
- **跨 Bench（表）**：25 个指标对里 24 个方向一致，全部为负。唯一不一致的是 H_A vs action_valid。除 H_A 与 G3 / G5 / action_valid 这三项是 SW 更强外，其余都是 TWX 的效应更强。

**归纳成 5 条**：
1. Thought 大多是旁白，只在开局和少数关键选择上起作用。
2. Thought 起作用不等于起好作用：I 和单步奖励在两边都稳定负相关。I 衡量的是 Thought 在哪里有影响，不是 Thought 好不好。
3. 单步层面的指标预测不了未来（R² 约 0，也没有失败预警）。
4. Thought 开头的熵和本步合法性相关最强，但大部分来自“上一步刚报错”这一混杂，控制之后很弱。动作熵和后续收益的负相关在 SW 更稳。整段平均的 H_C 很弱。
5. D_self 几乎处处为负，直接和优势相乘会有符号冲突。但“约 2/3 token”这个比例主要由接近 0 的微小负值构成，量级大的冲突集中在少数 token 上。

**对之前说法的两处更正**：
- “I 越高越容易不合法”只对 I_neutral（两边）和 TWX 的 I_empty 成立；SW 的 I_empty 不显著。
- “H_C 和结果关系弱”应说成：整段平均的 H_C 在 SW 完全不显著、在 TWX 是弱负相关。head 熵的相关看起来更强，但主要反映上一步是否报错（见 Fig09 条）。

## 7. 对 w = f(C, H) 的初步回答（2026-11-20）

**问题**：设计中的 CoT 优势写作

$$
A^C_{t,i} = w_{t,i} \cdot D_{t,i}, \quad w_{t,i} = f(C_t, H^C_{t,i})
$$

两条待检验的假设：(a) 正向 CoT 里高熵 token 更值得更新；(b) 负向 CoT 里低熵 token 应罚得更重（“自信地错”）。

**结论：只能给半个答案。** 诊断数据能说明 f 不能是什么样，但定不了 f 具体怎么写，后者需要训练对比。以下都是 step 级分析；分组用 |C| > 0.01，H 按任务内 H_C_mean 的中位数切分；A 用的是 A_turn_proxy（含当前步的 G3_incl 在任务内标准化），不是真实优势；表中 G3 为严格未来收益 G3_strict。

**7.1 “用 C_t 判断 CoT 正负”实际退化成 A_t 的符号**
- I_t 有 93% 为正。在 C ≠ 0 的步里，sign(C) = sign(A) 的比例是 97.2%（SW）和 96.4%（TWX），方向几乎全部来自 A。
- |C| < 0.01 的步占 65.7% / 50.2%，也就是说大部分步 CoT 根本不起作用。
- I 本身不分好坏：I 和单步奖励负相关，T→F（换个动作然后出错）的步 I 也很高。
- 因此 C_t 适合做门控或幅度（“这一步 CoT 有没有起作用”），好坏只能靠 A_t。A_t 现在是代理值，单步奖励有 92%–95% 为 0，这一步的瓶颈在 A 的质量。

**7.2 w 不能带符号去乘 D（证据最硬）**
- C < 0 的步占 80.1% / 82.8%，D_self 在 98% 的步里均值为负。负 w 乘负 D，等于把 Teacher 不认可的 token 往上推。
- 步内 token 熵和 D 的相关平均为 -0.248 / -0.198（81.7% / 75.8% 的步为负）；按熵分段，D 的中位数从熵 ≤ 0.05 的约 0 降到熵 > 1 的 -1.37 / -1.38。单个 token 上反转幅度最大的是高熵的分岔 token。
- 例：TWX coin s1000 第 4 步，模型说“硬币不在浴室和储藏室”，实际从没进去过，D = -2.82。这一步 C < 0，相乘后反而会奖励这句编造。
- 因此 f ≥ 0，只决定幅度；逐 token 方向交给 D，不翻转。“CoT 好坏”的符号单独作为一项 RL 信号相加。

**7.3 假设 (b)（负向 CoT 低熵罚更重）：没有一致证据**

| SW 分组 | n | valid | G3 | succ | 打转步占比 | D_self 均值 |
| --- | --- | --- | --- | --- | --- | --- |
| C>0 高 H | 81 | 0.68 | 14.49 | 0.25 | 0.02 | -0.61 |
| C>0 低 H | 43 | 0.84 | 13.02 | 0.49 | 0 | -0.61 |
| C<0 高 H | 396 | 0.45 | -0.737 | 0.05 | 0.09 | -1.02 |
| C<0 低 H | 139 | 0.40 | 0.835 | 0.03 | 0.12 | -1.18 |

| TWX 分组 | n | valid | G3 | succ | 打转步占比 | D_self 均值 |
| --- | --- | --- | --- | --- | --- | --- |
| C>0 高 H | 27 | 0.81 | 0.431 | 0.52 | 0.11 | -1.15 |
| C>0 低 H | 23 | 0.78 | 0.255 | 0.43 | 0 | -1.32 |
| C<0 高 H | 396 | 0.51 | 0.012 | 0.17 | 0.09 | -1.40 |
| C<0 低 H | 239 | 0.55 | 0.027 | 0.15 | 0.11 | -1.52 |

- C < 0 里，低 H 的步没有一致地更差：SW 合法率 0.40 vs 0.45，G3_strict 是 +0.84 vs -0.74；TWX 合法率 0.55 vs 0.51。但这依赖收益口径：若用含当前步的 G3_incl（构造 A 的那个量），SW 低 H 组反而更差（-2.04 vs -1.26）。
- “自信地错”大多是打转：H_C 最低 1/4 的不合法步里，打转步占 84%（SW）/ 46%（TWX），其余不合法步里是 50% / 33%。
- **更正（2026-11-20 token 级直接检验）**：之前由步级汇总推出的“低熵 token 上 D 接近 0、w·D 罚不到它们”不成立。熵 ≤ 0.05 的 token 占 67.2% / 66.0%，D 中位数约 0，但其中 14.7% / 16.0% 满足 D < -1，贡献了全部负 gap 的 44.9% / 47.4%。也就是说，学生很自信而 Teacher 不认可的 token 确实存在，D 本身就对它们给出大幅负值；按熵加权会直接改变这部分 token 的权重。
- 低熵 token 里有大量模板词（“I need to”、“the”），好的 CoT 里也有，加重惩罚会误伤。

**7.4 假设 (a)（正向 CoT 高熵更值得更新）：判断不了**
- 样本少：C > 0.01 的步只有 124（SW）/ 50（TWX）。
- 方向不一致：SW 低 H 更好（合法率 0.84 vs 0.68，成功率 0.49 vs 0.25）；TWX 高 H 略好。
- 高熵 token 的单 token D 更负，w 取常数时它们的更新已经最大；再按熵加权是在同一位置加码。但按总量算，熵 > 0.5 的 token 只贡献约 27% 的负 gap。
- 外部参考：数学推理 RLVR 中只更新少数高熵 token 效果不差（Wang et al. 2025, *Beyond the 80/20 Rule*），但那是结果奖励 RL，不是 OPD，也不是长程 agent。

**7.5 建议拿去训练对比的初步形式**

$$
A^C_{t,i} = \alpha \cdot g(I_t) \cdot A_t + \beta \cdot D_{t,i}
$$

- g(I_t) ∈ [0, 1]，随 I 增大：CoT 起作用的步才把 RL 的功过记到 CoT 上。
- A_t 给好坏的符号；D_{t,i} 给逐 token 方向，永不翻转。
- 熵先不加。要检验两条熵假设，分别把熵乘在 RL 项和 OPD 项上做消融。

**7.6 要定下 f 还缺什么**
1. 训练对比：f 取 1、g(I)、g(I)×熵递增、g(I)×熵递减等几种写法。
2. 更好的 A_t：critic 或 MC rollout，替换 G3 代理。
3. 可信的 Teacher：先做“保留历史 Thought 格式、内容换成中性句”的对照（排除 Teacher 只看到 `Action:` 历史造成的格式偏移），再上 32B 强 Teacher。
4. E14 人工盲标，确认 I_t 在人看来量的是什么。
5. token 级检验已部分完成（§7.3 的更正，按熵分段的表见报告表 4）；还没做按 C 的符号分组的 token 级对比。读 `token_stats.parquet` 需要较新的 pyarrow（见 §8）。

## 8. 已知局限

- SW 合法率 42.5%，低于 E01 定的 60% 门槛；打转步占 1/3。主因是失败后原样重试（固着），不是不知道命令。
- mapreader 20/20 饱和；SW chem / tc 和 TWX cookingworld 都是 0/20。终局层面的结论样本很窄。
- 所有图表都包含打转步（见 §5.4）。
- Fig02、Fig08、Fig10 的趋势，以及 Fig06 里的“约 80%”，是看图读出的，没有单独出数。
- E14 人工盲标还没做，所以 I_t 和人工判断的一致性还没有验证。
- 旧报告 `results/experiment_report.html` 在 2026-11-19 的核对中发现几处解读站不住：
  - 头条“报错后自救步 I_t 暴涨”的倍数主要来自打转步，TWX 去掉打转步后不显著。
  - TWX“I≤0 成功率高”主要是任务构成造成的（I≤0 的步有 46% 来自 mapreader）。
  - E15“低熵麻木”和步数深度混在一起，TWX 不显著。
  - test-conductivity s16 案例写的 D_self=-1.78，和数据（-0.87）不符。
- 2026-11-20 已重写报告：`scripts/06_build_html_report.py` 生成仓库根目录的 `index.html`，公式用 KaTeX（`assets/katex/`，本地 vendored）渲染，图片按相对路径引用 `results/*/figures`。上面几条都已在新报告中改正，旧文件已删除。
  - 生成报告需要读取 parquet：本地 pyarrow 19 读不了服务器上 pyarrow 26 写出的文件（报错 `Repetition level histogram size mismatch`）。可用 `uv venv` 建一个装有 `pyarrow>=21 pandas numpy scipy` 的环境运行脚本。

## 9. 待办

- [ ] **E14 人工盲标**（由用户完成）：在 `results/scienceworld_full/e14_blind_to_label.csv` 和 `results/textworld_express_full/e14_blind_to_label.csv`（各 80 行）里填好 3 个 0/1 列，然后运行：
  ```bash
  python scripts/04_e14_audit.py evaluate --data-dir data/scienceworld_full --labeled-csv results/scienceworld_full/e14_blind_to_label.csv --out-dir results/scienceworld_full
  python scripts/04_e14_audit.py evaluate --data-dir data/textworld_express_full --labeled-csv results/textworld_express_full/e14_blind_to_label.csv --out-dir results/textworld_express_full
  ```
- [ ] **待决定**：是否把“去掉打转步”的对照版并进 `scripts/03_analyze_and_plot.py` 的表和图里。
- [ ] **可选**：mapreader 换更难的参数后重跑（目前 20/20 饱和）。
- [ ] **可选**：开 Qwen3-32B strong teacher（`02_score_offline.py --run-strong-teacher`）补 E18。
- [ ] **建议**：加“格式保留、内容中性”的对照 Teacher（§7.6 第 3 条），需要 GPU。
- [ ] **建议**：把 §5.4 去打转后的相关系数写成入库脚本（§7 和报错后转移表已入库）。

## 10. 文件索引

- `data/{scienceworld,textworld_express}_full/`：`episodes.jsonl`、`turns.jsonl`、`turn_metrics.jsonl`、`token_stats.parquet`
- `data/*_smoke/`：E01 v2（接口说明书 prompt）；`data/*_smoke_v1/`：E01 v1（旧 prompt）
- `results/*_full/`：`figures/Fig01-Fig10`、`tables/*.csv`、`report_summary.md`、`e14_blind_to_label.csv`
- `results/cross_benchmark/`：`cross_benchmark_comparison.md`、`Fig_cross_benchmark_overview.png`、两个 csv
- `index.html`：交互式报告（由 `scripts/06_build_html_report.py` 生成；在线：https://heyxingxing.github.io/cot-action-diag/ ；本地需在仓库目录内打开，依赖 `results/*/figures` 和 `assets/katex/`）
- `docs/experiment_plan_qwen3_8b.md`：实验手册（E01–E18 定义、零样本规则）；`docs/cot_action_overview_same_tasks_v2.pdf`：诊断任务设计总览
- `logs/full_log.txt`：全量运行日志；`logs/e01_v2_log.txt`：E01 v2 日志

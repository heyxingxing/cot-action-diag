# `Qwen/Qwen3-8B` 长程 CoT→Action 诊断实验完整手册（ScienceWorld & TextWorld）

> **这份文档怎么用**：这是把 `cot_action_overview_same_tasks_v2.pdf` 里的全部诊断任务（**E01–E18**）落到实处的执行手册。
> 负责人 A（ScienceWorld）和负责人 B（TextWorld）**共用同一套模型、参数、打分代码和画图代码**，只换各自的环境接口。照着这份手册一步步做，就能产出完全同口径、可以直接横向对比的结果。

---

## 一、先用一分钟搞懂：这个实验到底在干嘛？

在开始写 RL 训练代码之前，我们先花一两天做一次**纯体检（诊断实验）**：
- **完全不训练模型**，只用现成的开源 **`Qwen/Qwen3-8B`** 去玩两款文字交互游戏（ScienceWorld 和 TextWorld）。
- 玩的过程里，每一步模型都会先写一段思考（**CoT**，`Thought:`），再做一个动作（**Action**，`Action:`）。我们把玩完的 60 局游戏全部录下来，保存成文本和对应的 `token IDs`。
- 录完以后，**不用重新玩游戏**，直接在显卡上把录下来的历史记录喂回给模型，做离线打分（`teacher forcing`），算出几个我们最关心的数：
  1. **桥梁函数 $I_t$**：如果把这一步的 CoT 遮住不给模型看，模型做出刚才那个 Action 的概率会降多少？降得多（$I_t > 0$），说明这段 CoT 帮模型更笃定地选了这个 Action；反而升了（$I_t < 0$），说明这段 CoT 在拖这个 Action 的后腿。
  2. **CoT 精确熵 $H^C$**：模型在写这段 CoT 的每个词（token）时，心里到底有多犹豫？（熵高 = 拿不准、犹豫；熵低 = 很自信。）
  3. **Action 精确熵 $H^A$**：模型在写 Action 时有多犹豫？（拿来当对照组，和 CoT 熵比一比。）
  4. **游戏真实奖励 $r_t$、未来几步收益 $G_t^{(3)}, G_t^{(5)}$ 和最后赢没赢**：用来检查上面算出来的 $I_t$ 和 $H^C$ 到底跟任务好坏有没有真关系。
  5. **Teacher Gap $D_{t,i}$（可选/顺手算）**：如果把历史里模型自己乱写的旧 CoT 都删掉，只留干净的“做过什么动作 + 看到什么环境”（State-OPD 的做法），让 Teacher 对当前 CoT 的每个词重新打分，看看它和完整历史打分差多少。

### 整体工作流（4 步搞定，绝不反复重跑）

```text
第 0 步：A 和 B 对齐锁死同一份 configs/protocol.yaml（模型、参数、格式完全一致）
   │
   ▼
第 1 步（在线玩游戏，用 vLLM）：跑 E01（5 局试水）→ 没问题后跑 E02（60 局正式采集）
   │    存下 episodes.jsonl + turns.jsonl（包含原始文本、token IDs、奖励）
   ▼
第 2 步（离线算分，用 HF Transformers）：加载一次模型，对刚才存的 60 局连续跑 3~4 遍前向计算
   │    一次性把 E03(空CoT) + E05(CoT熵) + E06(Action熵) + E13(中性CoT) + E18(Teacher Gap) 全算完！
   │    存下 turn_metrics.jsonl + token_stats.parquet
   ▼
第 3 步（CPU 画图统计）：A 和 B 跑同一份分析脚本，自动算出 E04 / E07~E12 / E15~E17，输出 10 张图
   │
   ▼
第 4 步（人工抽查 E14）：各自抽 50~100 步肉眼标一遍，确认数学指标和人类直觉对不对得上
```

### 人员分工
- **负责人 A**：负责 **ScienceWorld**，跑完 E01–E18（先 P0，再 P1）。
- **负责人 B**：负责 **TextWorld（TextWorldExpress）**，跑完一模一样的 E01–E18（先 P0，再 P1）。
- **如果来了第三个人**：不改变 A 和 B 各自跑全套的安排，优先让第三个人负责**交叉复核两边产出的数据和图**，并统一主抓 **E18 Teacher Gap 深入分析**（比如补跑 `Qwen3-32B` 强 Teacher）。

---

## 二、机器与环境怎么配？（1 张 A800 80GB 就够）

### 2.1 算力与耗时实话账单
当前诊断阶段**租 1 张 A800（80GB）就绰绰有余**，完全不需要一开始租 2 张（等后面真要跑双 LoRA RL 训练时再扩到 2 张）：

| 干什么 | 用什么跑 | 显存账单（权重 + KV Cache + 开销） | 跑完 1 个 Bench（60 局 ≈ 1,500 步）要多久 |
|---|---|---|---|
| **第 1 步：玩游戏采集（E01–E02）** | vLLM（单卡，开 Prefix Caching） | **实际必需 ~22–38 GB**（16GB 权重 + 并发 KV Cache + 激活）；<br>但 `gpu_memory_utilization=0.85` 时 vLLM 启动会**预占 ~68 GB**（把剩余 ~48GB 全划成 KV Cache 池，支持高并发和跨轮缓存复用；若不想占满可调成 `0.5` 占 40GB） | 10–25 分钟 |
| **第 2 步：离线算分 Pass 1–3（E03/05/06/13）** | HF `transformers` + FA2 | ~30–38 GB | 15–25 分钟 |
| **第 2 步：离线算 E18（Self-Teacher，同一 8B）** | HF `transformers` + FA2 | ~28 GB | 5–10 分钟 |
| **第 2 步：离线算 E18（可选，强 Teacher `Qwen3-32B`）** | HF `transformers` + FA2（bs=1） | ~68–74 GB（单卡 80G 刚好塞下） | 20–35 分钟 |
| **第 3 步：统计与画图** | CPU | 0 GB | 2 分钟以内 |

### 2.2 必须装好的软件（踩坑预警：两个游戏都要 Java！）

```bash
# 1. 先装 Java！ScienceWorld 和 TextWorldExpress 底层引擎都是 Java 写的，没装直接报错
sudo apt-get update && sudo apt-get install -y openjdk-11-jdk
java -version   # 看到 11 或 17 就行

# 2. 建 Python 环境
conda create -n cot_diag python=3.10 -y
conda activate cot_diag

# 3. 装推理和打分框架（注意：transformers 必须 >= 4.51.0 才认识 Qwen3 和 enable_thinking）
pip install "torch>=2.4.0" --index-url https://download.pytorch.org/whl/cu124
pip install "transformers>=4.51.0" "accelerate>=1.0.0" "vllm>=0.8.5"
pip install flash-attn --no-build-isolation

# 4. 装数据处理和画图包
pip install pyarrow pandas numpy scipy scikit-learn matplotlib seaborn pyyaml tqdm

# 5. 装两个 Benchmark（建议 A 和 B 都装，方便互看）
pip install scienceworld
pip install textworld-express
```

---

## 三、全组锁死的统一参数配置（`protocol.yaml`）

下面这份配置就是 A 和 B 的**共同合同**，每一项参数我都用人话写清楚了**为什么要这么设**：

```yaml
# configs/protocol.yaml
experiment:
  name: "cot_action_diag_qwen3_8b_v1"
  seed: 42

model:
  hf_id: "Qwen/Qwen3-8B"
  revision: "下载后填具体 commit hash 锁死，防止中途模型仓库更新"
  dtype: "bfloat16"
  # 【重要】主线实验先关掉 Qwen3 原生的 <think> 长思考，统一用 ReAct 的 Thought: / Action:
  # 原因：1）开原生 <think> 一轮动辄几百上千字，35 轮游戏上下文直接爆掉；
  #      2）不会出现 <think> 里想一遍、Thought: 里又想一遍的“双层 CoT”混乱；
  #      3）跟文档里写的 ReAct 模板完全对齐。
  enable_thinking: false
  max_context_tokens: 12288              # 最多喂给模型 12K token，超过就把最早的中间轮次裁掉并记 truncated=true

rollout:
  engine: "vllm"
  tensor_parallel_size: 1
  # vLLM 会按这个比例启动时预占整卡显存（80G × 0.85 ≈ 68G），权重(~16G)之外基本全给 KV Cache 池；
  # 独享整卡就留 0.85（配合 prefix caching 跑极快）；若同卡还要跑别的东西可降到 0.5（~40G，仍够 10+ 局并发）
  gpu_memory_utilization: 0.85
  enable_prefix_caching: true            # 强烈建议开！多轮游戏每步前缀 95% 一样，复用 KV Cache 能快 3-5 倍
  # 以下 4 个是 Qwen3-8B 官方推荐的 Non-Thinking 采样参数
  # 【绝对不要设 temperature=0】Qwen3 官方明确提醒：贪婪解码会导致模型变傻、陷入死循环复读！
  temperature: 0.7
  top_p: 0.8
  top_k: 20
  min_p: 0.0
  # BF16 下先设 0.0，保证玩游戏时采样的概率分布和离线打分时的原始模型分布尽量一致；
  # 如果 E01 试水发现模型疯狂复读同一句话，A 和 B 再一起把它改成 0.5 并记录下来。
  presence_penalty: 0.0
  repetition_penalty: 1.0
  max_tokens_per_turn: 256               # 一轮 Thought(1-3句) + Action(1句) 绝对够用，防止模型话痨跑飞
  stop_sequences:                        # 防止模型自己替环境编造下一步 Observation
    - "\nObservation:"
    - "\nUser:"
    - "<|im_end|>"
  max_turns_per_episode: 35              # 每局游戏最多玩 35 步，A 和 B 保持一致

offline_scoring:
  engine: "hf_transformers"              # 为什么离线打分不用 vLLM？因为 vLLM 只给 top-k 概率，算不出全词表精确熵！
  attn_implementation: "flash_attention_2"
  dtype: "bfloat16"
  entropy_compute_dtype: "float32"       # 算熵的时候临时转 fp32，防止精度误差或 NaN
  entropy_unit: "nats"                   # 用自然对数 ln 算熵，统一单位叫 nats
  # 【重要】离线算概率和熵时，必须用模型最原始的分布：temperature=1.0，不做 top_p/top_k 截断！
  scoring_temperature: 1.0
  batch_size_turns: 4                    # 80G 显存很宽裕，按长度分桶后可开 4~8
  # 【千万注意】绝不把全词表 logits 存到硬盘！（15万词表 × 1500步会吃掉 100GB+ 硬盘）
  # 正确做法：在显卡上当场算出每个 token 的 logprob(1个数) 和 entropy(1个数)，存完就把 logits 扔掉！
  save_full_vocab_logits: false
  neutral_cot_text: "I will choose the next action based on the current observation."
  teacher_mode:
    enable_self_teacher: true            # 顺手算 E18：同一 Qwen3-8B，把历史里的旧 Thought 删掉
    enable_strong_teacher: false         # 如果有余力想看强模型 Teacher，再改成 true 跑 Qwen/Qwen3-32B
    strong_teacher_hf_id: "Qwen/Qwen3-32B"

metrics:
  future_horizons: [3, 5]                # E08 算未来 3 步和 5 步收益
  bootstrap_resamples: 1000              # E09 按 episode 分组做 1000 次重采样算 95% 置信区间
  manual_audit_turns_per_bench: 80       # E14 每边抽 80 步人工标注
```

> **如果你后面想对比 Qwen3 原生思考模式（可选扩展）**：单独开一个新分支，把 `enable_thinking` 改成 `true`，采样参数同步换成官方 Thinking 推荐值（`temperature=0.6, top_p=0.95, top_k=20, max_tokens_per_turn=1024`），直接把 `<think>...</think>` 当成 CoT，把 `<think>\n\n</think>` 当成无 CoT 基线，**绝不要和主线混在一起**。

---

## 四、Prompt 怎么写？Token 怎么切？离线 4 遍打分怎么算？

### 4.1 统一的对话格式（Chat Template）
用 `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)` 拼接多轮对话：

- **`system`（系统指令）**：
  - **A 和 B 一字不差的公共指令**：
    ```text
    You are an interactive agent solving a multi-step task in a text environment.
    At every turn, you MUST respond in the exact two-line format below:
    Thought: <concise reasoning in 1-3 sentences about the current state and next step>
    Action: <exactly one valid environment command>

    Rules:
    - Always output "Thought:" first, followed by a newline and "Action:".
    - Output ONLY one action per turn. Do not add extra explanations after the action.
    - Do not invent objects or commands that are not allowed by the environment.
    ```
  - **各自追加的环境规则**（只允许写该游戏的合法命令长什么样 + 当前任务目标，**禁止**给某一边偷加解题技巧或示例）：
    - **零样本**：不放任何示范轨迹，不附每步合法动作列表，不改环境反馈。允许写的只有“接口说明书”：命令格式、命令本身的前置条件和效果（例如门要先开才能走），全部以环境自带模板和实测回显为准（原文见 `cot_action_diag/src/prompts.py`）。
    - **ScienceWorld**：`look around | look at OBJ | look in OBJ | inventory | task`、`go to LOC`（只接受相邻且门已打开的房间名）、`open/close OBJ`、`pick up OBJ`、`put down OBJ`（放到当前房间）、`move OBJ to OBJ`（放进/放到另一物体）、`pour OBJ in OBJ | mix OBJ | dunk OBJ in OBJ | eat OBJ | flush OBJ`、`activate/deactivate OBJ | use OBJ on OBJ | read OBJ`、`connect OBJ to OBJ | disconnect OBJ`、`wait | wait1`、`focus on OBJ`（选错对象直接判负），并注明没有 `take` / `put ... in`、容器内物体要用观测里的全名，加上当前任务描述 `task_description`。
    - **TextWorldExpress**：按游戏给各自真实存在的命令（取自该游戏的 `validActions`）：coin = `look around | inventory | move DIR | open/close door to DIR | take OBJ`（只有门能开关）；mapreader = `look around | inventory | task | move DIR | take OBJ | put OBJ in OBJ | read OBJ`；cookingworld = `look around | inventory | examine OBJ | read OBJ | move DIR | open/close OBJ | take OBJ | put OBJ in OBJ | chop/slice/dice OBJ`（需持有刀）`| cook OBJ in stove/oven/barbeque | prepare meal | eat meal`；加上当前任务描述 `task_description`。（原列表里的 `fry/roast/grill` 在 TWX 中不存在，已删除。）
- **每轮 `user`（环境给的新画面）**：
  ```text
  Turn {t}:
  Observation: {observation_t}
  ```
- **每轮 `assistant`（模型说的话）**：
  ```text
  Thought: {cot_t}
  Action: {action_t}
  ```

### 4.2 怎么解析模型输出（Parser 规则）
用固定正则把 `Thought:` 后面的内容抠出来当 `cot`，把 `Action:` 那一行抠出来当 `action`：
- **正常情况（`parse_ok = True`）**：同时抠到了非空的 `cot` 和非空的 `action`。
- **没按格式说话（`parse_ok = False`）**：**不要把这一步删掉！** 原样存下来，记清楚错在哪（比如没写 Thought、没写 Action、格式全乱）。如果还能勉强抽出 `Action:` 就照常发给环境，抽不出来就发空指令让环境反馈错误；**连续 3 步格式全坏就提前结束这一局**，记终止原因为 `repeated_parse_fail`。

### 4.3 切 Token 的铁律（这一步错了，后面全白算！）
模型生成的一整轮话长得像这样：
`[Thought: ]` + **`[CoT 正文]`** + `[\nAction: ]` + **`[Action 正文]`** + `[结束符]`

必须死守三条规矩：
1. **玩游戏时直接把生成出来的 `token_ids` 存下来**，不要存完纯文本后再重新分词，不然空格和换行符容易把词切歪。
2. **算 CoT 熵和 Teacher Gap 时，只算 `[CoT 正文]` 的那些 token**，不要把 `"Thought: "` 这几个模板字算进去。
3. **算桥梁函数 $I_t$ 和 Action 熵时，只算 `[Action 正文]` 的那些 token**，**绝对不能把 `"\nAction: "` 和结束符算进去！** 并且离线算“有 CoT / 空 CoT / 中性 CoT”三遍分时，代码里必须加一行断言（`assert`），检查这三遍末尾拼上去的 `[Action 正文]` 的 `token_ids` **每一个数字都完全相同**。

### 4.4 第 2 步离线打分：加载一次模型，连续算 4 遍（每遍在干嘛？）

假设某一步模型实际说了 `Thought: z_t \nAction: a_t`，我们在显卡上喂给模型 4 份不同的输入，都不生成新词，只算已有词的概率和熵（teacher forcing）：

| 第几遍 | 给模型看的“历史” | 给模型看的“当前轮开头” | 让模型对哪段词算分 | 算出什么指标 |
|---|---|---|---|---|
| **Pass 1（原样）** | 完整历史 $h_t$（含旧 CoT、旧 Action、旧 Obs） | `Thought: {z_t}\nAction: ` | `z_t` 和 `a_t` | ① CoT 精确熵 $H^C$（E05）<br>② Action 精确熵 $H^A$（E06）<br>③ 有 CoT 时的动作概率 $\log\pi_A(a_t \mid h_t, z_t)$（E03）<br>④ Student 自己的 CoT 概率 $\log\pi_S(z_{t,i} \mid h_t, z_{t,<i})$（E18） |
| **Pass 2（空 CoT）** | 同上（完整历史 $h_t$） | `Thought:\nAction: ` | 同一个 `a_t` | 空 CoT 时的动作概率 $\log\pi_A^{\text{empty}}(a_t \mid h_t)$ → 算出 **$I_t^{\text{empty}}$（E03）** |
| **Pass 3（中性 CoT）** | 同上（完整历史 $h_t$） | `Thought: I will choose the next action based on the current observation.\nAction: ` | 同一个 `a_t` | 中性 CoT 时的动作概率 $\log\pi_A^{\text{neutral}}(a_t \mid h_t)$ → 算出 **$I_t^{\text{neutral}}$（E13）** |
| **Pass 4（Teacher，E18）** | 干净历史 $s_t$（**把以前所有轮的 `Thought:` 全删掉**，只留任务目标 + 历史 Action + 历史/当前 Obs） | `Thought: ` | 同一个 `z_t` | Teacher 的 CoT 概率 $\log\pi_T(z_{t,i} \mid s_t, z_{t,<i})$ → 算出 **$D_{t,i}$（E18）** |

---

## 五、两个 Benchmark 的任务挑选与 Reward 说明书

挑任务的共同大原则：**不要挑 5 局全赢的送分题，也不要挑 5 局全零分的地狱题**。每个 Bench 挑 **3 类不同难度的任务 × 每类 20 局 = 60 局**，保证有赢有输、有中间得分，这样后面才分得出好坏。

### 5.1 负责人 A：ScienceWorld（`scienceworld`）
- **环境参数**：`env = ScienceWorldEnv("", envStepLimit=35)`，加载时传 `simplificationStr=""`（标准原版，不开作弊简化）。
- **推荐 3 类任务（共 60 局）**：
  1. **偏简单（L1）**：`find-living-thing`（`variationIdx = 0..19`）—— 找房间里的生物放进盒子，比较容易拿到中间分和通关。
  2. **中等（L2）**：`chemistry-mix-paint-secondary-color`（`variationIdx = 0..19`）—— 找到两种原色颜料倒进同一容器、混合出二次色并 focus，典型的多步规划。（原定 `melt` 的标准解法步数中位数 34，只有 4/20 个变体能在 25 步内完成，35 步上限下几乎做不完，E01 后替换。）
  3. **偏难（L3）**：`test-conductivity`（`variationIdx = 0..19`）—— 搭电路测未知物体导电性，容易出各种长程推理错误。
  - *（如果 E01 试水发现某类 5 局全 0 或全 100，从备选池里换：`find-non-living-thing`、`boil`、`freeze`、`grow-plant`、`mix-paint`。）*
- **Reward 到底是什么意思？（E07 必写清楚）**：
  - `score`（`info["score"]`）：当前累计任务进度，正常是 **0 到 100**，满分 100 算赢（`terminal_success = True`）；如果把任务搞砸了（比如弄死动物、毁坏关键物品）会直接变成 **-100**。
  - `reward_raw`（`step()` 返回的 `reward`）：这一步相比上一步的进度变化量（`score_t - score_{t-1}`）。绝大多数正常的赶路、开门、拿东西都是 **0**，只有踩中某个子目标里程碑才给正分，踩雷才给负分。**所以 0 绝对不是坏动作！**
  - `action_valid`：动作被环境实际执行（回显不含 `"No known action matches"`、`"The door is not open"` 等失败提示，完整列表见 `scienceworld_adapter.py` 的 `_SW_FAILURE_PATTERNS`）。不用合法列表做判据，因为 `get_valid_action_object_combinations()` 漏掉了大量解析器能接受的写法；严格的列表匹配结果另存为 `action_in_valid_list`。

### 5.2 负责人 B：TextWorld（`textworld-express`）
- **环境参数**：`env = TextWorldExpressEnv(envStepLimit=35)`。
- **重要细节**：加载时设 `gameFold="dev"`、固定 `seed = 1000..1019`（这样把 `train` 和 `test` 留干净给以后真正的 RL 训练和测试，不偷看考题！）。
- **推荐 3 类任务（共 60 局）**：
  1. **偏简单（L1）**：`coin`（`gameParams="numLocations=7,includeDoors=1"`）—— 穿过带门的房间找硬币放回起点盒子（拿到硬币 0.5 分，放回去 1.0 分）。
  2. **中等（L2）**：`mapreader`（`gameParams="numLocations=11,includeDoors=1"`）—— 先读地图再按图找路，最能看出 CoT 有没有认真做路径规划。（7 个地点生成不了默认 `maxDistanceApart=4`，dev seed 1000 会崩；E01 用 7 地点 + 距离 3 时 2/2 全成功、比 L1 还容易，改为 11 地点，20 个 dev seed 均可生成，标准解法 7–13 步。）
  3. **偏难（L3）**：`cookingworld`（`gameParams="numLocations=5,numIngredients=3,includeDoors=1"`）—— 读菜谱→到处找 3 种食材→按要求切/煎/烤→做菜→吃掉，步骤多、容易忘状态。
  - *（如果 E01 试水发现太简单或太难，直接改 `numLocations` 5~11 或 `numIngredients` 2~4 调到有赢有输即可。）*
- **Reward 到底是什么意思？（E07 必写清楚）**：
  - `score`（`infos["score"]`）：已经归一化好的累计进度，范围 **0.0 到 1.0**。
  - `reward_aux`（`infos["scoreRaw"]`）：没归一化的原始整数得分。
  - `reward_raw`（`step()` 返回的 `reward`）：这一步相比上一步的得分增量。同样大部分正常走路、翻箱倒柜都是 **0**，只有完成关键节点（拿到食材、加工成功、做好菜等）才给正分。
  - `action_valid`：动作（小写化、去冠词后）在 `infos["validActions"]` 里，且解析器没返回 `"Unknown action"`。TWX 解析器区分大小写，所以动作先转小写再发给环境。
  - `terminal_success`：直接看 `infos["tasksuccess"]`。

---

## 六、完整实验清单（逐项对应 `cot_action_overview_same_tasks_v2.pdf` 的 E01–E18）

> **说明**：下面把原文档里的 **P0（E01–E12）** 和 **P1（E13–E18）** 全部列齐。每一项都写清：**① 干嘛用的 ② 具体怎么做 ③ 必须记下哪些东西 ④ 避坑点**。A 和 B 逐项照做，一个都不漏。

---

### P0：A、B 两人都必须完成（E01–E12）

#### [ ] E01 跑通环境和 ReAct（5 局试水）
- **干嘛用的**：别一上来就跑 60 局，先各跑 5 局看看代码、环境、Prompt、解析器和打分有没有低级 bug，顺便确认选的 3 类任务难易度合不合适。
- **怎么做**：在 3 类任务里挑 5 个样例（比如 L1 2 局、L2 2 局、L3 1 局），完整走一遍 `Observation → CoT → Action → 下一 Observation`，并且把这 5 局塞进 Stage 2 离线打分脚本跑通。
- **必须记录**：
  - 5 局里每一步的原始输出、`parse_ok`（解析成没成功）、`action_valid`（动作合不合法）、每局怎么结束的（赢了 / 触发致命错误 / 走满 35 步 / 连续解析失败）。
  - 打印 2~3 条完整的“有 CoT / 无 CoT”切片文本和 `token_ids`，肉眼核对 `[Action 正文]` 有没有夹带 `Action:`。
- **过关门槛（4 条全过才能进 E02）**：
  1. 解析成功率 `parse_ok ≥ 90%`；
  2. 合法动作率 `action_valid ≥ 60%`；
  3. 3 类任务合起来不是全 0 分也不是全满分（有赢有输或有明显子目标得分拉开）；
  4. 离线算出的 $I_t$ 和精确熵 $H^C$ 没有 `NaN` / `Inf`，且 Pass 1/2/3 的 Action token IDs 逐位一致。

#### [ ] E02 批量采集（60 局正式数据）
- **干嘛用的**：用锁死的模型和参数，批量录下后续所有分析要用的轨迹原料。
- **怎么做**：严格按 `protocol.yaml`（`Qwen/Qwen3-8B`, `enable_thinking=False`, `temp=0.7, top_p=0.8, top_k=20`, `max_turns=35`），跑完 3 类任务 × 每类 20 个 variation/seed = **60 局**（预计每 Bench 产出 1,200–2,000 个 Turn）。
- **必须记录**：写入 `episodes.jsonl` 和 `turns.jsonl`（具体字段见第七节总表），核心包含：完整历史、当前 Observation、模型原始输出、切好的 CoT 和 Action 文本、**精确的 `prompt_token_ids`、`generated_token_ids`、`cot_token_span`、`action_token_span`**、以及原始奖励与终局状态。

#### [ ] E03 CoT→Action 桥梁函数 $I_t$
- **干嘛用的**：量一下“这一步写的 CoT，到底把实际做出的这个 Action 的概率拉高了还是拉低了”。
- **怎么做**：在 Stage 2 的 Pass 1 和 Pass 2 里，对**同一个已经采样出来的 Action $a_t$**、用**同一个模型**，只改变“提不提供当前这一步的 CoT $z_t$”（历史 $h_t$ 保持不动，不重新跑环境），按 Action 的 token 数取平均：
  $$I_t = \frac{\log \pi_A(a_t \mid h_t, z_t) - \log \pi_A(a_t \mid h_t)}{|a_t|}$$
  - **怎么读懂这个数**：$I_t > 0$ 表示加了当前 CoT 之后，模型更倾向于说出这个 Action；$I_t < 0$ 表示加了当前 CoT 之后，模型反而更不想说这个 Action（但它还是被采样出来了）。
  - **同时顺手记录未除以 $|a_t|$ 的版本** $I_t^{\text{raw}} = \log \pi_A(a_t \mid h_t, z_t) - \log \pi_A(a_t \mid h_t)$，方便后面 E16 检查是不是动作长短影响了结果。
- **必须记录**：`logp_a_with_cot`, `logp_a_empty`, `action_token_len`, `I_empty_norm`, `I_empty_raw`。
- **避坑提醒**：**千万别把 $I_t$ 当成“CoT 和 Action 语义一不一致”，也别当成“CoT 的真实因果贡献”！** 它只是一个方便算出来的概率变化代理指标。

#### [ ] E04 桥梁函数分布
- **干嘛用的**：先看看 $I_t$ 长什么样——大部分时候是正的还是负的？不同任务、不同步数下变化大不大？
- **怎么做**：对所有有效 Turn 统计 $I_t$ 的**均值、P10 / P50（中位数）/ P90、正数占比、负数占比**；并按 4 个维度分别分组对比画图：
  1. 按**任务类型**（L1 / L2 / L3）；
  2. 按**Turn 深度**（比如第 1–5 步、6–15 步、16–35 步）；
  3. 按**Action 长度**（按 $|a_t|$ token 数分短/中/长）；
  4. 按**该局最终成功 vs 失败**。
- **必须记录**：各分组的样本量（多少局、多少步）、均值、P10/P50/P90、正负比例（存 `summary_stats.csv` + 出 **Fig 01, Fig 02**）。

#### [ ] E05 CoT token 熵（核心）
- **干嘛用的**：测模型在写这段 CoT 时，每个词到底有多拿不准（不确定性）。**记住：整个课题以后要做“熵加权”，指的都是这里的 CoT 熵！**
- **怎么做**：在 Stage 2 的 Pass 1 里，对 CoT 正文的每一个 token $i$，把模型输出的**全词表 logits（151,936 维）转成 `float32`**，算精确香农熵（单位 `nats`）：
  $$H_{t,i}^C = -\sum_{v=1}^{|\mathcal{V}|} p(v \mid h_t, z_{t,<i}) \log p(v \mid h_t, z_{t,<i})$$
  - 如果万不得已用了 top-k 近似，必须在字段里标 `entropy_is_approx = true`；按本方案用 HF 跑 Pass 1 就能直接拿到**精确熵**（`entropy_is_approx = false`）。
- **必须记录**：
  - 逐 token 级（存 `token_stats.parquet`）：每个 CoT token 的 `token_pos, token_id, logp_student, entropy_nats`。
  - 每轮汇总级（存 `turn_metrics.jsonl`）：`cot_token_len`、`H_C_mean`（均值）、`H_C_p50`、`H_C_p90`（90 分位）、`H_C_max`（最大值）、`H_C_std`、以及**高熵 token 位置**（比如熵最高的 top-3 token 及其相对位置 `H_C_peak_pos_rel`）。

#### [ ] E06 Action token 熵（对照）
- **干嘛用的**：当对照组！看看“思考时的犹豫（CoT 熵）”和“做动作时的犹豫（Action 熵）”到底一不一样，是不是 CoT 熵更有信息量。
- **怎么做**：同样在 Pass 1 里，顺手对 `[Action 正文]` 的每个 token $j$ 算精确熵 $H_{t,j}^A$。
- **必须记录**：
  - 逐 token 级（同存 `token_stats.parquet`，`segment="action"`）：每个 Action token 的 `entropy_nats`。
  - 每轮汇总级（存 `turn_metrics.jsonl`）：`action_token_len`、`H_A_mean`、`H_A_max`。
  - 出图对比 $H^C$ 与 $H^A$ 的分布和相关性（**Fig 03**）。

#### [ ] E07 Reward 覆盖率与语义盘点
- **干嘛用的**：摸清这两个游戏发奖励的脾气——有多少步是 0 奖励、多少步给正分、多少步扣分？防止后面误把 0 奖励的正常铺垫步当成坏动作。
- **怎么做**：统计所有 Turn 中 `reward_raw > 0`、`reward_raw == 0`、`reward_raw < 0` 各自的比例，以及合法/非法动作比例；把该 Bench 奖励怎么触发的规律写进结论里。
- **必须记录**：每轮保存 `reward_raw`、`reward_aux`、`reward_source`、`score`、`score_norm`、`action_valid`；汇总正/零/负比例表 + 出 **Fig 04（左）**。

#### [ ] E08 后续收益（$G_t^{(3)}, G_t^{(5)}$ 与终局得分）
- **干嘛用的**：很多好动作（比如拿钥匙、开火炉）当下这一步奖励是 0，但过两三步就会拿到奖励！所以不能只看这一步的 $r_t$，还要往后多看 3 步和 5 步。
- **怎么做**：直接在已有轨迹上离线把后面的步子加起来（末尾剩下不到 3 步或 5 步的，就加到最后一步为止）。
  - **我们特意算两套（超级重要，帮 E10 防踩坑）**：
    1. **含当前步**：$G_t^{(3,\text{incl})} = r_t + r_{t+1} + r_{t+2}$（$G_t^{(5,\text{incl})}$ 同理）；
    2. **严格未来步（不含当前 $r_t$）**：$G_t^{(3,\text{strict})} = r_{t+1} + r_{t+2} + r_{t+3}$（$G_t^{(5,\text{strict})}$ 同理）。
- **必须记录**：`G3_incl, G5_incl, G3_strict, G5_strict, final_score_norm, terminal_success`。

#### [ ] E09 指标与 Reward 的关联（按 Episode 聚类！）
- **干嘛用的**：正面检查核心问题——$I_t$、CoT 熵、Action 熵，到底和“这一步得分、未来 3/5 步得分、最后赢没赢”有没有统计上靠谱的相关性？
- **怎么做**：分别计算 4 个指标（$I_t$、$H^C_{\text{mean}}$、$H^C_{\text{P90}}$、$H^A_{\text{mean}}$）与 4 个结果（$r_t$、$G_3$、$G_5$、终局成功 $Y_{\text{succ}}$）的相关系数（Spearman $\rho$ 和 Pearson $r$）。
- **必须守住的统计规矩（人话解释）**：
  - 同一局游戏里的 30 步不是互相独立的（一局卡死循环可能刷出 30 步垃圾，不能让这一局顶 6 局成功局！）。
  - 所以必须**按 `episode_id` 聚类**来算不确定度：做 1,000 次 **Episode-level Cluster Bootstrap**（每次随机有放回地抽 60 局，再拿这 60 局里的所有步算相关系数），得到 95% 置信区间；同时额外算一个**局内平均相关**（每局自己内部算一个相关系数，再对 60 局取平均）。
- **必须记录**：每组检验的 `N_episodes`、`N_turns`、相关系数及 95% CI（存 `e09_corr.csv` + 出 **Fig 05**）。

#### [ ] E10 两指标联合分析（高/低熵 × 正/负 $I_t$，以及乘积项检验）
- **干嘛用的**：看看把“CoT 熵”和“桥梁函数 $I_t$”拼在一起，是不是比单看一个更有用，顺便提前检验我们以后想构造的信号靠不靠谱。
- **怎么做（拆成 3 个动作，彻底避开“自己猜自己”的坑）**：
  1. **2×2 分组看均值**：按 `H_C_mean`（任务内中位数切高/低）× `$I_t$ 正/负`，把所有 Turn 分成 4 组，看这 4 组的 `action_valid`、$r_t$、$G_3^{\text{strict}}$、$G_5^{\text{strict}}$ 和终局成功率。
  2. **四象限拆解（检验原文档 §4 的“负×负=正”）**：用任务内减去均值的代理优势 $\tilde{A}_t$（有正有负，比如 $G_3^{\text{incl}} - \text{该任务平均 } G_3^{\text{incl}}$）× `$I_t$ 正/负` 分 4 组，**专门把 $(\tilde{A}_t < 0,\; I_t < 0)$ 这一格的数据量、非法动作率和典型例子拎出来看**，看看它到底是“好 CoT 预警了坏动作”还是“坏 CoT 胡言乱语”。
  3. **公平检查乘积项 $I_t \times r_t$ 和 $I_t \times G_3$ 有没有额外预测价值**：
     - **注意坑**：如果拿 $I_t \times r_t$ 去预测 $r_t$，并且只跟**单独的 $I_t$** 比，那是自己猜自己，必然显得更准，不算数！
     - **正确比法**：拿特征去预测**不包含自己的严格未来目标**（比如用 $I_t \times r_t$ 去预测严格未来收益 $G_3^{\text{strict}}$ 或终局赢没赢），并且同时对比 3 组：① 只用 $I_t$；② 只用 $r_t$（或 $G_3$）；③ $r_t + I_t + I_t \times r_t$。只有③比②更准，才说明乘上 $I_t$ 真的多带来了信息！
- **必须记录**：4 组/四象限统计表 + 预测对比表（存 `e10_quadrants.csv` + 出 **Fig 06**）。
- **口径提醒**：这一步**只叫做相关分析，绝对不要宣称它是真实的 CoT Advantage**。

#### [ ] E11 零奖励与错误动作大解剖
- **干嘛用的**：文字游戏里 70%~90% 的步子都是 $r_t = 0$，但同样是 0 分，有的是“正在拿钥匙（好步）”，有的是“撞墙/输错命令（蠢步）”。我们要看 $I_t$ 和 CoT 熵能不能把这几类 0 分动作区分开！
- **怎么做**：把所有 `reward_raw == 0` 的 Turn 抽出来，先用下面这套 **5 选 1 互斥规则树**自动分类（A/B 完全一样，客观可复现），再抽几条人工复核：
  1. `parse_fail`（格式解析失败）：`parse_ok == False`；
  2. `illegal_action`（非法操作）：格式对了，但 `action_valid == False`；
  3. `repeat_noop`（重复行动/原地打转）：动作执行后 `next_observation == observation`（环境啥也没变），或者这一局里之前在同一个画面下已经做过一模一样的动作；
  4. `effective_setup`（必要铺垫 / 有效探索）：合法、环境确实变了，而且接下来 5 步内拿到了正分（$G_t^{(5,\text{strict})} > 0$）或者这一局最后赢了；
  5. `aimless_explore`（无用动作 / 瞎逛）：合法、环境变了，但接下来 5 步一分没拿到而且这一局最后输了。
- **必须记录**：每个 Turn 打上 `zero_reward_category` 标签；统计 5 类各自占多少比例，以及这 5 类各自的 $I_t$ 和 $H^C$ 分布（出 **Fig 04（右）**）。

#### [ ] E12 统一交付打包
- **干嘛用的**：把 P0 阶段所有原料、图表和结论打包成标准格式，方便 A 和 B 直接并排对照。
- **必须交付的 4 样东西**：
  1. **数据文件**：`episodes.jsonl`、`turns.jsonl`、`turn_metrics.jsonl`、`token_stats.parquet`；
  2. **复现命令与配置**：实际跑过的 `protocol.yaml` + 一键复现命令行记录；
  3. **8–10 张统一格式关键图**（见第八节清单）；
  4. **1–2 页诊断结论（`report_summary.md`）**：必须写清每个结论的样本量，并且**专门贴出 3–5 个真实的异常例子与典型失败模式**（比如高 $I_t$ 却做蠢事、低熵却自信胡说、$I_t < 0$ 到底长什么样）。

---

### P1：P0 顺利后两人都做（E13–E18）
*(工程建议：**E13 和 E18 在 Stage 2 加载模型时顺手一起算掉最省时间**，分析画图可以等 P0 验收完再出！)*

#### [ ] E13 无 CoT 基线稳健性（空 `Thought:` vs 中性 `Thought:`）
- **干嘛用的**：排雷！确认我们算出来的 $I_t$ 不是被“把 `Thought:` 后面掏空导致的奇怪格式”骗出来的。
- **怎么做**：拿 Stage 2 Pass 2 算出来的 $I_t^{\text{empty}}$（空 `Thought:`）和 Pass 3 算出来的 $I_t^{\text{neutral}}$（固定写一句废话 `"I will choose the next action based on the current observation."`）做逐 Turn 对比。
- **必须记录**：两者的 Pearson / Spearman 相关系数、正负号一致率、差值 $\Delta I_t^{\text{fmt}} = I_t^{\text{empty}} - I_t^{\text{neutral}}$ 的均值和分布（出 **Fig 07**）。如果空基线几乎全是正数、而中性基线正负参半，就说明空基线有严重格式偏差，后面必须改用中性基线。

#### [ ] E14 语义一致性人工抽检（每 Bench 抽 50–100 个 Turn）
- **干嘛用的**：数学数字靠不靠谱，最终得拿人的眼睛校准一遍！看看 $I_t$ 算得高的时候，这段 CoT 在人眼里是不是真的讲清楚了行动意图、且没跟 Action 打架。
- **怎么做**：从每个 Bench **分层抽取 80 个 Turn**（保证同时覆盖赢局/输局、$I_t$ 明显为正 / 接近 0 / 明显为负、以及非法动作），导成一张干净的 CSV（**标注时先隐藏 $I_t$ 数值，防止先入为主！**），每条标 3 个 0/1 项：
  1. `cot_explicit_intent`：CoT 里是不是明确说出了下一步要干啥？（1 = 明确指出，0 = 没说/很模糊）
  2. `action_contradicts_cot`：实际做的 Action 是不是跟 CoT 说的自相矛盾？（1 = 矛盾/脱节，0 = 一致）
  3. `cot_too_vague`：CoT 是不是太空泛笼统、全是废话？（1 = 过于笼统，0 = 具体有信息）
  - 标完后再揭晓 $I_t$ 和 $H^C$，比较人工标注与 $I_t$ 的一致度（AUROC 和分组均值），**切忌把 $I_t$ 直接混同于语义一致性**。
- **必须记录**：标注好的 `e14_manual_audit.csv` + 各标注类别对应的 $I_t$ / $H^C$ 均值与典型案例。

#### [ ] E15 长程失败模式演化
- **干嘛用的**：看看模型在长程任务里是怎么一步步“走偏/崩盘”的——死循环或失败前那几步，CoT 熵和 $I_t$ 提前有没有征兆？
- **怎么做**：挑出两类典型事故现场：
  1. **失败轨迹终止前的最后 3–5 步**（按倒数第 5、4、3、2、1 步对齐）；
  2. **连续重复同一个 Action（死循环）发生的前 2 步到循环中**。
  画出这两类场景下 $H^C_{\text{mean}}$、$H^C_{\text{P90}}$ 和 $I_t$ 的变化曲线（出 **Fig 08**）。
- **必须记录**：倒计时各步的样本量与均值/中位数变化，判断失败前究竟是“越来越迷茫（熵飙升）”还是“越来越固执（低熵死循环）”。

#### [ ] E16 控制变量检验
- **干嘛用的**：防假象！比如也许仅仅因为“难任务里 CoT 写得长、熵也高、又容易输”，让人误以为“高熵导致输”，其实全是任务难度造成的假相关。
- **怎么做**：做两件事：
  1. **分层拆开看**：按任务难度（L1/L2/L3）、CoT 长度分箱、Turn 深度分箱、Action 合法与否，分别在组内重算一次 $I_t$ 和 $H^C$ 与收益的关系；
  2. **做一次多元回归（带 Episode 聚类标准误）**：把 `task_id`、当前步数 `turn_id`、`cot_len`、`action_len`、`action_valid` 全部放进去当控制变量，看控制完之后 $I_t$ 和 $H^C$ 的系数是不是依然显著、方向有没有翻车。
- **必须记录**：回归系数表（`e16_reg.csv`）与分组对照图（合入 **Fig 02**）。

#### [ ] E17 CoT 位置分析（开头 / 中部 / 结尾 & 局部高熵峰值）
- **干嘛用的**：一段 CoT 可能有 80 个词，大多数是“Okay, I need to...”这种没信息的连接词，真正决定动作的往往只有中间或结尾那几个关键词。我们要看**局部高熵峰值**是不是比**整段平均熵**更能预测 Action 好不好。
- **怎么做**：把每段 CoT 按 token 位置三等分：**开头（前 1/3）、中部（中 1/3）、结尾（后 1/3）**（画图时也可以切 10 份看精细曲线），分别算平均熵；再比较 `H_C_mean`（全段均值）、`H_C_p90`、`H_C_max`（局部峰值）谁对动作合法性和未来收益预测力更强（出 **Fig 09**）。
- **必须记录**：`H_C_head_mean, H_C_mid_mean, H_C_tail_mean, H_C_peak_pos_rel` 以及各汇总量的预测对比。

#### [ ] E18 Teacher Gap 分析（可选，但建议 Stage 2 顺手算）
- **干嘛用的**：提前帮未来的 State-OPD 探路——把历史里乱七八糟的旧 CoT 全删掉后，Teacher 对当前 CoT 的打分和 Student 差多少？这个差值和 $I_t$、CoT 熵、Reward 之间有没有规律？
- **怎么做**：无需训练！在 Stage 2 Pass 4 里，用固定 Teacher（默认先跑同模型 `Qwen/Qwen3-8B` 删历史 CoT 做 Self-Teacher；单卡 80G 有余力可再跑一遍 `Qwen/Qwen3-32B`），只喂任务目标 + 历史 Action + 历史/当前 Observation，对同一个当前 CoT token 计算：
  $$D_{t,i} = \log \pi_T(z_{t,i} \mid s_t, z_{t,<i}) - \log \pi_S(z_{t,i} \mid h_t, z_{t,<i})$$
  - **怎么读懂 $D_{t,i}$**：$D_{t,i} > 0$ 表示干净视野下的 Teacher 比 Student 更认可这个词；$D_{t,i} < 0$ 表示 Teacher 觉得 Student 在胡说/跑偏。
  - **额外重点看一眼**：在“坏 CoT”（$\tilde{A}_t \cdot I_t < 0$）里，同时出现 $D_{t,i} < 0$ 的 token 占多少比例？（这能直接验证后续训练时千万不能把带负号的权重直接乘在 $D_{t,i}$ 上，否则会负负得正！）
- **必须记录**：`D_self_mean, D_self_p10, D_self_p50, D_self_p90, D_self_neg_frac`，逐 token 的 `logp_teacher_self`（存 `token_stats.parquet`），以及 $D_{t,i}$ 与 $H_{t,i}^C$、$I_t$、Reward 的关系图（**Fig 10**）。

---

## 七、所有必须记录的数据字段总表（精确到每一列）

为了避免 A 和 B 存出来的格式对不上，这里把 4 个输出文件的每一列都定死（**完全覆盖原文档里的统一日志字段，同时补全了原文档被截断的 `score`，并把全词表 logits 改为轻量的逐 token 标量**）：

### 1. `episodes.jsonl`（每局游戏 1 行，每个 Bench 共 60 行）
| 字段名 | 类型 | 含义 |
|---|---|---|
| `benchmark` | str | `"scienceworld"` 或 `"textworld_express"` |
| `task_id` | str | 任务名（如 `"chemistry-mix-paint-secondary-color"`, `"cookingworld"`） |
| `difficulty_tier` | str | `"L1"` / `"L2"` / `"L3"` |
| `seed` | int | ScienceWorld 的 `variationIdx` 或 TextWorld 的 `seed` |
| `episode_id` | str | 全局唯一 ID，如 `"{benchmark}_{task_id}_{seed}"` |
| `model_hf_id` / `model_revision` | str | 模型名与精确 commit hash |
| `num_turns` | int | 这一局一共走了多少步 |
| `total_reward` | float | 全程 `reward_raw` 之和 |
| `final_score` / `final_score_norm` | float | 终局原始分 / 归一化到 `[0,1]` 的分 |
| `terminal_success` | bool | 这一局最后赢没赢 |
| `termination_reason` | str | `"success"` / `"failure"` / `"max_turns"` / `"repeated_parse_fail"` |
| `parse_fail_count` / `invalid_action_count` | int | 全程解析失败次数 / 非法动作次数 |

### 2. `turns.jsonl`（每一步 1 行，约 1,200–2,000 行，Stage 1 产出）
| 字段名 | 类型 | 含义 |
|---|---|---|
| `benchmark, task_id, seed, episode_id, turn_id` | str/int | 定位哪一局、第几步（`turn_id` 从 0 开始） |
| `history` | list[dict] | 截至当前轮发给模型的完整对话消息列表（不含当前回复） |
| `observation` / `next_observation` | str | 执行动作前的画面 / 执行动作后的新画面 |
| `raw_output` | str | 模型原始吐出来的整段文字 |
| `cot` / `action` | str | 解析出来的纯 CoT 正文 / 纯 Action 正文 |
| `parse_ok` / `parse_error_type` | bool/str | 格式解析是否成功 / 失败原因 |
| `action_valid` | bool | 该动作是不是环境认可的合法动作 |
| `reward_raw` | float | 环境这一步返回的原始单步奖励 |
| `reward_aux` | float | 辅助原始分数（如 TextWorld 的 `scoreRaw`，ScienceWorld 可留同 `score`） |
| `reward_source` | str | 标明奖励来源口径（`"scienceworld_subgoal_delta"` / `"textworld_express_delta"`） |
| `score` / `score_norm` | float | 执行完这一步后的累计任务得分（原始值 / 归一化值） |
| `done` / `truncated` | bool | 游戏是否在这一步结束 / 这一步输入是否触发了 12K 上下文截断 |
| `prompt_token_ids` / `generated_token_ids` | list[int] | 精确的输入与生成 token ID 序列 |
| `cot_token_span` / `action_token_span` | `[int, int]` | 在 `generated_token_ids` 中，纯 CoT 正文和纯 Action 正文的起止下标（左闭右开） |

### 3. `turn_metrics.jsonl`（每一步 1 行，Stage 2 离线打分产出）
- **定位键**：`episode_id, turn_id`
- **桥梁函数（E03 & E13）**：`logp_a_with_cot, logp_a_empty, logp_a_neutral, action_token_len, I_empty_norm, I_empty_raw, I_neutral_norm, I_neutral_raw, I_fmt_delta`
- **CoT 精确熵（E05 & E17）**：`cot_token_len, entropy_is_approx, H_C_mean, H_C_p50, H_C_p90, H_C_max, H_C_std, H_C_head_mean, H_C_mid_mean, H_C_tail_mean, H_C_peak_pos_rel, top3_high_entropy_tokens`
- **Action 精确熵（E06）**：`H_A_mean, H_A_max`
- **后续收益与类别（E08, E10, E11）**：`G3_incl, G5_incl, G3_strict, G5_strict, A_turn_proxy, A_ep_proxy, zero_reward_category`
- **Teacher Gap（E18）**：`D_self_mean, D_self_p10, D_self_p50, D_self_p90, D_self_neg_frac, corr_D_HC_within_turn`（若跑了 32B 强 Teacher，再加同名 `D_strong_*`）

### 4. `token_stats.parquet`（逐 token 标量表，60 局仅 5–15 MB）
`episode_id, turn_id, segment ("cot"|"action"), token_pos, token_id, token_text, logp_student, entropy_nats, logp_teacher_self`

---

## 八、全组统一约定、10 张交付图与汇总方式

### 8.1 全组 5 条统一约定（直接沿用并细化原文档）
1. **模型一致**：同一个 `Qwen/Qwen3-8B` checkpoint（锁 commit hash）、同一个 ReAct 结构 `Thought: / Action:`、同一套采样与打分参数；两个 Bench 只允许换任务规则、合法动作说明和环境 API。
2. **桥梁函数输入一致**：有 CoT 与无 CoT 必须对**同一个已采样 Action（完全一样的 token IDs）**打分，固定模型参数，无 CoT 对照不重新执行环境。
3. **Reward 分开保存、不硬比绝对值**：分别保存 `reward_raw, reward_aux, reward_source, score`；ScienceWorld（0–100）和 TextWorld（0–1）语义不同，**绝对不直接把两边原始 reward 值拼在一起比大小**，而是各自在 Bench 内算完相关性、分组对比和标准化趋势后，再横向对照规律。
4. **先不训练、不乱戴高帽**：本阶段不把单步 reward、$I_t$ 或二者乘积宣称为“真实 CoT 因果 credit”；先诚实验证这些指标稳不稳、准不准、能不能解释现象。
5. **代码与图表同源**：A 和 B 跑同一份 Stage 2 打分脚本和 Stage 3 画图脚本，保证横纵坐标、分箱口径、置信区间算法 100% 一致。

### 8.2 统一输出的 10 张关键图（每张图必须标明 `N_episodes` 和 `N_turns`）

| 图号 | 对应任务 | 画什么图 | 要看懂什么问题 |
|---|---|---|---|
| **Fig 01** | E04 | 直方图 + 成功/失败箱线图 | $I_t^{\text{empty}}$ 和 $I_t^{\text{neutral}}$ 的整体长相（均值、P10/P50/P90、正负占比），赢局和输局有没有区别？ |
| **Fig 02** | E04 + E16 | 分组箱线图/折线图 | $I_t$ 按任务难度（L1/L2/L3）、步数深浅、动作长短拆开后，规律还在不在？ |
| **Fig 03** | E05 + E06 | 双分布对比 + 散点图 | CoT 熵和 Action 熵分别长什么样，两者相关性强不强？ |
| **Fig 04** | E07 + E11 | 堆叠柱状图 + 五类箱线图 | 正/零/负奖励占比是多少？0 奖励拆成 5 类动作后，$I_t$ 和 CoT 熵分不分得开它们？ |
| **Fig 05** | E09 | 森林图（带 95% 置信区间） | 4 个指标和 4 个收益指标的按局聚类相关性，哪些显著不为 0？ |
| **Fig 06** | E10 | 2×2 热力图 + 四象限拆解 | 高/低熵 × 正/负 $I_t$ 谁表现最好？$(\tilde{A}_t < 0, I_t < 0)$ 象限到底是不是噪音？乘积项比单独 $r_t$ 强吗？ |
| **Fig 07** | E13 | 散点对比图 | 空 `Thought:` 和中性 `Thought:` 算出来的 $I_t$ 一致性高不高？有没有被格式骗了？ |
| **Fig 08** | E15 | 失败倒计时折线图 | 失败前最后 5 步、以及死循环前后，CoT 熵和 $I_t$ 是怎么演变的？ |
| **Fig 09** | E17 | CoT 位置熵曲线 | CoT 开头、中部、结尾哪里最犹豫？局部峰值是不是比全段均值更准？ |
| **Fig 10** | E18 (+ E14) | 相关性热力图 + 人工校准柱状图 | Teacher Gap $D_{t,i}$ 跟熵/$I_t$/奖励什么关系？（做完 E14 后再附上人工标注的对齐度） |

### 8.3 最终怎么汇总 & 拿结果做什么决定？
A 和 B 各自交齐 4 样交付物（E12）后，把两边的 **Fig 01–10 和统计表并排贴在一起对照**，重点区分两类发现：
- **跨 Bench 共同成立的规律**（ScienceWorld 和 TextWorld 都出现）：这才是可以放心写进后续 RL 方法的核心机制！
- **只在某一个 Bench 成立的现象**：明确记为环境特性（比如某个环境奖励更稀疏或动作空间更窄），不把它过度推广成通用结论。

最后用跑出来的实锤数据，直接回答下一阶段做 RL 训练时的 4 个拍板问题：
1. **无 CoT 基线选哪个？**（看 E13：空基线如果明显偏正，就换中性基线。）
2. **$I_t < 0$ 要不要参与负负得正？**（看 E10 + E14：如果 $(\tilde{A}_t < 0, I_t < 0)$ 里全是胡言乱语/脱节，后续就把 $I_t$ 截断到 $\ge 0$ 或改门控，绝不做负负得正。）
3. **熵加权到底加在高熵还是低熵？用均值还是峰值？**（看 E05 + E15 + E17 的真实曲线说话。）
4. **要不要上 State-OPD Teacher？**（看 E18：如果删掉历史 CoT 后 $D_{t,i}$ 信号干净稳定，后续就把 OPD 和 RL 信号**相加**结合；如果删掉历史 CoT 反而让 Teacher 失忆变傻，就趁早调整 Teacher 上下文或换强模型 Teacher。）

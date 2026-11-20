# 诊断实验总结报告：textworld_express（Qwen/Qwen3-8B）

## 1. 样本规模与基础体检（E01 / E02 / E07）
- **总计**：`N_episodes = 60`，`N_turns = 1375`（平均 `22.9` 步/局）
- **格式解析成功率 (`parse_ok`)**：`100.0%`
- **动作合法率 (`action_valid`)**：`59.9%`
- **终局成功率 (`terminal_success`)**：`50.0%`，归一化平均终局得分：`0.544`
- **单步 Reward 占比**：正 `5.2%` / 零 `94.8%` / 负 `0.0%`

## 2. 核心指标分布（E03–E06 & E13）
- **桥梁函数 `I_empty_norm`**：均值 `1.2900`，P10/P50/P90 = `0.000 / 0.027 / 4.534`，正值占比 `92.6%`
- **中性基线 `I_neutral_norm`**：均值 `1.5728`，正值占比 `83.6%`；与空基线 Spearman = `0.849`，正负号一致率 = `87.1%`
- **CoT 精确熵 `H_C_mean`**：均值 `0.1629 nats`，P90 = `0.5018 nats`；对照 Action 熵 `H_A_mean` 均值 `0.0264 nats`

## 3. E09 按 Episode 聚类核心相关性（节选）

| x | y | n_episodes | n_turns | spearman_pooled | ci_low | ci_high | pearson_pooled | within_ep_spearman_mean | within_ep_valid_eps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| I_empty_norm | reward_raw | 60 | 1375 | -0.2677 | -0.3078 | -0.225 | -0.1007 | -0.3534 | 34 |
| I_empty_norm | G3_strict | 60 | 1375 | -0.2103 | -0.2886 | -0.1282 | -0.0651 | 0.1068 | 34 |
| I_empty_norm | G5_strict | 60 | 1375 | -0.1777 | -0.2561 | -0.0891 | -0.0412 | 0.2013 | 34 |
| I_empty_norm | terminal_success | 60 | 1375 | -0.233 | -0.3357 | -0.1306 | -0.0653 |  | 0 |
| I_empty_norm | action_valid | 60 | 1375 | -0.1723 | -0.2647 | -0.0814 | -0.0093 | -0.0595 | 38 |
| I_neutral_norm | reward_raw | 60 | 1375 | -0.2929 | -0.3315 | -0.2498 | -0.1181 | -0.353 | 34 |
| I_neutral_norm | G3_strict | 60 | 1375 | -0.2347 | -0.3038 | -0.1522 | -0.092 | 0.1382 | 34 |
| I_neutral_norm | G5_strict | 60 | 1375 | -0.215 | -0.2919 | -0.1262 | -0.0748 | 0.2044 | 34 |
| I_neutral_norm | terminal_success | 60 | 1375 | -0.2665 | -0.3626 | -0.1634 | -0.1023 |  | 0 |
| I_neutral_norm | action_valid | 60 | 1375 | -0.2962 | -0.3786 | -0.2099 | -0.0746 | -0.1439 | 38 |
| H_C_mean | reward_raw | 60 | 1375 | -0.1807 | -0.2392 | -0.1263 | -0.1484 | -0.3292 | 34 |
| H_C_mean | G3_strict | 60 | 1375 | -0.1319 | -0.2129 | -0.0483 | -0.0955 | 0.1285 | 34 |
| H_C_mean | G5_strict | 60 | 1375 | -0.1224 | -0.2076 | -0.0332 | -0.1125 | 0.192 | 34 |
| H_C_mean | terminal_success | 60 | 1375 | -0.153 | -0.2545 | -0.0428 | -0.1571 |  | 0 |
| H_C_mean | action_valid | 60 | 1375 | -0.1489 | -0.2874 | -0.0106 | -0.1622 | -0.0903 | 38 |
| H_C_p90 | reward_raw | 60 | 1375 | -0.1596 | -0.2179 | -0.1074 | -0.1512 | -0.3056 | 34 |
| H_C_p90 | G3_strict | 60 | 1375 | -0.1146 | -0.1938 | -0.0338 | -0.0848 | 0.1518 | 34 |
| H_C_p90 | G5_strict | 60 | 1375 | -0.1064 | -0.1888 | -0.0227 | -0.1018 | 0.2094 | 34 |
| H_C_p90 | terminal_success | 60 | 1375 | -0.1489 | -0.2498 | -0.0406 | -0.1536 |  | 0 |
| H_C_p90 | action_valid | 60 | 1375 | -0.1425 | -0.27 | -0.008 | -0.1416 | -0.0836 | 38 |
| H_A_mean | reward_raw | 60 | 1375 | -0.1433 | -0.196 | -0.0878 | -0.0587 | -0.2831 | 34 |
| H_A_mean | G3_strict | 60 | 1375 | -0.0664 | -0.1297 | -0.0084 | 0.0117 | 0.1001 | 34 |
| H_A_mean | G5_strict | 60 | 1375 | -0.0771 | -0.1472 | -0.0038 | 0.0121 | 0.1018 | 34 |
| H_A_mean | terminal_success | 60 | 1375 | -0.0844 | -0.1739 | -0.0012 | 0.0244 |  | 0 |
| H_A_mean | action_valid | 60 | 1375 | 0.0077 | -0.1055 | 0.1272 | 0.0155 | -0.0467 | 38 |

## 4. E10 无泄露预测力对比（GroupKFold CV）

| target | feature_set | cv_mean | cv_std | n_folds_used |
| --- | --- | --- | --- | --- |
| G3_strict (R2) | I_t_only | 0.0013 | 0.0084 | 5 |
| G3_strict (R2) | r_t_only | 0.001 | 0.0035 | 5 |
| G3_strict (R2) | r_t + I_t + I_t*r_t | 0.0032 | 0.008 | 5 |
| terminal_success (AUROC) | I_t_only | 0.6667 | 0.0863 | 5 |
| terminal_success (AUROC) | r_t_only | 0.5744 | 0.0225 | 5 |
| terminal_success (AUROC) | r_t + I_t + I_t*r_t | 0.6724 | 0.0847 | 5 |

## 5. E16 控制变量回归（Episode 聚类稳健标准误）

| outcome | term | coef | se_clustered | p_value | r2 | n_turns | n_episodes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| reward_raw | I_empty_norm | -0.00181 | 0.00071 | 0.01358 | 0.1182 | 1375 | 60 |
| reward_raw | H_C_mean | 0.02946 | 0.05684 | 0.60615 | 0.1182 | 1375 | 60 |
| reward_raw | H_C_p90 | -0.03726 | 0.025 | 0.14138 | 0.1182 | 1375 | 60 |
| reward_raw | turn_id | -5e-05 | 0.00036 | 0.89954 | 0.1182 | 1375 | 60 |
| reward_raw | cot_token_len | -0.00219 | 0.00065 | 0.00132 | 0.1182 | 1375 | 60 |
| reward_raw | action_token_len | 0.00626 | 0.00378 | 0.10356 | 0.1182 | 1375 | 60 |
| reward_raw | action_valid_int | 0.01837 | 0.00679 | 0.0089 | 0.1182 | 1375 | 60 |
| G3_strict | I_empty_norm | -0.00372 | 0.00208 | 0.07925 | 0.2436 | 1375 | 60 |
| G3_strict | H_C_mean | -0.02741 | 0.10615 | 0.79714 | 0.2436 | 1375 | 60 |
| G3_strict | H_C_p90 | 0.00496 | 0.03534 | 0.88894 | 0.2436 | 1375 | 60 |
| G3_strict | turn_id | 0.00048 | 0.00069 | 0.49536 | 0.2436 | 1375 | 60 |
| G3_strict | cot_token_len | 1e-05 | 0.0008 | 0.98818 | 0.2436 | 1375 | 60 |
| G3_strict | action_token_len | -0.01785 | 0.00483 | 0.00048 | 0.2436 | 1375 | 60 |
| G3_strict | action_valid_int | 0.0118 | 0.01174 | 0.319 | 0.2436 | 1375 | 60 |
| final_score_norm | I_empty_norm | -0.00199 | 0.0044 | 0.65311 | 0.5055 | 1375 | 60 |
| final_score_norm | H_C_mean | 0.22147 | 0.26973 | 0.41491 | 0.5055 | 1375 | 60 |
| final_score_norm | H_C_p90 | -0.12901 | 0.10439 | 0.22142 | 0.5055 | 1375 | 60 |
| final_score_norm | turn_id | -0.00343 | 0.0015 | 0.0259 | 0.5055 | 1375 | 60 |
| final_score_norm | cot_token_len | -0.00388 | 0.00363 | 0.28993 | 0.5055 | 1375 | 60 |
| final_score_norm | action_token_len | -0.03464 | 0.01781 | 0.05649 | 0.5055 | 1375 | 60 |
| final_score_norm | action_valid_int | -0.04598 | 0.05232 | 0.38298 | 0.5055 | 1375 | 60 |

## 6. E18 Teacher Gap 与符号冲突检验
- 在 `C_proxy = A_turn_proxy * I_t < 0` 的 `1139` 个负向 CoT Turn 中，同时出现 `D_self < 0` 的 token 平均占比为 **`65.4%`**（直接衡量若误将负权重乘在 `D` 上发生“负负得正”的比例）。

## 7. 典型异常案例摘录（供人工复核）
- **负收益且 I_t < 0（原表第 4 行）** (`textworld_express__mapreader__s1019`, turn=1, `I_empty=-0.247`, `H_C=0.091`, `r_t=0.0`, `valid=True`, `succ=True`)
  - CoT: `The laundromat is not visible here. I should check other directions to find the laundromat.`
  - Action: `move West`
- **高 I_t 但动作非法或终局失败** (`textworld_express__cookingworld__s1007`, turn=25, `I_empty=20.777`, `H_C=0.098`, `r_t=0.0`, `valid=True`, `succ=False`)
  - CoT: `The chicken wing is roasted. Now I need to dice the red apple as per the recipe instructions. I should check the counter for the red apple.`
  - Action: `examine counter`
- **低熵却做出非法动作（自信犯错）** (`textworld_express__coin__s1019`, turn=11, `I_empty=2.063`, `H_C=0.002`, `r_t=0.0`, `valid=False`, `succ=False`)
  - CoT: `The kitchen cupboard is already open, but I did not find the coin inside. I should continue checking other closed objects in the kitchen, such as the cutlery dr`
  - Action: `open cutlery drawer`

## 8. 交付图表清单（位于 `figures/`）
- `Fig01_E04_bridge_dist.png` ~ `Fig10_E18_teacher_gap.png`（共 10 张，300 DPI）
# 诊断实验总结报告：scienceworld（Qwen/Qwen3-8B）

## 1. 样本规模与基础体检（E01 / E02 / E07）
- **总计**：`N_episodes = 60`，`N_turns = 1924`（平均 `32.1` 步/局）
- **格式解析成功率 (`parse_ok`)**：`100.0%`
- **动作合法率 (`action_valid`)**：`42.5%`
- **终局成功率 (`terminal_success`)**：`13.3%`，归一化平均终局得分：`0.425`
- **单步 Reward 占比**：正 `7.8%` / 零 `92.0%` / 负 `0.2%`

## 2. 核心指标分布（E03–E06 & E13）
- **桥梁函数 `I_empty_norm`**：均值 `0.8395`，P10/P50/P90 = `0.000 / 0.005 / 2.940`，正值占比 `93.8%`
- **中性基线 `I_neutral_norm`**：均值 `0.9402`，正值占比 `89.6%`；与空基线 Spearman = `0.777`，正负号一致率 = `92.2%`
- **CoT 精确熵 `H_C_mean`**：均值 `0.1533 nats`，P90 = `0.4708 nats`；对照 Action 熵 `H_A_mean` 均值 `0.0148 nats`

## 3. E09 按 Episode 聚类核心相关性（节选）

| x | y | n_episodes | n_turns | spearman_pooled | ci_low | ci_high | pearson_pooled | within_ep_spearman_mean | within_ep_valid_eps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| I_empty_norm | reward_raw | 60 | 1924 | -0.2173 | -0.2595 | -0.1779 | -0.083 | -0.237 | 55 |
| I_empty_norm | G3_strict | 60 | 1924 | -0.0534 | -0.113 | 0.0131 | 0.0251 | -0.0187 | 53 |
| I_empty_norm | G5_strict | 60 | 1924 | -0.0621 | -0.1236 | 0.0061 | 0.0301 | -0.0002 | 53 |
| I_empty_norm | terminal_success | 60 | 1924 | -0.0457 | -0.096 | 0.0078 | 0.0355 |  | 0 |
| I_empty_norm | action_valid | 60 | 1924 | -0.0602 | -0.1297 | 0.0039 | 0.1605 | -0.0522 | 59 |
| I_neutral_norm | reward_raw | 60 | 1924 | -0.231 | -0.2684 | -0.192 | -0.0912 | -0.2219 | 55 |
| I_neutral_norm | G3_strict | 60 | 1924 | -0.1222 | -0.1848 | -0.0565 | 0.0094 | -0.0581 | 53 |
| I_neutral_norm | G5_strict | 60 | 1924 | -0.1405 | -0.2034 | -0.0724 | 0.011 | -0.0539 | 53 |
| I_neutral_norm | terminal_success | 60 | 1924 | -0.1075 | -0.1732 | -0.0269 | 0.0089 |  | 0 |
| I_neutral_norm | action_valid | 60 | 1924 | -0.2506 | -0.3227 | -0.1791 | 0.0686 | -0.1883 | 59 |
| H_C_mean | reward_raw | 60 | 1924 | -0.0638 | -0.1191 | 0.0001 | -0.0401 | -0.0617 | 55 |
| H_C_mean | G3_strict | 60 | 1924 | -0.0443 | -0.1248 | 0.039 | -0.0782 | -0.0481 | 53 |
| H_C_mean | G5_strict | 60 | 1924 | -0.0457 | -0.1308 | 0.0425 | -0.0687 | -0.0313 | 53 |
| H_C_mean | terminal_success | 60 | 1924 | -0.0421 | -0.0983 | 0.0169 | -0.0513 |  | 0 |
| H_C_mean | action_valid | 60 | 1924 | -0.038 | -0.1649 | 0.0838 | -0.0566 | -0.0837 | 59 |
| H_C_p90 | reward_raw | 60 | 1924 | -0.042 | -0.0988 | 0.0181 | -0.0232 | -0.0368 | 55 |
| H_C_p90 | G3_strict | 60 | 1924 | -0.0374 | -0.1133 | 0.0417 | -0.0832 | -0.0388 | 53 |
| H_C_p90 | G5_strict | 60 | 1924 | -0.0426 | -0.1226 | 0.0419 | -0.078 | -0.0267 | 53 |
| H_C_p90 | terminal_success | 60 | 1924 | -0.0388 | -0.095 | 0.0145 | -0.0423 |  | 0 |
| H_C_p90 | action_valid | 60 | 1924 | -0.0284 | -0.1452 | 0.0829 | -0.0339 | -0.0744 | 59 |
| H_A_mean | reward_raw | 60 | 1924 | -0.1146 | -0.1566 | -0.068 | -0.0369 | -0.0961 | 55 |
| H_A_mean | G3_strict | 60 | 1924 | -0.1443 | -0.2088 | -0.0699 | -0.015 | -0.0934 | 53 |
| H_A_mean | G5_strict | 60 | 1924 | -0.167 | -0.2348 | -0.0933 | 0.0143 | -0.1034 | 53 |
| H_A_mean | terminal_success | 60 | 1924 | -0.0694 | -0.1299 | -0.0041 | -0.0081 |  | 0 |
| H_A_mean | action_valid | 60 | 1924 | -0.1405 | -0.2323 | -0.0508 | 0.0344 | -0.1207 | 59 |

## 4. E10 无泄露预测力对比（GroupKFold CV）

| target | feature_set | cv_mean | cv_std | n_folds_used |
| --- | --- | --- | --- | --- |
| G3_strict (R2) | I_t_only | -0.0121 | 0.0095 | 5 |
| G3_strict (R2) | r_t_only | -0.0102 | 0.0106 | 5 |
| G3_strict (R2) | r_t + I_t + I_t*r_t | -0.0107 | 0.009 | 5 |
| terminal_success (AUROC) | I_t_only | 0.4626 | 0.0383 | 5 |
| terminal_success (AUROC) | r_t_only | 0.6207 | 0.0183 | 5 |
| terminal_success (AUROC) | r_t + I_t + I_t*r_t | 0.5956 | 0.0618 | 5 |

## 5. E16 控制变量回归（Episode 聚类稳健标准误）

| outcome | term | coef | se_clustered | p_value | r2 | n_turns | n_episodes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| reward_raw | I_empty_norm | -0.46661 | 0.15571 | 0.00399 | 0.0488 | 1924 | 60 |
| reward_raw | H_C_mean | -6.51253 | 5.24296 | 0.2191 | 0.0488 | 1924 | 60 |
| reward_raw | H_C_p90 | 1.72776 | 2.09218 | 0.41223 | 0.0488 | 1924 | 60 |
| reward_raw | turn_id | -0.10976 | 0.02593 | 8e-05 | 0.0488 | 1924 | 60 |
| reward_raw | cot_token_len | -0.02116 | 0.02926 | 0.4723 | 0.0488 | 1924 | 60 |
| reward_raw | action_token_len | 0.18205 | 0.06993 | 0.01167 | 0.0488 | 1924 | 60 |
| reward_raw | action_valid_int | 2.66257 | 0.55374 | 1e-05 | 0.0488 | 1924 | 60 |
| G3_strict | I_empty_norm | 0.21857 | 0.21814 | 0.32045 | 0.0872 | 1924 | 60 |
| G3_strict | H_C_mean | 10.35907 | 8.65697 | 0.23624 | 0.0872 | 1924 | 60 |
| G3_strict | H_C_p90 | -7.32559 | 3.63714 | 0.04857 | 0.0872 | 1924 | 60 |
| G3_strict | turn_id | -0.22246 | 0.05514 | 0.00016 | 0.0872 | 1924 | 60 |
| G3_strict | cot_token_len | -0.27636 | 0.09307 | 0.00431 | 0.0872 | 1924 | 60 |
| G3_strict | action_token_len | 0.03839 | 0.12834 | 0.76591 | 0.0872 | 1924 | 60 |
| G3_strict | action_valid_int | 3.66366 | 0.79614 | 2e-05 | 0.0872 | 1924 | 60 |
| final_score_norm | I_empty_norm | -0.00116 | 0.00391 | 0.76739 | 0.3578 | 1924 | 60 |
| final_score_norm | H_C_mean | 0.33801 | 0.20939 | 0.11181 | 0.3578 | 1924 | 60 |
| final_score_norm | H_C_p90 | -0.05953 | 0.07541 | 0.43306 | 0.3578 | 1924 | 60 |
| final_score_norm | turn_id | -0.0012 | 0.0011 | 0.27887 | 0.3578 | 1924 | 60 |
| final_score_norm | cot_token_len | -0.00095 | 0.00337 | 0.77948 | 0.3578 | 1924 | 60 |
| final_score_norm | action_token_len | 0.00984 | 0.00762 | 0.20146 | 0.3578 | 1924 | 60 |
| final_score_norm | action_valid_int | 0.09324 | 0.03258 | 0.00582 | 0.3578 | 1924 | 60 |

## 6. E18 Teacher Gap 与符号冲突检验
- 在 `C_proxy = A_turn_proxy * I_t < 0` 的 `1542` 个负向 CoT Turn 中，同时出现 `D_self < 0` 的 token 平均占比为 **`68.4%`**（直接衡量若误将负权重乘在 `D` 上发生“负负得正”的比例）。

## 7. 典型异常案例摘录（供人工复核）
- **负收益且 I_t < 0（原表第 4 行）** (`scienceworld__find-living-thing__s18`, turn=23, `I_empty=-0.097`, `H_C=0.058`, `r_t=0.0`, `valid=False`, `succ=False`)
  - CoT: `I have focused on the cherry tree, which is a living thing. Now I need to move it to the red box in the bedroom.`
  - Action: `move cherry tree in reproducing stage to red box`
- **高 I_t 但动作非法或终局失败** (`scienceworld__test-conductivity__s16`, turn=28, `I_empty=15.436`, `H_C=0.292`, `r_t=0.0`, `valid=True`, `succ=False`)
  - CoT: `I need to test the electrical conductivity of the aluminum foil. I will use the battery and wires in the workshop to create a simple circuit and observe the res`
  - Action: `pick up battery`
- **低熵却做出非法动作（自信犯错）** (`scienceworld__chemistry-mix-paint-secondary-color__s6`, turn=14, `I_empty=0.002`, `H_C=0.002`, `r_t=0.0`, `valid=False`, `succ=False`)
  - CoT: `I need to open the door to the bee hive before interacting with it. The correct command is "open door to bee hive."`
  - Action: `open door to bee hive`

## 8. 交付图表清单（位于 `figures/`）
- `Fig01_E04_bridge_dist.png` ~ `Fig10_E18_teacher_gap.png`（共 10 张，300 DPI）
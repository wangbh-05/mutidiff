# TODO

## 算法流程图

- [ ] **ActionPWD 算法流程图**：单个环境，同一 obs 复制 K 份 → DPP → K 个 action → PWD 计算 → 执行 action[0]
- [ ] **Independent TrajDiversity 算法流程图**：K 个环境，每个独立复制 obs → DPP → 选 action[k] → 分叉。对比 Batched TrajDiversity（K 个不同 obs 混批）

## Results 可视化

- [ ] **ActPWD vs Gamma 折线图**（含 SEM error bar）—— 主结果
- [ ] **DPP vs Temperature vs Noise 柱状对比图**（3 组并排）
- [ ] **Block 终点散点图**：baseline vs DPP，K 条轨迹的 block 终点分布
- [ ] **Per-action[k] 成功率分布**：证明 action[0..7] 地位对称
- [ ] **算法开销 profiling 表**：Tweedie / DPP energy / autograd / projection 各占多少 ms

## Toy Example

- [ ] ✅ `toy_dpp_gradient.png` — DPP 梯度自适应排斥
- [ ] ✅ `toy_evolution.png` — DPP vs 随机噪声 100 步演化
- [ ] ✅ `toy_k_norm.png` — K=2/8/32 归一化对比
- [ ] ✅ `toy_orthogonal.png` — 正交投影分解
- [ ] **新 toy**：「DDIM 风格 2D 去噪 + DPP」示意——点从噪声分布收敛到目标，同时被 DPP 推开
- [ ] **新 toy**：用实际实验数据替代合成数据（K=4/8/16 的 ActPWD 值）

## 论文写作

- [ ] Experiments 章节（已完成初稿）
- [ ] Results 章节（已完成初稿）
- [ ] Introduction 章节
- [ ] Related Work 章节
- [ ] Method 章节（有数学推导，需补充伪代码）
- [ ] Discussion 章节
- [ ] Conclusion 章节

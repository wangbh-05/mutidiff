# Diversity Guidance for Diffusion Policy — 答辩文字稿

---

## 1. 预备知识

### 1.1 Diffusion Policy (DP)

**Diffusion Policy** 是一种基于扩散模型的模仿学习框架。给定机器人观测 $o$，
DP 通过学习从噪声中逐步去噪来生成动作序列 $a$。

**训练**：对干净动作 $a_0$ 逐步加噪至 $a_T \sim \mathcal{N}(0,I)$，
训练噪声预测网络 $\epsilon_\theta(a_t, t, o)$ 最小化 MSE：

$$\mathcal{L} = \mathbb{E}_{t, a_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(a_t, t, o)\|^2 \right]$$

### 1.2 DDIM 采样

训练后的 DDIM 从纯噪声开始确定性去噪（eta=0）：

$$\hat{a}_0 = \frac{a_t - \sqrt{1-\bar{\alpha}_t} \cdot \epsilon_\theta(a_t, t)}{\sqrt{\bar{\alpha}_t}} \quad \text{(Tweedie 公式)}$$

$$a_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \hat{a}_0 + \sqrt{1-\bar{\alpha}_{t-1}} \cdot \epsilon_\theta$$

DDIM 是确定性的：给定初始噪声，总是产生相同的动作。这限制了
多样性——同一个观测下只有一个动作建议。

### 1.3 Push-T 任务

- 圆形推杆（agent）在 512×512 平面上推动 T 形块（block）到目标区域
- 观测：2 帧 × 20 维 keypoint（9 block keypoints × 2D + agent position × 2D）
- 动作：8 帧 × 2 维 agent 位移（dx, dy）
- 成功：block 覆盖目标区域 IoU ≥ 0.9

### 1.4 为什么需要多样性

真实机器人任务中，给定同一个观测可能有多种有效解法：
- 避开左侧障碍物 vs 避开右侧障碍物
- 从上方推 vs 从侧面推 T 形块

标准 Diffusion Policy 只输出一种解。我们需要在**不降低成功率**的前提下
生成多样化的动作建议。

---

## 2. 相关工作

### 2.1 OSCAR (Orthogonal Stochastic Control for Alignment-Respecting Diversity)

OSCAR 在 Flow Matching 推理时通过修改 ODE 来注入多样性：

$$dx_t = [f_\theta(x_t, t) + g_\perp(x_t, t)] dt + \sigma(t) \Pi_\perp(x_t, t) dW_t$$

核心思想：
- **体积最大化**：通过 $\log\det$ 能量函数推动样本在特征空间分散
- **正交投影**：将多样性梯度投影到主方向的正交补上，保护去噪质量
- **随机噪声**：注入正交噪声增加探索（可选）

### 2.2 从 Flow Matching 迁移到 Diffusion 的三个关键修正

| 问题 | OSCAR (Flow Matching) | 本文 (Diffusion) |
|------|----------------------|-------------------|
| 终点估计 | Heun 直线外推 $z_{ep} = z_k + v \cdot (0-t)$ | **Tweedie 公式** |
| 时间值域 | $t \in [0,1]$ | **$t / 100$ 归一化** |
| K 尺度 | 无归一化（K=2 无效，K=32 崩塌） | **$E \leftarrow E/(K-1)$** |

---

## 3. 方法

### 3.1 DPP 多样性能量

对 $K$ 个并行样本的估计干净动作 $Z \in \mathbb{R}^{K \times D}$：

1. **Pairwise 距离**：$D_{ij} = \|Z_i - Z_j\|^2$
2. **Median 归一化**（尺度不变）：$D \leftarrow D / \text{median}(D_{\text{off-diag}})$
3. **高斯核**：$L_{ij} = \exp(-h \cdot D_{ij})$
4. **DPP 得分**：$\mathcal{LL} = \log\det(L) - \log\det(L+I)$
5. **最小化能量**：$E = [\log\det(L+I) - \log\det(L)] / (K-1)$

**物理意义**：$\log\det(L)$ 衡量样本在特征空间的体积。最大化体积 = 最大化多样性。
$-\log\det(L+I)$ 提供有界正则化。$/(K-1)$ 使梯度不随 batch size K 变化。

### 3.2 正交投影

保护去噪质量的核心机制：

$$\Delta a_{\text{base}} = a_{t-1}^{\text{DDIM}} - a_t$$

$$g_\perp^{(k)} = g_d^{(k)} - \omega \cdot \frac{\langle g_d^{(k)}, \Delta a_{\text{base}}^{(k)}\rangle}{\|\Delta a_{\text{base}}^{(k)}\|^2} \cdot \Delta a_{\text{base}}^{(k)}$$

$\omega = 0.95$：保留 5% 平行分量，允许轻微的去噪方向修正。

### 3.3 时间门控

$$\gamma_{\text{eff}} = \gamma \cdot \max\left(0, \min\left(1, \frac{t_{\text{norm}} - t_{\text{gate}}^{\text{end}}}{t_{\text{gate}}^{\text{start}} - t_{\text{gate}}^{\text{end}}}\right)\right)$$

仅在 $t_{\text{norm}} \in (0.05, 0.9]$ 注入 guidance。
- $t > 0.9$：Tweedie 估计不可靠，关闭
- $t < 0.05$：最后精细去噪，关闭

### 3.4 最终更新

$$a_{t-1}^{(k)} = a_{t-1}^{\text{DDIM},(k)} - \gamma_{\text{eff}} \cdot g_\perp^{(k)}$$

### 3.5 伪代码

```
输入: K 个并行样本, N 步 DDIM, 观测 o
For t = T, T-1, ..., 0:
    预测噪声: ε = ε_θ(a_t, t, o)
    估计干净动作: â_0 = Tweedie(a_t, ε, ᾱ_t)
    DDIM 步进: a_{t-1}^{DDIM} = DDIM_step(â_0, ε, ᾱ_{t-1})
    计算 DPP 能量: E = [logdet(L+I) - logdet(L)] / (K-1)
    计算梯度: g_d = ∂E/∂a_t
    正交投影: g_⊥ = g_d - ω·g_∥
    时间门控: γ_eff = γ · gate(t)
    应用: a_{t-1} = a_{t-1}^{DDIM} - γ_eff · g_⊥
返回 â_0
```

### 3.6 关键创新

1. **K 归一化**：$E/(K-1)$，消除 batch size 对有效引导强度的影响
2. **Tweedie 替代 Heun**：扩散模型的正确终点估计
3. **修正的 Trajectory Diversity 评估**：K 个并行环境各执行不同 action[k]
4. **发现 DPP 需要高质量模型**：欠拟合时 guidance 有害
5. **Block 终点多样性 ×2.5**：在全部成功的 seeds 中显著提升

---

## 4. 实验

### 4.1 实验设置

| 参数 | 值 |
|------|-----|
| 模型 | Conditional UNet1D, epoch=550, 100 DDPM 步训练 |
| 推理 | DDIM 16 步, K=8 并行样本 |
| 任务 | Push-T (legacy), max_steps=300 |
| N (主实验) | 200 action seeds + 200 trajectory seeds × 60 configs, 8 GPU ~7h |
| N (最终) | 1100 seeds (baseline + 最佳 DPP), 合并 3 批数据 |

**评估指标**：

- **ActSucc**：K=8 个 action 中执行 action[0]，主路径成功率
- **ActPWD**：同一 obs 下 K 个 16 维 action 向量的 pairwise L2 均值
- **TrajSucc**：K 个并行环境各执行 action[k]，分叉轨迹成功比例
- **MeanPathPWD / FinalPWD / BlockFinalPWD**：轨迹 agent/block 位置的 pairwise 距离

### 4.2 实验网格（60 configs）

| 参数 | 值 |
|------|-----|
| γ | [0, 1, 3, 5, 7, 10, 15] |
| h | [0.5, 1.0, 2.0, 5.0] |
| K | [4, 8] |
| t_gate_start | [0.7, 0.8, 0.9, 1.0] |

### 4.3 核心结果（N=1100）

| Metric | Baseline (γ=0) | DPP (γ=7, h=2.0) | Δ | p |
|--------|---------------|-------------------|------|------|
| **ActSucc** | 0.867 ± 0.010 | **0.876 ± 0.010** | +0.8pp | **0.57 ns** |
| **ActPWD** | 59.3 ± 0.4 | **71.7 ± 0.5** | **+21%** | **<0.001** |
| 95% CI ActSucc | — | — | [−2.0pp, +3.6pp] | — |

**核心结论**：DPP 在无统计可检测的影响（95% CI [−2pp, +3.6pp], p=0.57）下，
将动作空间多样性**显著提升 21%**（p<0.001）。

### 4.4 Gamma Sweep (K=8, h=1.0)

| γ | ActSucc | ActPWD | TrajSucc |
|---|---------|--------|----------|
| 0 | 0.840 | 58.7 | 0.864 |
| 1 | 0.840 | 55.6 | 0.854 |
| 3 | 0.880 | 66.8 | 0.833 |
| 5 | 0.875 | 87.7 | 0.798 |
| 7 | 0.805 | 108.5 | 0.712 |
| 10 | 0.770 | 130.1 | 0.636 |
| 15 | 0.645 | 150.4 | 0.493 |

γ ∈ [3, 5] 是成功率最优区间。γ=7 提供最佳 diversity-quality trade-off。

### 4.5 Full Grid K=8（γ × h）

| γ | h=0.5 | h=1.0 | h=2.0 | h=5.0 |
|---|-------|-------|-------|-------|
| 1 | 0.810/56.7 | 0.840/55.6 | 0.855/55.7 | 0.845/56.9 |
| 3 | 0.815/79.1 | 0.880/66.8 | 0.870/55.3 | 0.855/56.8 |
| 5 | 0.775/104 | **0.875/87.7** | 0.865/59.8 | 0.865/56.6 |
| 7 | 0.765/129 | 0.805/108 | **0.900/73.5** | 0.855/55.4 |
| 10 | 0.725/144 | 0.770/130 | 0.820/86.2 | 0.840/54.6 |
| 15 | 0.675/163 | 0.645/150 | 0.855/111 | 0.865/55.2 |

**h=2.0, γ=7 为全局最优**（ActSucc 0.900, N=200 时）。h=5.0 使 guidance 失效
（PWD≈baseline）。

### 4.6 带宽 h 的影响

物理机制：$L_{ij} = \exp(-h \cdot D_{ij}/\text{median})$

- **h=0.5**：长程排斥，远距离样本也受力 → PWD 高但成功低
- **h=2.0**：sweet spot，近邻充分排斥，远邻自然衰减
- **h=5.0**：核衰减太快，仅极近样本对有效 → 梯度消失

### 4.7 t_gate_start Sweep

| t_gate_start | ActSucc | ActPWD |
|-------------|---------|--------|
| 0.7 | 0.850 | 72.7 |
| 0.8 | 0.870 | 69.7 |
| **0.9** | **0.905** | 65.6 |
| 1.0 | 0.840 | 61.7 |

t_gate_start=0.9 最优——验证了"关闭前 10% 高噪声阶段 guidance"的假设。

### 4.8 Block 终点多样性

在全部 K=8 rollout 成功的 seeds 子集中：

| Config | BlockFinalPWD | p |
|--------|--------------|------|
| Baseline | 8.7 | — |
| **DPP γ=7 h=2.0** | **21.5** | **<0.001** |

**DPP 将 block 终点位置多样性提升了 2.5 倍**——证明了 guidance 产生了物理上
有意义的、而非仅仅是动作空间的多样性。

### 4.9 欠拟合模型验证

| Epoch | Baseline Succ | DPP Succ | Baseline PWD |
|-------|-------------|----------|-------------|
| 160 (~0.5) | 0.430 | **0.310** (−12pp) | 109.9 |
| 180 (~0.8) | 0.540 | 0.490 (−5pp) | 116.7 |
| 550 (well-trained) | 0.867 | **0.876** (+0.8pp) | 59.3 |

关键发现：**Guidance 只在高质量模型上有效。** 欠拟合模型 baseline PWD ≈110
（well-trained 的 2 倍），模型本身噪声大。此时 DPP 梯度在错误方向上推开样本，
反而降低成功率。这验证了 guidance 的设计前提：需要模型先有足够的确定性。

### 4.10 纯噪声 Baseline (N=300 each)

| η | ActSucc | ActPWD |
|---|---------|--------|
| 0.3 | 0.810 | 59.1 |
| 0.5 | 0.870 | 58.5 |
| 1.0 | 0.807 | 58.4 |

纯正交随机噪声**不产生任何 PWD 增益**（≈baseline 59）。DPP 的结构化梯度
（基于行列式的数据驱动排斥）是不可替代的。

### 4.11 动作空间可视化 (PCA)

PCA 箭头图显示：DPP 将样本以**放射状**推开——"拥挤"的样本对受到更大的力，
"离群"的样本几乎不受力。这是 $L^{-1}$ 的自然行为：行列式对最近点对最敏感。

### 4.12 完整统计汇总

| 实验 | N | ActPWD Δ | p | ActSucc Δ | p |
|------|---|---------|-----|----------|-----|
| 主实验 | 200 | +14.8 | <0.001 | +6.0pp | 0.075 |
| P0 合并 | 500 | +12.2 | <0.001 | +3.0pp | 0.168 |
| **N=1100 最终** | **1100** | **+12.4** | **<0.001** | **+0.8pp** | **0.57** |
| Undertrain ep=160 | 100 | −5.5 | 0.018 | −12pp | 0.080 |
| Block 8/8 成功 | 54/50 | — | — | — | **<0.001** |

---

## 5. 讨论

### 5.1 主要贡献

1. 将 DPP 多样性引导成功迁移到 Diffusion Policy 推理
2. 提出 K 归一化 $E/(K-1)$，解决梯度爆炸问题
3. 设计修正的轨迹多样性评估协议
4. 在大规模实验（N=1100, 60 configs）上严格验证方法的有效性

### 5.2 DPP vs OSCAR

DPP 的 median 归一化和高斯核在 diffusion 场景下远比 OSCAR 稳健。
OSCAR 对特征尺度敏感，K 归一化后完全失效。

### 5.3 MeanPathPWD 为何不变

所有 config 的 MeanPathPWD ≈149——DDIM 噪声本身产生 ~149 的 agent 路径
spread。Guidance 在此基础上的增益被环境动力学和模型重规划消解。
但 **Block 终点多样性**（4.8 节）证明 DPP 产生的多样性是物理上有意义的。

### 5.4 局限性

1. **单任务**：仅在 Push-T 上验证
2. **单模型**：一个 checkpoint
3. **Block 多样性 N 较小**：8/8 成功子集仅 50-54 seeds
4. **无真实机器人验证**

---

## 6. 未来发展

1. **多任务验证**：Kitchen, Block Push Multimodal, Robomimic
2. **真实机器人**：UR5 + RealSense 平台
3. **在线选择**：训练一个 value function 从 K 个候选中挑选最优
4. **自适应 h**：根据观测动态调整核带宽
5. **理论分析**：从 $\det$ 的 K 依赖性严格推导最优归一化因子

---

## 7. 参考文献

1. Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", RSS 2023.
2. Song et al., "Denoising Diffusion Implicit Models", ICLR 2021.
3. OSCAR: "Orthogonal Stochastic Control for Alignment-Respecting Diversity", 2024.
4. Kulesza & Taskar, "Determinantal Point Processes for Machine Learning", 2012.
5. Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023.

---

## 8. 代码

所有代码位于 `standard_dpp/`：

| 文件 | 功能 |
|------|------|
| `diverse_guidance.py` | 核心模块：DPP/OSCAR 能量、正交投影 |
| `eval_paper.py` | 主实验（60 configs, 8 GPU） |
| `eval_p0.py` | P0 统计力提升（+300 seeds, N=500） |
| `eval_n1100.py` | N=1100 追加实验 |
| `eval_undertrain.py` | 欠拟合模型验证 |
| `eval_block_diversity.py` | DPP vs 纯噪声 block 终点多样性 |
| `viz_paper_final.py` | 论文图表生成 |
| `viz_traj_compare.py` | 全量轨迹对比可视化 |
| `viz_action_space.py` | 动作空间 PCA 可视化 |

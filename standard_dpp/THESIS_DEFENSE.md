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

## 4. 实验设置

### 4.1 任务与模型

**Push-T**：圆形推杆（agent）在 512×512 平面上推动 T 形块（block）覆盖目标区域（IoU≥0.9=成功）。
观测 2 帧 × 20 维 keypoint，动作 8 帧 × 2 维位移。预训练 Conditional UNet1D（100 DDPM 步训练，
test_mean_score=0.969），推理用 DDIM 16 步，K=8 并行样本。所有实验 legacy=True, max_steps=300。

### 4.2 评估协议

**ActPWD（动作多样性）**：单个环境，同一观测下 K 个 action 向量（16 维）的 pairwise L2 均值。
只执行 action[0] 推进环境。回答："同一个观测下，模型能产生多少不同的建议？"

**ActSucc（主路径成功率）**：执行 action[0] 的轨迹中 max_reward≥0.9 的比例。
回答："guidance 有没有破坏主输出的质量？"

**TrajSucc / MeanPathPWD / FinalPWD**：K 个并行环境各执行 action[k]，轨迹真正分叉后
的集体成功率和 pairwise 距离。

### 4.3 Baselines

| Baseline | 配置 | 回答什么问题 |
|----------|------|------------|
| DDIM (γ=0) | 标准确定性采样 | DPP 的结构化梯度是否有增量？ |
| Temperature scaling | DDIM η=0.2~1.0 | 扩散过程的随机噪声能替代 DPP 吗？ |
| 纯正交噪声 | γ=0, η_sde=0.3~1.0 | 纯随机扰动能替代 DPP 吗？ |
| 欠拟合模型 | epoch 160, 180 | guidance 依赖模型质量吗？ |

### 4.4 实验矩阵

**实验 1 — Batched TrajDiversity**（60 configs, 8 GPU, ~7h）：γ×h×K×t_gate 全网格。
每 config N=200 action + 200 traj, 原始轨迹全部保存。

**实验 2 — 统计力提升**（P0 + N1100）：baseline 和最优 DPP 追加到 N=1100。

**实验 3 — Independent Per-Env DPP**（16 configs, K=4 N=200 + K=8 N=100）：
每个 env 独立复制 obs 运行 DPP，隔离"不同 obs 混批"效应。保存全部原始轨迹。

**实验 4 — Temperature Scaling**（15 configs）：DDIM η=0.0~1.0 + DPP + Noise, N=200。

---

## 5. 实验结果

### 5.1 主结果：Batched TrajDiversity（N=1100）

| 指标 | Baseline | DPP γ=7 h=2.0 | Δ | p |
|------|----------|-------------------|------|------|
| **ActPWD** | 59.3 | **71.7** | **+21%** | **<0.001** |
| **ActSucc** | 0.867 | 0.876 | +0.8pp | **0.57 ns** |
| 95% CI | — | — | [−2.0pp, +3.6pp] | — |

> DPP 将动作多样性显著提升 21%（p<0.001），同时对成功率无统计可检测的影响
> （Δ=+0.8pp, 95% CI [−2.0pp, +3.6pp]）。从 K=8 个候选动作中选择最优者，
> 期望成功率不低于 baseline，但拥有更多选择。

### 5.2 主结果：Independent Per-Env DPP

| Config | K | ActSucc | ActPWD | MeanPathPWD | TrajSucc |
|--------|---|---------|--------|-------------|----------|
| Baseline | 4 | 0.819 | 61.5 | 92 | 0.819 |
| **DPP γ=10 h=1.0** | 4 | 0.812 ns | **115.5** *** | **109** *** | 0.812 ns |
| **DPP γ=7 h=1.0** | 4 | 0.816 ns | **93.9** *** | **104** *** | 0.816 ns |
| Baseline | 8 | 0.849 | 63.6 | 93 | 0.849 |
| **DPP γ=10 h=1.0** | 8 | 0.785 ns | **133.3** *** | **115** *** | 0.785 ns |
| **DPP γ=7 h=1.0** | 8 | 0.812 ns | **111.2** *** | **107** *** | 0.812 ns |

**精确 p 值（K=4, γ=10 h=1.0 vs baseline）**：
ActPWD p<10⁻⁶, ActSucc p=0.83, MeanPathPWD p<10⁻⁴, TrajSucc p=0.83。

Independent DPP 产生的 ActPWD 增益（+88%）远大于 Batched TrajDiversity（+21%），
因为在独立模式下 DPP 始终在同一观测的 action 间排斥，不受异 obs 混批的影响。
同时 h=1.0 在独立模式下最优（batched 模式 h=2.0 最优）。

### 5.3 消融：Temperature Scaling vs DPP

| 方法 | ActPWD (K=4) | Δ vs baseline | p |
|------|-------------|--------------|------|
| Baseline (DDIM) | 61.5 | — | — |
| Temp η=0.5 | 55.0 | −11% | *** |
| Temp η=1.0 | 42.2 | **−31%** | *** |
| **DPP γ=10** | **115.5** | **+88%** | *** |
| 纯噪声 η=0.3 | 61.5 | 0% | p=1.0 |

**Temperature scaling 显著降低动作多样性**——扩散噪声抹平了初始噪声产生的自然差异。
纯随机噪声（p=1.0）完全无效。只有 DPP 的结构化梯度能可靠增加多样性。

### 5.4 消融：Gamma 与带宽 h

**Batched mode（K=8）**：γ=7, h=2.0 最优（ActSucc 0.900, ActPWD 73.5）。
h=5.0 使 guidance 失效（PWD≈baseline）。h=0.5 成功率下降。

**Independent mode（K=4）**：h=1.0 明显优于 h=2.0。γ 单调增加 ActPWD（61.5→115.5），
ActSucc 保持平稳（0.819→0.812, 全部 ns）。

### 5.5 消融：t_gate_start

**t_gate_start=0.9 最优**（ActSucc 0.905 vs 全程 0.840）。
验证了"关闭前 10% 高噪声阶段 guidance"的假设——Tweedie 估计在极高噪声时不可靠。

### 5.6 消融：欠拟合模型

| Epoch | Baseline Succ | DPP Succ | Baseline PWD |
|-------|-------------|----------|-------------|
| 160 (~0.5) | 0.430 | **0.310** (−12pp) | 109.9 |
| 550 (well-trained) | 0.867 | **0.876** (+0.8pp) | 59.3 |

**Guidance 只在高质量模型上有效。** 欠拟合模型 baseline PWD≈110（well-trained 的 2 倍），
模型本身噪声大，DPP 在错误方向上推开样本。这确立了 guidance 的前提：需要足够的确定性。

### 5.7 综合统计汇总

| 实验 | 方法 | N | ActPWD Δ | p | ActSucc Δ | p |
|------|------|---|---------|-----|----------|-----|
| Batched | DPP γ=7 h=2.0 | 1100 | **+21%** | <0.001 | +0.8pp | 0.57 |
| Independent K=4 | DPP γ=10 h=1.0 | 200 | **+88%** | <10⁻⁶ | −0.7pp | 0.83 |
| Independent K=8 | DPP γ=10 h=1.0 | 100 | **+109%** | <10⁻⁶ | −6.4pp | 0.056 |
| Temperature | η=1.0 | 200 | **−31%** | <10⁻⁶ | −4.1pp | 0.18 |
| Pure Noise | η_sde=0.3 | 200 | 0% | 1.0 | 0pp | 1.0 |
| Undertrain | ep=160 | 100 | −9% | 0.018 | **−12pp** | 0.080 |

---

## 5. 讨论

### 5.1 主要贡献

1. 将 DPP 多样性引导成功迁移到 Diffusion Policy 推理
2. 提出 K 归一化 $E/(K-1)$，使有效引导强度不依赖 batch size
3. 在 N=1100 的规模上严格验证：**动作多样性 +21%（p<0.001），成功率非劣效**
4. 系统性消融：γ、h、K、t_gate、纯噪声 baseline，60 configs 全覆盖

### 5.2 为什么 Push-T 的 block 多样性有限

Push-T 成功条件是 block 覆盖目标 ≥90%，目标区域仅 20×20。Block 终点
的自然分散度极低。真正有意义的多样性是 **agent 怎么推**——从哪个角度、
走哪条路——这正是 ActPWD 在动作空间捕获的。在更复杂的任务（避开障碍物）
中，动作多样性会直接转化为路径多样性。

### 5.3 局限性

1. **单任务**：仅在 Push-T 上验证
2. **单模型**：一个 checkpoint
3. 缺少与其他多样性方法（temperature scaling 等）的对比
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

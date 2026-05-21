# DPP Diversity Guidance Experiment — Complete Specification

## 1. What is being tested

The experiment injects a diversity gradient into diffusion model sampling,
then measures whether parallel samples (a) differ from each other in action space,
and (b) lead to genuinely different trajectories in the environment.

**Core hypothesis**: The DPP orthogonal diversity gradient pushes samples apart
in directions orthogonal to the denoising path, increasing diversity without
degrading task success.

**Tested variables** (`eval_dpp_fixed.py`):
- K = batch size (number of parallel samples): [4, 8, 16]
- γ = guidance strength: [0.0, 0.3, 0.5, 1.0]

**Fixed parameters**: DPP energy, DDIM 16 steps, ω = 0.95, h = 1.0, η = 0,
Push-T pretrained checkpoint (test_mean_score=0.969), legacy=True, max_steps=300.

---

## 2. Algorithm: How guidance modifies each DDIM step

Location: `diffusion_policy/policy/diverse_guidance.py`, function `diverse_guidance_step()` (line 198).

### Step 2.1 — Tweedie estimate of clean action

```
Â_0 = (A_t − √(1−ᾱ_t) · ε_θ) / √ᾱ_t
```

Code (line 239–246):
```python
sqrt_alpha = alpha_prod_t.sqrt()
sqrt_one_minus = (1.0 - alpha_prod_t).sqrt()
x_0_hat = (sample - sqrt_one_minus * model_output) / sqrt_alpha
```

`model_output` is the noise prediction ε_θ(A_t, t).  `alpha_prod_t` = ᾱ_t.

### Step 2.2 — Diversity energy

**DPP energy** (line 80–131):
```
Z = flatten(Â_0)                    # (K, D) where D = horizon × action_dim
D_ij = ||Z_i − Z_j||²              # squared pairwise distances
D ← D / median(D)                   # scale invariance
L_ij = exp(−h · D_ij)               # Gaussian kernel matrix (K × K)
E = [log det(L+I) − log det(L)] / (K−1)
```

The `log det(L)` term pushes samples apart (maximising volume).
The `−log det(L+I)` term provides a bounded regulariser.
Division by (K−1) makes the effective gradient independent of batch size K.

Code (line 108–131):
```python
K = Z.shape[0]
D = _pairwise_sq_distances(Z)
off_diag = D[~torch.eye(K, ...)]
med = off_diag.median().clamp(min=eps)
D = D / med
L = torch.exp(-h * D)
L_reg = L + eps * I_K
L_plus_I_reg = L_reg + I_K
energy = torch.logdet(L_plus_I_reg) - torch.logdet(L_reg)
if K > 1:
    energy = energy / (K - 1)
```

### Step 2.3 — Gradient w.r.t. A_t

```
g_d = ∂E/∂A_t = (∂E/∂Â_0) · (∂Â_0/∂A_t)
```

Since Â_0 = (A_t − √(1−ᾱ_t)ε)/√ᾱ_t and ε is detached,
∂Â_0/∂A_t = 1/√ᾱ_t · I.

Code (line 251–290):
```python
with torch.enable_grad():
    sample_grad = sample.detach().requires_grad_(True)
    x_0_hat_grad = (sample_grad - sqrt_one_minus * model_output.detach()) / sqrt_alpha
    Z = x_0_hat_grad.reshape(B, -1)
    energy = compute_dpp_energy(Z, h=..., eps=..., use_median_norm=True)
    g_d = torch.autograd.grad(energy, sample_grad)[0]
```

### Step 2.4 — Orthogonal projection

```
ΔA_base = A_{t-1}^{DDIM} − A_t          # base DDIM step direction
α_k = ⟨g_d[k], ΔA_base[k]⟩ / ‖ΔA_base[k]‖²
g_∥[k] = α_k · ΔA_base[k]               # parallel component
g_⊥[k] = g_d[k] − ω · g_∥[k]            # orthogonal component
```

Code (line 292–296):
```python
base_direction = prev_sample - sample
g_orthogonal = orthogonal_projection(g_d, base_direction, config.ortho_coeff)
```

### Step 2.5 — Apply to DDIM output

```
A_{t-1} = A_{t-1}^{DDIM} − γ_eff · g_⊥
```

where `γ_eff = γ · max(0, min(1, (t_norm−t_gate_end)/(t_gate_start−t_gate_end)))`.

Code (line 298–319):
```python
frac = (t_norm - config.t_gate_end) / (config.t_gate_start - config.t_gate_end)
frac = max(0.0, min(1.0, frac))
gamma_eff = config.gamma * frac
prev_sample = prev_sample - gamma_eff * g_orthogonal
```

### Integration into DDIM loop

`diffusion_unet_lowdim_policy.py`, line 83–112:
```python
for t in scheduler.timesteps:
    trajectory[condition_mask] = condition_data[condition_mask]
    model_output = model(trajectory, t, ...)
    sample_before = trajectory                               # A_t
    trajectory = scheduler.step(...).prev_sample              # A_{t-1}^{DDIM}
    if self.diverse_config is not None:
        trajectory = diverse_guidance_step(
            sample=sample_before, model_output=model_output,
            prev_sample=trajectory,
            alpha_prod_t=scheduler.alphas_cumprod[t.item()],
            t_norm=t.item() / scheduler.config.num_train_timesteps,
            config=self.diverse_config, prediction_type='epsilon')
```

---

## 3. Evaluation Metrics — Exact definitions

### 3.1 Action Pairwise Distance (Action PWD)

**What**: Mean pairwise L2 distance between K action vectors, averaged over all
timesteps in a rollout, then averaged over N=30 test seeds.

**Per-timestep computation** (`eval_dpp_fixed.py`, line 61–64):
```python
actions = result['action']                          # (K, n_action_steps, Da)
acts_flat = actions.reshape(K, -1).float()          # (K, D) where D = 8×2 = 16
dists = [norm(acts_flat[i] - acts_flat[j])
         for i in range(K) for j in range(i+1, K)]  # K(K-1)/2 pairs
pwd_t = mean(dists)                                 # scalar per timestep
```

**Mathematically**:
```
PWD_step = (2/(K(K−1))) · Σ_{i<j} ‖a_i − a_j‖₂
           where a_i ∈ ℝ^{n_action_steps × action_dim} flattened to ℝ^D
PWD = (1/T) · Σ_{t=1}^T PWD_step^{(t)}
```

This measures diversity of **proposed action sequences** at a single observation.

**Interpretation**:
- PWD ≈ 55–60 with γ = 0: natural diversity from DDIM initial noise alone.
- Higher PWD = guidance is pushing actions further apart in Euclidean action space.
- PWD is invariant to whether those differences translate to environment outcomes.

### 3.2 Success Rate

**What**: Fraction of rollouts where `max_reward ≥ 0.9`.

**Reward**: IoU between T-shaped block and goal region (0 to 1 continuous).

**Per-seed** (`eval_dpp_fixed.py`, line 54–56):
```python
max_r = max(max_r, float(reward))
success = 1.0 if max_r >= 0.9 else 0.0
```

**Mathematically**:
```
Success = (1/N) · Σ_{i=1}^N 𝟙[max_{t} reward^{(i)}(t) ≥ 0.9]
```
where N = 30 independent test seeds.

### 3.3 Corrected Trajectory Diversity

**Old (broken)** (`eval_diverse_guidance.py`):
```python
for k in range(K):
    torch.manual_seed(seed * 1000 + k)
    r = run_rollout(policy, seed)   # always executes action[0]
```
→ K rollouts from same start state with different DDIM noise, all executing
action[0].  This ONLY measures sensitivity to DDIM initial noise.

**New (corrected)** (`eval_dpp_fixed.py`, function `run_traj_diversity`):
```python
# K parallel environments, same start seed
envs = [make_env(seed) for _ in range(K)]
for _ in range(MAX_STEPS):
    obs_batch = stack([obs_k for env_k])     # (K, To, Do*2)
    actions = policy.predict_action(obs_batch)  # diversity guidance across K envs
    for k in range(K):
        envs[k].step(actions[k])             # env_k executes action[k] !!
```

**Key difference**: `env_k` executes `action[k]`, not `action[0]`. The guidance
batch contains K different observations (trajectories have diverged), so the
diversity gradient pushes truly different states apart.

#### 3.3.1 Final Position Pairwise Distance (FinalPWD)

**What**: Pairwise L2 between the K agent final positions (at the last step of
each rollout).

```python
final_positions = [agent_trajs[k][-1][:2] for k in range(K)]   # (x, y) each
final_dists = [norm(final_positions[i] - final_positions[j])
               for i < j]
FinalPWD = mean(final_dists)
```

**Unit**: pixels on the 96×96 Push-T board.

**Interpretation**:
- Small FinalPWD (~10): trajectories end at nearly the same place despite
  potentially different action sequences. Environment dynamics absorb differences.
- Large FinalPWD (~70): trajectories end at genuinely different positions.
  Diversity in action space translates to diversity in state space.

#### 3.3.2 Trajectory Path Pairwise Distance (PathPWD)

**What**: Full path similarity measured as L2 between concatenated agent
position sequences.

```python
traj_vec_k = concatenate([pos_t[:2] for pos_t in agent_trajs[k]])  # all (x,y) over time
PathPWD = mean([norm(traj_vec_i - traj_vec_j) for i < j])
```

**Unit**: pixels (cumulative over all timesteps, so usually 1000+).

#### 3.3.3 Trajectory Success Rate (TrajSucc)

**What**: Fraction of K parallel rollouts that succeed (from the same start
state).

```python
successes_k = [1.0 if max_rewards[k] >= 0.9 else 0.0 for k in range(K)]
TrajSucc = mean(successes_k)
TrajSucc_std = std(successes_k)    # reported as ±
```

**Per-seed, averaged over N_traj = 5 trajectory seeds**.

---

## 4. Data provenance

Each number in the results table comes from:

| Metric | N (samples) | Source |
|--------|------------|--------|
| Action success rate | 30 independent test seeds | `eval_dpp_fixed.py:run_action_rollout()` |
| Action PWD | 30 seeds × T steps each | same function, averaged over seeds |
| TrajSucc | 5 seeds × K parallel envs each | `eval_dpp_fixed.py:run_traj_diversity()` |
| FinalPWD | 5 seeds × K(K−1)/2 pairwise | same function, final positions |
| PathPWD | 5 seeds × K(K−1)/2 pairwise | same function, full position sequences |

---

## 5. Key result and its interpretation

```
K=8 Baseline (γ=0):  ActPWD=59.2, FinalPWD=12.7
K=8 DPP (γ=1.0):     ActPWD=57.8, FinalPWD=67.2
```

Action-space PWD is nearly identical, but DPP produces **5.3× larger spread
in final agent positions**.

This means:
1. Baseline DDIM noise creates action differences that the environment absorbs
   (different actions → same outcome).
2. DPP guidance creates action differences in *task-relevant directions* that the
   environment amplifies (different actions → genuinely different outcomes).
3. The orthogonal projection is doing its job: perturbations are ⊥ to the
   denoising direction and therefore do not degrade success (both 0.87).

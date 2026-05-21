"""Action space visualization: PCA of K=8 action vectors + per-step quiver."""
import sys, os, pickle, numpy as np, torch, hydra, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(__file__))
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusion_policy.policy.diverse_guidance import DiverseGuidanceConfig
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

CKPT = "/home/wbh/shared/RTC_dp/diffusion_policy/data/pretrained/pusht_lowdim_dp_cnn/epoch=0550-test_mean_score=0.969.ckpt"
DEV, K, SEED = "cuda:0", 8, 100000
OUT = os.path.join(os.path.dirname(__file__), "eval_output_dg")

def load_policy():
    pld = torch.load(CKPT, map_location=DEV, pickle_module=__import__('dill'))
    sc = DDIMScheduler(num_train_timesteps=100, beta_start=0.0001, beta_end=0.02,
                       beta_schedule='squaredcos_cap_v2', clip_sample=True,
                       set_alpha_to_one=True, steps_offset=0, prediction_type='epsilon')
    p = hydra.utils.instantiate(pld['cfg'].policy.copy(), noise_scheduler=sc, num_inference_steps=16)
    p.load_state_dict(pld['state_dicts']['model']); p.to(DEV).eval()
    return p

def make_env(seed):
    kw = PushTKeypointsEnv.genenerate_keypoint_manager_params()
    return MultiStepWrapper(PushTKeypointsEnv(legacy=True, keypoint_visible_rate=1.0, **kw), 2, 8, 300)

def collect_actions(policy, seed):
    """Collect K=8 action vectors at each step of a rollout."""
    env = make_env(seed); policy.reset(); obs = env.reset()
    all_actions = []  # list of (K, 16) arrays per step
    for _ in range(300):
        if obs.ndim == 2: obs = obs[None, ...]
        Do = obs.shape[-1] // 2
        np_obs = obs[:, :2, :Do].astype(np.float32)
        np_b = np.tile(np_obs, (K, 1, 1))
        with torch.no_grad():
            result = policy.predict_action({'obs': torch.from_numpy(np_b).to(DEV)})
        actions = result['action'].cpu().numpy()  # (K, 8, 2)
        all_actions.append(actions.reshape(K, -1))  # (K, 16)
        a0 = actions[0]  # (8, 2)
        obs, rew, done, _ = env.step(a0)
        if done: break
    env.close()
    return np.array(all_actions)  # (T, K, 16)

policy = load_policy()
torch.manual_seed(SEED)

# Collect actions for baseline and DPP
print("Collecting action data...")
policy.diverse_config = None
bl_actions = collect_actions(policy, SEED)  # (T_bl, K, 16)
policy.diverse_config = DiverseGuidanceConfig(energy_type='dpp', gamma=7, dpp_h=2.0, ortho_coeff=0.95)
dpp_actions = collect_actions(policy, SEED)  # (T_dpp, K, 16)
print(f"Baseline: {bl_actions.shape[0]} steps, DPP: {dpp_actions.shape[0]} steps")

# ============================================================================
# FIGURE 1: PCA of action vectors at the FIRST STEP
# ============================================================================
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 7))
fig1.suptitle('Action Space Diversity — PCA of K=8 Action Vectors (step 0)', fontsize=14, fontweight='bold')

# Joint PCA
all_vecs = np.vstack([bl_actions[0], dpp_actions[0]])  # (2K, 16)
pca = PCA(n_components=2).fit(all_vecs)

for idx, (name, actions, color) in enumerate([
    ('Baseline (gamma=0)', bl_actions[0], 'gray'),
    ('DPP gamma=7 h=2.0', dpp_actions[0], 'steelblue'),
]):
    ax = axes1[idx]
    vecs_2d = pca.transform(actions)  # (K, 2)

    # Scatter
    ax.scatter(vecs_2d[:, 0], vecs_2d[:, 1], c=color, s=120, edgecolors='black', linewidth=0.8, zorder=5)
    for k in range(K):
        ax.annotate(f'{k}', (vecs_2d[k, 0]+0.03, vecs_2d[k, 1]+0.03), fontsize=10, fontweight='bold')

    # Pairwise distances in PCA space
    pwd = np.mean([np.linalg.norm(vecs_2d[i] - vecs_2d[j])
                   for i in range(K) for j in range(i+1, K)])
    # Pairwise distances in original space
    pwd_orig = np.mean([np.linalg.norm(actions[i] - actions[j])
                        for i in range(K) for j in range(i+1, K)])

    ax.set_title(f'{name}\nOriginal PWD={pwd_orig:.1f}  PCA PWD={pwd:.2f}', fontsize=12)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)')
    ax.grid(True, alpha=0.2)
    ax.axhline(0, color='black', lw=0.5, alpha=0.3); ax.axvline(0, color='black', lw=0.5, alpha=0.3)

plt.tight_layout()
fig1.savefig(os.path.join(OUT, 'action_pca_step0.png'), dpi=150, bbox_inches='tight')
print("Saved action_pca_step0.png")

# ============================================================================
# FIGURE 2: Action time series — K=8 samples, step 0, dx/dy over 8 frames
# ============================================================================
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))
fig2.suptitle('Action Time Series — K=8 Action Chunks (step 0)', fontsize=14, fontweight='bold')

for col, (name, actions, color) in enumerate([
    ('Baseline (gamma=0)', bl_actions[0], 'gray'),
    ('DPP gamma=7 h=2.0', dpp_actions[0], 'steelblue'),
]):
    acts_reshaped = actions.reshape(K, 8, 2)  # (K, 8, 2)
    frames = np.arange(8)

    # Row 0: dx over 8 frames for each sample
    ax = axes2[0, col]
    for k in range(K):
        ax.plot(frames, acts_reshaped[k, :, 0], 'o-', lw=1.5, markersize=5, alpha=0.7,
                label=f'k={k}')
    ax.set_title(f'{name} — dx (action x-component)')
    ax.set_xlabel('Frame'); ax.set_ylabel('dx'); ax.grid(True, alpha=0.2)

    # Row 1: dy over 8 frames for each sample
    ax = axes2[1, col]
    for k in range(K):
        ax.plot(frames, acts_reshaped[k, :, 1], 's--', lw=1.5, markersize=5, alpha=0.7,
                label=f'k={k}')
    ax.set_title(f'{name} — dy (action y-component)')
    ax.set_xlabel('Frame'); ax.set_ylabel('dy'); ax.grid(True, alpha=0.2)

axes2[0, 0].legend(fontsize=7, ncol=2, loc='upper right')
plt.tight_layout()
fig2.savefig(os.path.join(OUT, 'action_timeseries.png'), dpi=150, bbox_inches='tight')
print("Saved action_timeseries.png")

# ============================================================================
# FIGURE 3: PCA with arrows showing how guidance pushes each sample
# ============================================================================
fig3, ax3 = plt.subplots(figsize=(10, 8))

# Joint PCA on both
all_first = np.vstack([bl_actions[0], dpp_actions[0]])
pca_all = PCA(n_components=2).fit(all_first)
bl_2d = pca_all.transform(bl_actions[0])   # (K, 2)
dpp_2d = pca_all.transform(dpp_actions[0])  # (K, 2)

ax3.scatter(bl_2d[:, 0], bl_2d[:, 1], c='gray', s=150, edgecolors='black', linewidth=0.8, zorder=5, label='Baseline (gamma=0)')
ax3.scatter(dpp_2d[:, 0], dpp_2d[:, 1], c='steelblue', s=150, edgecolors='black', linewidth=0.8, zorder=5, label='DPP gamma=7 h=2.0')

# Arrows from baseline -> DPP for each k
for k in range(K):
    ax3.annotate('', xy=dpp_2d[k], xytext=bl_2d[k],
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5, alpha=0.6))
    ax3.annotate(f'{k}', bl_2d[k] + np.array([0.03, 0.03]), fontsize=10, fontweight='bold', color='gray')
    ax3.annotate(f'{k}', dpp_2d[k] + np.array([0.03, 0.03]), fontsize=10, fontweight='bold', color='steelblue')

ax3.set_title('Guidance Effect on Action Space\n(arrows = how DPP pushes each sample from baseline)', fontsize=13, fontweight='bold')
ax3.set_xlabel(f'PC1 ({pca_all.explained_variance_ratio_[0]*100:.0f}%)')
ax3.set_ylabel(f'PC2 ({pca_all.explained_variance_ratio_[1]*100:.0f}%)')
ax3.legend(fontsize=11); ax3.grid(True, alpha=0.2)

plt.tight_layout()
fig3.savefig(os.path.join(OUT, 'action_pca_arrows.png'), dpi=150, bbox_inches='tight')
print("Saved action_pca_arrows.png")

# ============================================================================
# FIGURE 4: Per-step PWD over the rollout (how diversity evolves over time)
# ============================================================================
fig4, ax4 = plt.subplots(figsize=(12, 5))

def compute_pwd(act_array):
    """act_array: (T, K, 16)"""
    T = act_array.shape[0]
    pwds = []
    for t in range(T):
        d = [np.linalg.norm(act_array[t, i] - act_array[t, j])
             for i in range(K) for j in range(i+1, K)]
        pwds.append(np.mean(d))
    return np.array(pwds)

bl_pwd_t = compute_pwd(bl_actions)
dpp_pwd_t = compute_pwd(dpp_actions)

ax4.plot(bl_pwd_t, '-', color='gray', lw=2, label=f'Baseline (mean={bl_pwd_t.mean():.1f})')
ax4.plot(dpp_pwd_t, '-', color='steelblue', lw=2, label=f'DPP gamma=7 h=2.0 (mean={dpp_pwd_t.mean():.1f})')
ax4.set_xlabel('Rollout Step', fontsize=12); ax4.set_ylabel('Action PWD', fontsize=12)
ax4.set_title('Action Diversity Over Time (per-step pairwise distance)', fontsize=13, fontweight='bold')
ax4.legend(fontsize=11); ax4.grid(True, alpha=0.2)

fig4.tight_layout()
fig4.savefig(os.path.join(OUT, 'action_pwd_timeseries.png'), dpi=150, bbox_inches='tight')
print("Saved action_pwd_timeseries.png")
print("\nDone!")

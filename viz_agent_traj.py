"""Visualize agent trajectories for NO guidance, DPP, and pure noise."""
import sys, os, numpy as np, torch, hydra, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(__file__))
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusion_policy.policy.diverse_guidance import DiverseGuidanceConfig
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

CKPT = "/home/wbh/shared/RTC_dp/diffusion_policy/data/pretrained/pusht_lowdim_dp_cnn/epoch=0550-test_mean_score=0.969.ckpt"
DEV = "cuda:0"; BOARD = 512; GOAL = (256, 256)

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

def collect(policy, seed, K):
    envs = [make_env(seed) for _ in range(K)]; policy.reset()
    obs_list = [env.reset() for env in envs]
    max_rewards, dones = [0.0]*K, [False]*K
    agent_trajs = [[] for _ in range(K)]
    for _ in range(300):
        obs_batch = np.stack(obs_list); Do = obs_batch.shape[-1] // 2
        with torch.no_grad():
            result = policy.predict_action({'obs': torch.from_numpy(
                obs_batch[:, :2, :Do].astype(np.float32)).to(DEV)})
        for k in range(K):
            if dones[k]: continue
            obs_new, reward, done, info = envs[k].step(result['action'][k].cpu().numpy())
            obs_list[k] = obs_new
            rv = float(reward) if np.isscalar(reward) else float(np.asarray(reward).flat[0])
            max_rewards[k] = max(max_rewards[k], rv); dones[k] = done
            agent_trajs[k].append(obs_new[1, 18:20].copy() if obs_new.ndim == 2 else obs_new[18:20].copy())
        if all(dones): break
    for env in envs: env.close()
    return agent_trajs, [1.0 if mr >= 0.9 else 0.0 for mr in max_rewards]

policy = load_policy(); K = 4; seed = 100000
COLORS = plt.cm.tab10(np.linspace(0, 1, K))

configs = [
    ("NO guidance (gamma=0)", None),
    ("DPP gamma=1.0", DiverseGuidanceConfig(energy_type="dpp", gamma=1.0, ortho_coeff=0.95)),
    ("PURE NOISE eta=0.3 (gamma=0)", DiverseGuidanceConfig(energy_type="dpp", gamma=0.0, eta_sde=0.3, ortho_coeff=0.95)),
]

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle("Agent Trajectories -- K=4 Parallel Rollouts, seed=100000", fontsize=14, fontweight="bold")

for idx, (name, dc) in enumerate(configs):
    ax = axes[idx]
    policy.diverse_config = dc; torch.manual_seed(seed)
    agent_trajs, successes = collect(policy, seed, K)

    ax.set_xlim(0, BOARD); ax.set_ylim(0, BOARD); ax.set_aspect("equal")
    ax.set_title(f"{name}\n{int(sum(successes))}/{K} succeeded", fontsize=12, fontweight="bold")
    ax.set_facecolor("white")
    ax.add_patch(FancyBboxPatch((GOAL[0]-10, GOAL[1]-10), 20, 20,
                 boxstyle="round,pad=0.1", fill=True, facecolor="lightgreen",
                 edgecolor="green", lw=2, alpha=0.4))
    ax.annotate("GOAL", GOAL, fontsize=9, ha="center", va="center", fontweight="bold", color="darkgreen")

    for k in range(K):
        agent = np.array(agent_trajs[k]); succ = successes[k]; c = COLORS[k]
        ls = "-"; lw = 2.5
        if len(agent) > 1:
            ax.plot(agent[:, 0], agent[:, 1], ls, color=c, lw=lw, alpha=0.9)
            ax.scatter(agent[0, 0], agent[0, 1], color=c, marker="s", s=50, zorder=5, edgecolors="black", linewidth=0.8)
            mk = "o" if succ else "X"
            ax.scatter(agent[-1, 0], agent[-1, 1], color=c, marker=mk, s=80, zorder=6, edgecolors="black", linewidth=0.8)
        tag = f"k={k} OK" if succ else f"k={k} FAIL"
        ax.annotate(tag, agent[-1] + np.array([2, 2]), fontsize=8, color="black", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    if idx == 1: ax.set_xlabel("X (pixels)", fontsize=11)
    if idx == 0: ax.set_ylabel("Y (pixels)", fontsize=11)
    ax.grid(True, alpha=0.15)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "eval_output_dg", "agent_traj_verify.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")

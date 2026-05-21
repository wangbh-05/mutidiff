"""
Verify DPP fixes: K-normalised energy + corrected trajectory diversity.

Fixes applied:
  1. DPP energy divided by (K-1) — same effective γ regardless of K
  2. Trajectory diversity: K parallel envs, env_k executes action[k]

Compares: baseline vs DPP γ=[0.3, 0.5, 1.0] at K=[4, 8, 16]
          30 action seeds + 5 trajectory seeds
"""
import sys, os, time, json, argparse, numpy as np, torch, hydra
sys.path.insert(0, os.path.dirname(__file__))
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusion_policy.policy.diverse_guidance import DiverseGuidanceConfig
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

CKPT = "/home/wbh/shared/RTC_dp/diffusion_policy/data/pretrained/pusht_lowdim_dp_cnn/epoch=0550-test_mean_score=0.969.ckpt"
DEVICE_DEFAULT = "cuda:0"
LEGACY, MAX_STEPS, OBS_STEPS, ACT_STEPS = True, 300, 2, 8
DDIM_STEPS = 16
N_ACTION, N_TRAJ = 30, 5
TEST_SEED = 100000
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "eval_output_dg")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_policy(device):
    pld = torch.load(CKPT, map_location=device, pickle_module=__import__('dill'))
    sc = DDIMScheduler(num_train_timesteps=100, beta_start=0.0001, beta_end=0.02,
                       beta_schedule='squaredcos_cap_v2', clip_sample=True,
                       set_alpha_to_one=True, steps_offset=0, prediction_type='epsilon')
    p = hydra.utils.instantiate(pld['cfg'].policy.copy(), noise_scheduler=sc, num_inference_steps=DDIM_STEPS)
    p.load_state_dict(pld['state_dicts']['model']); p.to(device).eval()
    return p

def make_env(seed):
    kw = PushTKeypointsEnv.genenerate_keypoint_manager_params()
    return MultiStepWrapper(PushTKeypointsEnv(legacy=LEGACY, keypoint_visible_rate=1.0, **kw),
                            OBS_STEPS, ACT_STEPS, MAX_STEPS)

# ============================================================================
# 1. Action diversity (unchanged — same as before for comparison)
# ============================================================================
def run_action_rollout(policy, seed, K):
    env = make_env(seed); policy.reset(); obs = env.reset()
    max_r = 0.0; all_dists = []
    for _ in range(MAX_STEPS):
        if obs.ndim == 2: obs = obs[None, ...]
        Do = obs.shape[-1] // 2
        np_obs = obs[:, :OBS_STEPS, :Do].astype(np.float32)
        np_batch = np.tile(np_obs, (K, 1, 1))
        with torch.no_grad(): result = policy.predict_action({'obs': torch.from_numpy(np_batch).to(policy.device)})
        actions = result['action']
        a0 = actions[0].cpu().numpy()
        acts_f = actions.reshape(K, -1).float()
        dists = [torch.norm(acts_f[i]-acts_f[j]).item() for i in range(K) for j in range(i+1, K)]
        all_dists.append(np.mean(dists) if dists else 0.0)
        obs, reward, done, info = env.step(a0)
        max_r = max(max_r, float(reward) if np.isscalar(reward) else float(np.asarray(reward).flat[0]))
        if done: break
    env.close()
    return {'max_reward': max_r, 'success': 1.0 if max_r >= 0.9 else 0.0,
            'mean_pwd': float(np.mean(all_dists)) if all_dists else 0.0,
            'all_pwds': [float(d) for d in all_dists]}

# ============================================================================
# 2. CORRECTED Trajectory diversity — K parallel envs, env_k gets action[k]
# ============================================================================
def run_traj_diversity(policy, seed, K):
    """
    Corrected trajectory diversity:
      - K parallel environments, same start state
      - At each step: stack K observations → batch → DPP guidance → K actions
      - env_k executes action[k]  (NOT action[0]!)
      - Trajectories naturally diverge
    """
    # Create K identical environments
    envs = [make_env(seed) for _ in range(K)]
    policy.reset()

    # Reset all
    obs_list = [env.reset() for env in envs]
    max_rewards = [0.0] * K
    dones = [False] * K
    all_position_trajs = [[] for _ in range(K)]  # record block positions
    all_action_trajs = [[] for _ in range(K)]

    for _ in range(MAX_STEPS):
        # Stack observations from all K envs → (K, To, Do*2)
        obs_batch = np.stack(obs_list)
        Do = obs_batch.shape[-1] // 2
        np_obs = obs_batch[:, :OBS_STEPS, :Do].astype(np.float32)
        obs_dict = {'obs': torch.from_numpy(np_obs).to(policy.device)}

        with torch.no_grad():
            result = policy.predict_action(obs_dict)

        actions = result['action']  # (K, n_action_steps, Da)

        # Each env_k executes action[k]
        for k in range(K):
            if dones[k]:
                continue
            a_k = actions[k].cpu().numpy()
            obs_new, reward, done, info = envs[k].step(a_k)
            obs_list[k] = obs_new

            reward_val = float(reward) if np.isscalar(reward) else float(np.asarray(reward).flat[0])
            max_rewards[k] = max(max_rewards[k], reward_val)
            dones[k] = done

            # Record agent position from the LATEST observation frame
            # obs_new shape: (To=2, Do*2=40)
            #   obs_new[1, 0:18] = block keypoints (9 × 2D)
            #   obs_new[1, 18:20] = agent position (x, y)
            #   obs_new[1, 20:40] = visibility mask
            if obs_new.ndim == 2:
                agent_pos = obs_new[1, 18:20].copy()
                block_center = obs_new[1, 0:18].reshape(9, 2).mean(axis=0)
            else:
                # Fallback for single-frame (should not happen with MultiStepWrapper)
                agent_pos = obs_new[18:20].copy()
                block_center = obs_new[0:18].reshape(9, 2).mean(axis=0)
            all_position_trajs[k].append(np.concatenate([agent_pos, block_center]))
            all_action_trajs[k].append(a_k.copy())

        if all(dones):
            break

    for env in envs:
        env.close()

    successes = [1.0 if mr >= 0.9 else 0.0 for mr in max_rewards]
    success_rate = float(np.mean(successes))

    # ---- Trajectory pairwise diversity metrics ----
    # Each stored entry: [agent_x, agent_y, block_center_x, block_center_y]
    #
    # Metric 1: FinalPWD — pairwise L2 of final agent positions
    # Metric 2: MeanPathPWD — mean over timesteps of agent L2, interpolated to N points
    # Metric 3: MaxPathPWD  — maximum agent L2 at any aligned timestep (worst-case divergence)

    N_INTERP = 100  # interpolate all trajectories to 100 equally-spaced points

    # Interpolate each trajectory to N_INTERP points along its duration
    interp_agents = []
    for traj in all_position_trajs:
        if len(traj) < 2:
            interp_agents.append(np.zeros((N_INTERP, 2)))
            continue
        arr = np.array(traj)                     # (T_k, 4)
        T_k = len(arr)
        t_orig = np.linspace(0, 1, T_k)
        t_new = np.linspace(0, 1, N_INTERP)
        # Interpolate each coordinate independently
        interp = np.zeros((N_INTERP, 2))
        for d in range(2):                       # agent_x, agent_y only
            interp[:, d] = np.interp(t_new, t_orig, arr[:, d])
        interp_agents.append(interp)

    # Compute pairwise distances using interpolated trajectories
    final_dists = []
    mean_path_dists = []
    max_path_dists = []
    for i in range(K):
        for j in range(i+1, K):
            agent_i = interp_agents[i]           # (N_INTERP, 2)
            agent_j = interp_agents[j]

            # Per-timestep L2 distances
            step_dists = np.linalg.norm(agent_i - agent_j, axis=1)  # (N_INTERP,)

            # Final position distance (last interpolated point)
            final_dists.append(float(step_dists[-1]))

            # Mean over full trajectory
            mean_path_dists.append(float(np.mean(step_dists)))

            # Maximum divergence at any point
            max_path_dists.append(float(np.max(step_dists)))

    return {
        'success_rate': success_rate,
        'success_std': float(np.std(successes)),
        'max_rewards': [float(mr) for mr in max_rewards],
        'successes': successes,
        'final_pwd': float(np.mean(final_dists)) if final_dists else 0.0,
        'final_pwd_std': float(np.std(final_dists)) if final_dists else 0.0,
        'mean_path_pwd': float(np.mean(mean_path_dists)) if mean_path_dists else 0.0,
        'max_path_pwd': float(np.mean(max_path_dists)) if max_path_dists else 0.0,
        'per_seed_final_dists': final_dists,
        'per_seed_mean_path_dists': mean_path_dists,
        'per_seed_max_path_dists': max_path_dists,
    }


# ============================================================================
# Main evaluation
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    device = f"cuda:{args.gpu}"

    policy = load_policy(device)
    log(f"DPP FIXED experiment — K-normalised energy")

    # Configs to test
    configs = [
        # (name, K, gamma, h)
        ("Baseline_K=4", 4, 0.0, 1.0),
        ("DPP_K=4_g=0.3", 4, 0.3, 1.0),
        ("DPP_K=4_g=1.0", 4, 1.0, 1.0),
        ("Baseline_K=8", 8, 0.0, 1.0),
        ("DPP_K=8_g=0.3", 8, 0.3, 1.0),
        ("DPP_K=8_g=0.5", 8, 0.5, 1.0),
        ("DPP_K=8_g=1.0", 8, 1.0, 1.0),
        ("Baseline_K=16", 16, 0.0, 1.0),
        ("DPP_K=16_g=0.3", 16, 0.3, 1.0),
        ("DPP_K=16_g=1.0", 16, 1.0, 1.0),
    ]

    all_results = {}
    for cfg_name, K, gamma, h in configs:
        dc = None if gamma == 0.0 else DiverseGuidanceConfig(
            energy_type="dpp", gamma=gamma, dpp_h=h, ortho_coeff=0.95, eta_sde=0.0, t_gate_end=0.05)
        policy.diverse_config = dc

        t0 = time.time()
        log(f"\n{'='*50}")
        log(f"  {cfg_name}  (K={K}, γ={gamma})")
        log(f"{'='*50}")

        # --- Action diversity ---
        action_results = []
        for i in range(N_ACTION):
            seed = TEST_SEED + i
            torch.manual_seed(seed)
            r = run_action_rollout(policy, seed, K)
            action_results.append(r)
            if (i+1) % 10 == 0:
                succ = sum(x['success'] for x in action_results)
                avg_d = np.mean([x['mean_pwd'] for x in action_results])
                log(f"    action [{i+1}/{N_ACTION}] succ={succ/(i+1):.3f} pwd={avg_d:.2f}")

        act_succ = sum(x['success'] for x in action_results) / N_ACTION
        act_reward = float(np.mean([x['max_reward'] for x in action_results]))
        act_pwd = float(np.mean([x['mean_pwd'] for x in action_results if x['mean_pwd'] > 0]))

        # --- CORRECTED Trajectory diversity ---
        traj_results = []
        for i in range(N_TRAJ):
            seed = TEST_SEED + i
            torch.manual_seed(seed)
            r = run_traj_diversity(policy, seed, K)
            traj_results.append(r)
            log(f"    traj seed={seed}: succ={r['success_rate']:.2f}±{r['success_std']:.2f}  "
                f"final_pairwise={r['final_pairwise_dist_mean']:.3f}  traj_pairwise={r['traj_pairwise_dist_mean']:.1f}")

        traj_succ = float(np.mean([t['success_rate'] for t in traj_results]))
        traj_final_pwd = float(np.mean([t['final_pwd'] for t in traj_results]))
        traj_mean_path = float(np.mean([t['mean_path_pwd'] for t in traj_results]))
        traj_max_path = float(np.mean([t['max_path_pwd'] for t in traj_results]))

        elapsed = time.time() - t0
        log(f"  [{cfg_name}] DONE {elapsed:.0f}s: "
            f"act_succ={act_succ:.3f} act_pwd={act_pwd:.2f} "
            f"traj_succ={traj_succ:.3f} final={traj_final_pwd:.1f} mean_path={traj_mean_path:.1f} max_path={traj_max_path:.1f}")

        all_results[cfg_name] = {
            'K': K, 'gamma': gamma, 'h': h,
            'action': {'success_rate': act_succ, 'avg_reward': act_reward, 'avg_pwd': act_pwd},
            'trajectory': {
                'success_rate': traj_succ, 'final_pwd': traj_final_pwd,
                'mean_path_pwd': traj_mean_path, 'max_path_pwd': traj_max_path,
                'per_seed': [{'seed': TEST_SEED+i, 'success_rate': t['success_rate'],
                              'final_pwd': t['final_pwd'],
                              'mean_path_pwd': t['mean_path_pwd'],
                              'max_path_pwd': t['max_path_pwd']}
                             for i, t in enumerate(traj_results)],
            },
        }

    # Save & summary
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(OUTPUT_DIR, f"dpp_fixed_{ts}_gpu{args.gpu}.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"\nSaved to {out_path}")

    # Summary table
    log("\n" + "="*80)
    log("SUMMARY — DPP FIXED (K-normalised + corrected traj diversity)")
    log("="*80)
    hdr = f"{'Config':<25} {'K':>3} {'γ':>5} {'ActSucc':>8} {'ActPWD':>8} {'TrajSucc':>9} {'Final':>7} {'MeanPath':>9} {'MaxPath':>8}"
    print(hdr); print("-"*len(hdr))
    for cfg_name, r in all_results.items():
        t = r['trajectory']
        print(f"{cfg_name:<25} {r['K']:>3} {r['gamma']:>5.1f} "
              f"{r['action']['success_rate']:>7.3f} {r['action']['avg_pwd']:>8.2f} "
              f"{t['success_rate']:>8.3f} {t['final_pwd']:>7.1f} "
              f"{t['mean_path_pwd']:>9.1f} {t['max_path_pwd']:>8.1f}")

    log("DONE")

if __name__ == '__main__':
    main()

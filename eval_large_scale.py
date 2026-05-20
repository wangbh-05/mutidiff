"""
Large-scale diversity guidance experiment (~2h, 8 GPUs).

Sweeps:
  A: Gamma × Energy      (action N=100 + traj N=10)
  B: Ortho coefficient    (action N=100 + traj N=10 for key points)
  C: K (batch size)       (action N=100)
  D: DDIM steps           (action N=100)
  E: eta_sde (noise)      (action N=100)
  F: DPP bandwidth h      (action N=100)

Usage: python eval_large_scale.py --gpu <N> --group <A|B|C|D|E|F>
"""
import sys, os, time, json, argparse
import numpy as np
import torch, hydra
sys.path.insert(0, os.path.dirname(__file__))
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusion_policy.policy.diverse_guidance import DiverseGuidanceConfig
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

# ---------------------------------------------------------------------------
# Fixed experiment settings
# ---------------------------------------------------------------------------
CKPT = "/home/wbh/shared/RTC_dp/diffusion_policy/data/pretrained/pusht_lowdim_dp_cnn/epoch=0550-test_mean_score=0.969.ckpt"
LEGACY, MAX_STEPS, OBS_STEPS, ACTION_STEPS = True, 300, 2, 8
DDIM_DEFAULT, K_DEFAULT = 16, 8
N_ACTION, N_TRAJ = 100, 10          # action seeds, trajectory seeds
TEST_SEED_START = 100000
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "eval_output_dg")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ---------------------------------------------------------------------------
# Model loading (shared across all configs on a GPU)
# ---------------------------------------------------------------------------
def load_policy(device, ddim_steps=DDIM_DEFAULT):
    payload = torch.load(CKPT, map_location=device, pickle_module=__import__('dill'))
    sc = DDIMScheduler(num_train_timesteps=100, beta_start=0.0001, beta_end=0.02,
                       beta_schedule='squaredcos_cap_v2', clip_sample=True,
                       set_alpha_to_one=True, steps_offset=0, prediction_type='epsilon')
    p = hydra.utils.instantiate(payload['cfg'].policy.copy(), noise_scheduler=sc,
                                 num_inference_steps=ddim_steps)
    p.load_state_dict(payload['state_dicts']['model']); p.to(device).eval()
    return p

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
def make_env(seed):
    kw = PushTKeypointsEnv.genenerate_keypoint_manager_params()
    return MultiStepWrapper(PushTKeypointsEnv(legacy=LEGACY, keypoint_visible_rate=1.0, **kw),
                            OBS_STEPS, ACTION_STEPS, MAX_STEPS)

# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------
def run_rollout(policy, seed, K=K_DEFAULT):
    env = make_env(seed); policy.reset(); obs = env.reset()
    max_r = 0.0; all_dists = []
    for _ in range(MAX_STEPS):
        if obs.ndim == 2: obs = obs[None, ...]
        Do = obs.shape[-1] // 2
        np_obs = obs[:, :OBS_STEPS, :Do].astype(np.float32)
        np_batch = np.tile(np_obs, (K, 1, 1))
        with torch.no_grad():
            result = policy.predict_action({'obs': torch.from_numpy(np_batch).to(policy.device)})
        actions = result['action']           # (K, n_action_steps, Da)
        a0 = actions[0].cpu().numpy()         # (n_action_steps, Da)
        # Pairwise distances
        acts_flat = actions.reshape(K, -1).float()
        dists = [torch.norm(acts_flat[i]-acts_flat[j]).item()
                 for i in range(K) for j in range(i+1, K)]
        all_dists.append(np.mean(dists) if dists else 0.0)
        obs, reward, done, info = env.step(a0)
        max_r = max(max_r, float(reward) if np.isscalar(reward) else float(np.asarray(reward).flat[0]))
        if done: break
    env.close()
    return {'max_reward': max_r, 'success': 1.0 if max_r >= 0.9 else 0.0,
            'mean_pairwise_dist': float(np.mean(all_dists)) if all_dists else 0.0,
            'all_pairwise_dists': [float(d) for d in all_dists]}

# ---------------------------------------------------------------------------
# Evaluate single config
# ---------------------------------------------------------------------------
def eval_config(policy, cfg_name, diverse_config, n_action=N_ACTION, n_traj=N_TRAJ, K=K_DEFAULT):
    t0 = time.time()
    log(f"  [{cfg_name}] starting...")

    # Action diversity
    policy.diverse_config = diverse_config
    action_results = []
    for i in range(n_action):
        seed = TEST_SEED_START + i
        torch.manual_seed(seed)
        r = run_rollout(policy, seed, K=K)
        action_results.append(r)
        if (i+1) % 25 == 0:
            succ = sum(x['success'] for x in action_results)
            log(f"    action [{i+1}/{n_action}] succ={succ/(i+1):.3f}")

    succ_rate = sum(x['success'] for x in action_results) / n_action
    avg_reward = float(np.mean([x['max_reward'] for x in action_results]))
    avg_pwd = float(np.mean([x['mean_pairwise_dist'] for x in action_results if x['mean_pairwise_dist'] > 0]))

    # Trajectory diversity
    traj_results = []
    if n_traj > 0:
        for i in range(n_traj):
            seed = TEST_SEED_START + i
            traj_succ, traj_rew = [], []
            for k in range(K):
                torch.manual_seed(seed * 1000 + k)
                r = run_rollout(policy, seed, K=K)
                traj_succ.append(r['success']); traj_rew.append(r['max_reward'])
            traj_results.append({'seed': seed, 'success_rate': float(np.mean(traj_succ)),
                'success_std': float(np.std(traj_succ)),
                'reward_mean': float(np.mean(traj_rew)), 'reward_std': float(np.std(traj_rew))})
        log(f"    traj: succ={np.mean([t['success_rate'] for t in traj_results]):.3f}±{np.mean([t['success_std'] for t in traj_results]):.3f}")

    elapsed = time.time() - t0
    log(f"  [{cfg_name}] DONE in {elapsed:.0f}s: succ={succ_rate:.3f} reward={avg_reward:.3f} div={avg_pwd:.2f}")

    return {'config_name': cfg_name, 'elapsed_s': elapsed,
            'action': {'success_rate': succ_rate, 'avg_reward': avg_reward, 'avg_pairwise_dist': avg_pwd,
                       'per_seed': [{'seed': TEST_SEED_START+i, 'success': float(r['success']),
                                     'reward': float(r['max_reward']), 'pwd': r['mean_pairwise_dist']}
                                    for i, r in enumerate(action_results)]},
            'trajectory': traj_results}

# ---------------------------------------------------------------------------
# Config generators
# ---------------------------------------------------------------------------
def make_configs():
    """Return list of (group, cfg_name, diverse_config_or_None, n_action, n_traj, K, ddim)"""
    cfgs = []

    # === GROUP A: Gamma × Energy ===
    for g in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        dc = None if g == 0.0 else DiverseGuidanceConfig(energy_type="dpp", gamma=g, dpp_h=1.0, ortho_coeff=0.95)
        cfgs.append(('A', f'DPP_gamma={g}', dc, N_ACTION, N_TRAJ, K_DEFAULT, DDIM_DEFAULT))
    for g in [0.1, 0.3, 0.5, 0.7, 1.0, 1.5]:
        dc = DiverseGuidanceConfig(energy_type="oscar", gamma=g, ortho_coeff=0.95)
        cfgs.append(('A', f'OSCAR_gamma={g}', dc, N_ACTION, N_TRAJ, K_DEFAULT, DDIM_DEFAULT))

    # === GROUP B: Ortho coefficient ===
    for ortho in [0.0, 0.3, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0]:
        dc = DiverseGuidanceConfig(energy_type="dpp", gamma=1.0, dpp_h=1.0, ortho_coeff=ortho)
        n_t = N_TRAJ if ortho in [0.0, 0.95, 1.0] else 0
        cfgs.append(('B', f'DPP_g1.0_ortho={ortho}', dc, N_ACTION, n_t, K_DEFAULT, DDIM_DEFAULT))
    for ortho in [0.5, 0.8, 0.95, 1.0]:
        dc = DiverseGuidanceConfig(energy_type="oscar", gamma=0.3, ortho_coeff=ortho)
        cfgs.append(('B', f'OSCAR_g0.3_ortho={ortho}', dc, N_ACTION, 0, K_DEFAULT, DDIM_DEFAULT))

    # === GROUP C: K (batch size) ===
    for K in [2, 4, 8, 16, 32]:
        for g in [0.0, 0.3, 1.0]:
            dc = None if g == 0.0 else DiverseGuidanceConfig(energy_type="dpp", gamma=g, dpp_h=1.0, ortho_coeff=0.95)
            cfgs.append(('C', f'DPP_gamma={g}_K={K}', dc, N_ACTION, 0, K, DDIM_DEFAULT))

    # === GROUP D: DDIM steps ===
    for ddim in [8, 16, 32, 64]:
        for g in [0.0, 0.3, 1.0]:
            dc = None if g == 0.0 else DiverseGuidanceConfig(energy_type="dpp", gamma=g, dpp_h=1.0, ortho_coeff=0.95)
            cfgs.append(('D', f'DPP_gamma={g}_DDIM={ddim}', dc, N_ACTION, 0, K_DEFAULT, ddim))

    # === GROUP E: eta_sde (orthogonal noise) ===
    for eta in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
        dc = DiverseGuidanceConfig(energy_type="dpp", gamma=1.0, ortho_coeff=0.95, eta_sde=eta)
        cfgs.append(('E', f'DPP_g1.0_eta={eta}', dc, N_ACTION, 0, K_DEFAULT, DDIM_DEFAULT))
    for eta in [0.05, 0.1, 0.3]:
        dc = DiverseGuidanceConfig(energy_type="oscar", gamma=0.3, ortho_coeff=0.95, eta_sde=eta)
        cfgs.append(('E', f'OSCAR_g0.3_eta={eta}', dc, N_ACTION, 0, K_DEFAULT, DDIM_DEFAULT))

    # === GROUP F: DPP bandwidth h ===
    for h in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        dc = DiverseGuidanceConfig(energy_type="dpp", gamma=1.0, dpp_h=h, ortho_coeff=0.95)
        cfgs.append(('F', f'DPP_g1.0_h={h}', dc, N_ACTION, 0, K_DEFAULT, DDIM_DEFAULT))
    for h in [0.1, 0.5, 1.0, 2.0, 5.0]:
        dc = DiverseGuidanceConfig(energy_type="dpp", gamma=0.3, dpp_h=h, ortho_coeff=0.95)
        cfgs.append(('F', f'DPP_g0.3_h={h}', dc, N_ACTION, 0, K_DEFAULT, DDIM_DEFAULT))

    return cfgs

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--group', type=str, required=True, help='A|B|C|D|E|F')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    all_configs = make_configs()
    my_configs = [(g, n, dc, na, nt, K, dd) for g, n, dc, na, nt, K, dd in all_configs if g == args.group]
    if args.end_idx >= 0:
        my_configs = my_configs[args.start_idx:args.end_idx]
    else:
        my_configs = my_configs[args.start_idx:]

    if not my_configs:
        log(f"No configs found for group {args.group}"); return

    log(f"GPU {args.gpu}, Group {args.group}: {len(my_configs)} configs")
    log(f"Settings: legacy={LEGACY}, max_steps={MAX_STEPS}, N_action={N_ACTION}, N_traj={N_TRAJ}")

    policy = load_policy(device, DDIM_DEFAULT)
    log("Policy loaded")

    all_res = {}
    total_start = time.time()
    for i, (group, cfg_name, dc, n_act, n_traj, K, ddim) in enumerate(my_configs):
        # Rebuild policy if DDIM steps changed
        if ddim != DDIM_DEFAULT:
            policy = load_policy(device, ddim)
            log(f"  Reloaded policy with DDIM={ddim}")
        res = eval_config(policy, cfg_name, dc, n_action=n_act, n_traj=n_traj, K=K)
        all_res[cfg_name] = res

        # Save checkpoint every 5 configs
        if (i+1) % 5 == 0:
            ts = time.strftime('%Y%m%d_%H%M%S')
            tmp_path = os.path.join(OUTPUT_DIR, f"ckpt_{args.group}_gpu{args.gpu}_{ts}.json")
            with open(tmp_path, 'w') as f:
                json.dump({'configs_done': list(all_res.keys()), 'results': all_res}, f, indent=2)
            log(f"  Checkpoint saved: {len(all_res)}/{len(my_configs)} done")

    total_elapsed = time.time() - total_start
    log(f"\nGroup {args.group} DONE in {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")

    # Final save
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(OUTPUT_DIR, f"final_{args.group}_gpu{args.gpu}_{ts}.json")
    meta = {'group': args.group, 'gpu': args.gpu, 'legacy': LEGACY, 'max_steps': MAX_STEPS,
            'n_action': N_ACTION, 'n_traj': N_TRAJ, 'total_elapsed_s': total_elapsed,
            'n_configs': len(all_res)}
    with open(out_path, 'w') as f:
        json.dump({'meta': meta, 'results': all_res}, f, indent=2)
    log(f"Saved to {out_path}")

    # Summary table
    log("\n" + "="*70)
    log(f"GROUP {args.group} SUMMARY")
    log("="*70)
    print(f"{'Config':<35} {'Success':>8} {'Reward':>8} {'ActionDiv':>10} {'TrajSucc':>10}")
    print("-"*75)
    for cfg_name, r in all_res.items():
        a = r['action']
        ts = np.mean([t['success_rate'] for t in r['trajectory']]) if r['trajectory'] else float('nan')
        print(f"{cfg_name:<35} {a['success_rate']:>7.3f} {a['avg_reward']:>8.3f} {a['avg_pairwise_dist']:>10.2f} {ts:>10.3f}")

    log("DONE")

if __name__ == '__main__':
    main()

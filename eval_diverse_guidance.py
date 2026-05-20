"""
Evaluate diversity guidance on Push-T pretrained model (multi-GPU parallel).

Usage:
  python eval_diverse_guidance.py --gpu 0 --configs NO_guidance,DPP_g0.3,DPP_g1.0,OSCAR_g0.3
  python eval_diverse_guidance.py --gpu 1 --configs NO_guidance
"""
import sys
import os
import time
import json
import argparse
import numpy as np
import torch
import hydra

sys.path.insert(0, os.path.dirname(__file__))

from omegaconf import OmegaConf
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusion_policy.policy.diverse_guidance import DiverseGuidanceConfig
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

# ============================================================================
# Fixed experiment config (matching training evaluation)
# ============================================================================
CKPT_PATH = "/home/wbh/shared/RTC_dp/diffusion_policy/data/pretrained/pusht_lowdim_dp_cnn/epoch=0550-test_mean_score=0.969.ckpt"
DDIM_STEPS = 16
K = 8  # parallel samples per observation
N_TEST = 30
N_TRAJ_DIV = 3  # subset of seeds for trajectory diversity
TEST_START_SEED = 100000  # matches training eval
MAX_STEPS = 300  # matches training eval
OBS_STEPS = 2
ACTION_STEPS = 8
LEGACY = True  # matches training eval!
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "eval_output_dg")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def log(msg):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


# ============================================================================
# Load model once
# ============================================================================
def load_policy(device):
    payload = torch.load(CKPT_PATH, map_location=device, pickle_module=__import__('dill'))
    cfg_ckpt = payload['cfg']
    state_dicts = payload['state_dicts']
    orig_noise_cfg = cfg_ckpt.policy.noise_scheduler

    scheduler = DDIMScheduler(
        num_train_timesteps=orig_noise_cfg.num_train_timesteps,
        beta_start=orig_noise_cfg.beta_start,
        beta_end=orig_noise_cfg.beta_end,
        beta_schedule=orig_noise_cfg.beta_schedule,
        clip_sample=orig_noise_cfg.clip_sample,
        set_alpha_to_one=True, steps_offset=0,
        prediction_type=orig_noise_cfg.prediction_type,
    )
    policy_cfg = cfg_ckpt.policy.copy()
    policy = hydra.utils.instantiate(
        policy_cfg, noise_scheduler=scheduler, num_inference_steps=DDIM_STEPS,
    )
    policy.load_state_dict(state_dicts['model'])
    policy.to(device).eval()
    return policy


def make_env(seed):
    kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
    env = MultiStepWrapper(
        PushTKeypointsEnv(legacy=LEGACY, keypoint_visible_rate=1.0, **kp_kwargs),
        n_obs_steps=OBS_STEPS, n_action_steps=ACTION_STEPS,
        max_episode_steps=MAX_STEPS,
    )
    env.seed(seed)
    return env


# ============================================================================
# Single rollout
# ============================================================================
def run_rollout(policy, seed, record_all_actions=True):
    """Run one rollout. At each step: replicate obs K times, generate K actions,
    execute action[0], record pairwise distances among K actions."""
    env = make_env(seed)
    policy.reset()
    obs = env.reset()
    max_reward = 0.0
    all_pairwise_dists = []

    for _ in range(MAX_STEPS):
        if obs.ndim == 2:
            obs = obs[None, ...]
        Do = obs.shape[-1] // 2
        np_obs = obs[:, :OBS_STEPS, :Do].astype(np.float32)
        np_obs_batch = np.tile(np_obs, (K, 1, 1))
        obs_dict = {'obs': torch.from_numpy(np_obs_batch).to(policy.device)}

        with torch.no_grad():
            result = policy.predict_action(obs_dict)

        actions = result['action']
        a0 = actions[0].cpu().numpy()

        # Pairwise distances among K actions
        acts_flat = actions.reshape(K, -1).float()
        dists = []
        for i in range(K):
            for j in range(i + 1, K):
                dists.append(torch.norm(acts_flat[i] - acts_flat[j]).item())
        all_pairwise_dists.append(np.mean(dists) if dists else 0.0)

        obs, reward, done, info = env.step(a0)
        reward_val = float(reward) if np.isscalar(reward) else float(np.asarray(reward).flat[0])
        max_reward = max(max_reward, reward_val)
        if done:
            break

    env.close()
    success = 1.0 if max_reward >= 0.9 else 0.0
    return {'max_reward': max_reward, 'success': success,
            'mean_pairwise_dist': np.mean(all_pairwise_dists) if all_pairwise_dists else 0.0}


# ============================================================================
# Run config evaluation
# ============================================================================
def evaluate_config(policy, cfg_name, diverse_config):
    log(f"\n{'='*50}")
    log(f"Config: {cfg_name}  |  GPU: {policy.device}")
    log(f"{'='*50}")

    # ---- Action diversity (N_TEST seeds) ----
    results = []
    for i in range(N_TEST):
        seed = TEST_START_SEED + i
        torch.manual_seed(seed)
        r = run_rollout(policy, seed)
        results.append(r)
        if (i + 1) % 10 == 0:
            succ = sum(x['success'] for x in results)
            avg_d = np.mean([x['mean_pairwise_dist'] for x in results])
            log(f"  [{i+1}/{N_TEST}] success={succ:.0f}/{i+1} ({succ/(i+1):.2f}), avg_pairwise_dist={avg_d:.4f}")

    succ_rate = sum(x['success'] for x in results) / N_TEST
    avg_reward = float(np.mean([x['max_reward'] for x in results]))
    avg_pairwise = float(np.mean([x['mean_pairwise_dist'] for x in results if x['mean_pairwise_dist'] > 0]))
    log(f"  ACTION DIV [{cfg_name}]: success={succ_rate:.2f}, reward={avg_reward:.3f}, pairwise_dist={avg_pairwise:.4f}")

    # ---- Trajectory diversity (N_TRAJ_DIV seeds, K rollout each) ----
    traj_results = []
    for i in range(N_TRAJ_DIV):
        seed = TEST_START_SEED + i
        traj_rewards = []
        traj_success = []
        for k in range(K):
            torch.manual_seed(seed * 1000 + k)
            r = run_rollout(policy, seed)
            traj_rewards.append(r['max_reward'])
            traj_success.append(r['success'])
        traj_results.append({
            'seed': seed,
            'success_rate': float(np.mean(traj_success)),
            'success_std': float(np.std(traj_success)),
            'max_reward_mean': float(np.mean(traj_rewards)),
            'max_reward_std': float(np.std(traj_rewards)),
        })
        log(f"    traj seed={seed}: success={np.mean(traj_success):.2f}±{np.std(traj_success):.2f}, "
            f"reward={np.mean(traj_rewards):.3f}±{np.std(traj_rewards):.3f}")

    return {
        'config_name': cfg_name,
        'action_diversity': {
            'success_rate': succ_rate,
            'avg_max_reward': avg_reward,
            'avg_pairwise_dist': avg_pairwise,
            'per_seed': [{'seed': TEST_START_SEED + i, 'success': float(r['success']),
                         'max_reward': float(r['max_reward']),
                         'mean_pairwise_dist': float(r['mean_pairwise_dist'])}
                        for i, r in enumerate(results)],
        },
        'trajectory_diversity': traj_results,
    }


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--configs', type=str, default='NO_guidance,DPP_g0.3,DPP_g1.0,OSCAR_g0.3')
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    config_names = [c.strip() for c in args.configs.split(',')]

    CONFIG_MAP = {
        "NO_guidance": None,
        "DPP_g0.3": DiverseGuidanceConfig(energy_type="dpp", gamma=0.3, dpp_h=1.0,
                                          ortho_coeff=0.95, eta_sde=0.0, t_gate_end=0.05),
        "DPP_g1.0": DiverseGuidanceConfig(energy_type="dpp", gamma=1.0, dpp_h=1.0,
                                          ortho_coeff=0.95, eta_sde=0.0, t_gate_end=0.05),
        "OSCAR_g0.3": DiverseGuidanceConfig(energy_type="oscar", gamma=0.3,
                                            ortho_coeff=0.95, eta_sde=0.0, t_gate_end=0.05),
    }

    log(f"Starting on {device}, configs={config_names}")
    log(f"Settings: legacy={LEGACY}, max_steps={MAX_STEPS}, DDIM={DDIM_STEPS}, K={K}, N_test={N_TEST}, test_start_seed={TEST_START_SEED}")

    policy = load_policy(device)
    log("Policy loaded")

    all_results = {}
    for cfg_name in config_names:
        diverse_config = CONFIG_MAP[cfg_name]
        policy.diverse_config = diverse_config
        res = evaluate_config(policy, cfg_name, diverse_config)
        all_results[cfg_name] = res

    # Save
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(OUTPUT_DIR, f"results_{timestamp}_gpu{args.gpu}.json")
    output = {
        'config': {'ckpt': CKPT_PATH, 'device': device, 'ddim_steps': DDIM_STEPS,
                   'K': K, 'N_test': N_TEST, 'N_traj_div': N_TRAJ_DIV,
                   'max_steps': MAX_STEPS, 'legacy': LEGACY,
                   'test_start_seed': TEST_START_SEED},
        'results': all_results,
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    log(f"\nSaved to {out_path}")

    # Summary
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    print(f"{'Config':<22} {'Success':>8} {'Reward':>8} {'ActionDiv':>10}")
    print("-" * 50)
    for cfg_name, r in all_results.items():
        ad = r['action_diversity']
        print(f"{cfg_name:<22} {ad['success_rate']:>7.2f} {ad['avg_max_reward']:>8.3f} {ad['avg_pairwise_dist']:>10.4f}")

    log("DONE")


if __name__ == '__main__':
    main()

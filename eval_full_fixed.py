"""
Corrected DPP experiment — multi-GPU support.

Fixes applied:
  1. DPP energy / (K-1) — K-normalised
  2. Corrected trajectory diversity: K parallel envs, env_k executes action[k]
  3. Full trajectory metrics: FinalPWD, MeanPathPWD, MaxPathPWD (interpolation)
  4. Pure noise baseline: gamma=0, eta_sde>0
  5. t_gate_start sweep

Usage (one process per GPU):
  GPU 0: python eval_full_fixed.py --gpu 0 --total_gpus 4
  GPU 1: python eval_full_fixed.py --gpu 1 --total_gpus 4
  ...
  Launcher: python eval_full_fixed.py --launch 4   (auto-launches 4 GPUs)
"""
import sys, os, time, json, argparse, subprocess, numpy as np, torch, hydra
sys.path.insert(0, os.path.dirname(__file__))
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusion_policy.policy.diverse_guidance import DiverseGuidanceConfig
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

CKPT = "/home/wbh/shared/RTC_dp/diffusion_policy/data/pretrained/pusht_lowdim_dp_cnn/epoch=0550-test_mean_score=0.969.ckpt"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "eval_output_dg")
LEGACY, MAX_STEPS, OBS_S, ACT_S, DDIM = True, 300, 2, 8, 16
N_ACTION, N_TRAJ = 50, 20  # 50 action seeds per config, 20 trajectory seeds
TEST_SEED = 100000

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_policy(device):
    pld = torch.load(CKPT, map_location=device, pickle_module=__import__('dill'))
    sc = DDIMScheduler(num_train_timesteps=100, beta_start=0.0001, beta_end=0.02,
                       beta_schedule='squaredcos_cap_v2', clip_sample=True,
                       set_alpha_to_one=True, steps_offset=0, prediction_type='epsilon')
    p = hydra.utils.instantiate(pld['cfg'].policy.copy(), noise_scheduler=sc, num_inference_steps=DDIM)
    p.load_state_dict(pld['state_dicts']['model']); p.to(device).eval()
    return p

def make_env(seed):
    kw = PushTKeypointsEnv.genenerate_keypoint_manager_params()
    return MultiStepWrapper(PushTKeypointsEnv(legacy=LEGACY, keypoint_visible_rate=1.0, **kw),
                            OBS_S, ACT_S, MAX_STEPS)

# ---- Action diversity ----
def run_action(policy, seed, K):
    env = make_env(seed); policy.reset(); obs = env.reset(); mr = 0.0; dists = []
    for _ in range(MAX_STEPS):
        if obs.ndim == 2: obs = obs[None, ...]
        Do = obs.shape[-1] // 2
        np_obs = obs[:, :OBS_S, :Do].astype(np.float32)
        np_b = np.tile(np_obs, (K, 1, 1))
        with torch.no_grad(): result = policy.predict_action({'obs': torch.from_numpy(np_b).to(policy.device)})
        a0 = result['action'][0].cpu().numpy()
        af = result['action'].reshape(K, -1).float()
        d = [torch.norm(af[i]-af[j]).item() for i in range(K) for j in range(i+1, K)]
        dists.append(np.mean(d) if d else 0.0)
        obs, rew, done, _ = env.step(a0)
        mr = max(mr, float(rew) if np.isscalar(rew) else float(np.asarray(rew).flat[0]))
        if done: break
    env.close()
    return {'max_reward': mr, 'success': 1.0 if mr >= 0.9 else 0.0, 'mean_pwd': float(np.mean(dists)) if dists else 0.0}

# ---- Corrected trajectory diversity ----
def run_traj(policy, seed, K):
    envs = [make_env(seed) for _ in range(K)]; policy.reset()
    obs_list = [env.reset() for env in envs]
    mrs, dones = [0.0]*K, [False]*K
    agent_trajs = [[] for _ in range(K)]
    for _ in range(MAX_STEPS):
        obs_batch = np.stack(obs_list); Do = obs_batch.shape[-1] // 2
        with torch.no_grad():
            result = policy.predict_action({'obs': torch.from_numpy(obs_batch[:, :OBS_S, :Do].astype(np.float32)).to(policy.device)})
        for k in range(K):
            if dones[k]: continue
            obs_new, rew, done, info = envs[k].step(result['action'][k].cpu().numpy())
            obs_list[k] = obs_new
            rv = float(rew) if np.isscalar(rew) else float(np.asarray(rew).flat[0])
            mrs[k] = max(mrs[k], rv); dones[k] = done
            agent_trajs[k].append(obs_new[1, 18:20].copy() if obs_new.ndim == 2 else obs_new[18:20].copy())
        if all(dones): break
    for env in envs: env.close()

    successes = [1.0 if mr >= 0.9 else 0.0 for mr in mrs]

    # Interpolate to N points, compute pairwise metrics
    N_I = 100
    interp = []
    for traj in agent_trajs:
        if len(traj) < 2: interp.append(np.zeros((N_I, 2))); continue
        arr = np.array(traj)
        t_old = np.linspace(0, 1, len(arr)); t_new = np.linspace(0, 1, N_I)
        ip = np.zeros((N_I, 2))
        for d in range(2): ip[:, d] = np.interp(t_new, t_old, arr[:, d])
        interp.append(ip)

    fds, mps, mxs = [], [], []
    for i in range(K):
        for j in range(i+1, K):
            sd = np.linalg.norm(interp[i] - interp[j], axis=1)
            fds.append(float(sd[-1])); mps.append(float(np.mean(sd))); mxs.append(float(np.max(sd)))

    return {'success_rate': float(np.mean(successes)), 'success_std': float(np.std(successes)),
            'final_pwd': float(np.mean(fds)), 'mean_path_pwd': float(np.mean(mps)),
            'max_path_pwd': float(np.mean(mxs))}

# ---- All configs ----
def make_configs():
    cfgs = []

    # 1. Gamma sweep (K=8, DPP)
    for g in [0, 1, 2, 3, 5, 7, 10]:
        dc = None if g == 0 else DiverseGuidanceConfig(energy_type="dpp", gamma=g, dpp_h=1.0, ortho_coeff=0.95)
        cfgs.append((f"DPP_K=8_g={g}", dc, 8, N_TRAJ))

    # 2. Pure noise baseline
    for eta in [0.1, 0.3, 0.5]:
        dc = DiverseGuidanceConfig(energy_type="dpp", gamma=0.0, eta_sde=eta, ortho_coeff=0.95)
        cfgs.append((f"NOISE_eta={eta}", dc, 8, N_TRAJ))

    # 3. K-normalisation verify (DPP gamma=5)
    for K in [4, 8, 16]:
        dc = DiverseGuidanceConfig(energy_type="dpp", gamma=5.0, dpp_h=1.0, ortho_coeff=0.95)
        cfgs.append((f"DPP_K={K}_g=5", dc, K, N_TRAJ))
    for K in [4, 8, 16]:
        cfgs.append((f"BASELINE_K={K}", None, K, N_TRAJ))

    # 4. t_gate_start sweep (DPP gamma=5) — action only, secondary sweep
    for tgs in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        dc = DiverseGuidanceConfig(energy_type="dpp", gamma=5.0, dpp_h=1.0, ortho_coeff=0.95, t_gate_start=tgs)
        cfgs.append((f"DPP_g=5_tgate={tgs}", dc, 8, 0))

    # 5. h=2.0 verify (with traj)
    for h in [1.0, 2.0]:
        dc = DiverseGuidanceConfig(energy_type="dpp", gamma=5.0, dpp_h=h, ortho_coeff=0.95)
        cfgs.append((f"DPP_g=5_h={h}", dc, 8, N_TRAJ))

    # 6. Extra: OSCAR best gamma comparison
    for g in [3, 5]:
        dc = DiverseGuidanceConfig(energy_type="oscar", gamma=g, ortho_coeff=0.95)
        cfgs.append((f"OSCAR_K=8_g={g}", dc, 8, N_TRAJ))

    return cfgs

# ---- Run all configs for this GPU ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--total_gpus', type=int, default=1)
    parser.add_argument('--launch', type=int, default=0, help='Auto-launch N GPUs')
    args = parser.parse_args()

    if args.launch > 0:
        # Auto-launch all GPUs as subprocesses
        script = os.path.abspath(__file__)
        procs = []
        for g in range(args.launch):
            cmd = f"/home/wbh/mambaforge/envs/robodiff/bin/python {script} --gpu {g} --total_gpus {args.launch}"
            log_file = os.path.join(OUTPUT_DIR, f"full_fixed_gpu{g}.log")
            with open(log_file, 'w') as lf:
                p = subprocess.Popen(cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT)
            procs.append(p)
            log(f"Launched GPU {g} (PID={p.pid}) -> {log_file}")
        log(f"All {args.launch} GPUs launched. Waiting...")
        for p in procs: p.wait()
        log("ALL DONE")
        return

    device = f"cuda:{args.gpu}"
    all_cfgs = make_configs()

    # Distribute: GPU g gets configs at indices g, g+total, g+2*total, ...
    my_cfgs = [c for i, c in enumerate(all_cfgs) if i % args.total_gpus == args.gpu]

    log(f"GPU {args.gpu}/{args.total_gpus}: {len(my_cfgs)}/{len(all_cfgs)} configs")
    policy = load_policy(device)

    total_start = time.time()
    all_results = {}
    for cfg_idx, (cfg_name, dc, K, n_traj) in enumerate(my_cfgs):
        policy.diverse_config = dc
        t0 = time.time()
        log(f"\n[{cfg_idx+1}/{len(my_cfgs)}] {cfg_name} (K={K}, n_traj={n_traj})")

        # Action diversity
        action_res = []
        for i in range(N_ACTION):
            torch.manual_seed(TEST_SEED + i)
            action_res.append(run_action(policy, TEST_SEED + i, K))
            if (i+1) % 25 == 0:
                s = sum(r['success'] for r in action_res)
                log(f"  action [{i+1}/{N_ACTION}] succ={s/(i+1):.3f}")

        act_succ = sum(r['success'] for r in action_res) / N_ACTION
        act_rew = float(np.mean([r['max_reward'] for r in action_res]))
        act_pwd = float(np.mean([r['mean_pwd'] for r in action_res if r['mean_pwd'] > 0]))

        # Trajectory diversity
        traj_res = []
        for i in range(n_traj):
            torch.manual_seed(TEST_SEED + i)
            traj_res.append(run_traj(policy, TEST_SEED + i, K))
            if (i+1) % 10 == 0 or i == n_traj - 1:
                t = traj_res[-1]
                log(f"  traj [{i+1}/{n_traj}] succ={t['success_rate']:.2f} final={t['final_pwd']:.1f} mean_path={t['mean_path_pwd']:.1f}")

        elapsed = time.time() - t0
        if n_traj > 0:
            t_succ = float(np.mean([t['success_rate'] for t in traj_res]))
            t_final = float(np.mean([t['final_pwd'] for t in traj_res]))
            t_mean = float(np.mean([t['mean_path_pwd'] for t in traj_res]))
            t_max = float(np.mean([t['max_path_pwd'] for t in traj_res]))
            log(f"  DONE {elapsed:.0f}s: act_succ={act_succ:.3f} act_pwd={act_pwd:.1f} traj_succ={t_succ:.3f} final={t_final:.1f} mean_path={t_mean:.1f}")
        else:
            t_succ, t_final, t_mean, t_max = float('nan'), float('nan'), float('nan'), float('nan')
            log(f"  DONE {elapsed:.0f}s: act_succ={act_succ:.3f} act_pwd={act_pwd:.1f}")

        all_results[cfg_name] = {
            'K': K, 'action': {'success_rate': act_succ, 'avg_reward': act_rew, 'avg_pwd': act_pwd},
            'trajectory': {'success_rate': t_succ, 'final_pwd': t_final, 'mean_path_pwd': t_mean, 'max_path_pwd': t_max},
        }

        # Checkpoint every 5
        if (cfg_idx+1) % 5 == 0:
            ts = time.strftime('%Y%m%d_%H%M%S')
            ckpt_path = os.path.join(OUTPUT_DIR, f"full_fixed_ckpt_gpu{args.gpu}_{ts}.json")
            with open(ckpt_path, 'w') as f:
                json.dump(all_results, f, indent=2)

    total_elapsed = time.time() - total_start
    log(f"\nGPU {args.gpu} DONE in {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")

    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(OUTPUT_DIR, f"full_fixed_final_gpu{args.gpu}_{ts}.json")
    with open(out_path, 'w') as f:
        json.dump({'meta': {'gpu': args.gpu, 'elapsed_s': total_elapsed}, 'results': all_results}, f, indent=2)
    log(f"Saved {out_path}")

    # Summary
    log("\n" + "="*90)
    log(f"GPU {args.gpu} SUMMARY")
    log("="*90)
    hdr = f"{'Config':<30} {'K':>3} {'ActSucc':>8} {'ActPWD':>8} {'TrajSucc':>9} {'Final':>7} {'MeanPath':>9} {'MaxPath':>8}"
    print(hdr); print("-"*len(hdr))
    for cfg_name, r in all_results.items():
        a = r['action']; t = r['trajectory']
        print(f"{cfg_name:<30} {r['K']:>3} {a['success_rate']:>7.3f} {a['avg_pwd']:>8.1f} {t['success_rate']:>8.3f} {t['final_pwd']:>7.1f} {t['mean_path_pwd']:>9.1f} {t['max_path_pwd']:>8.1f}")

    log("DONE")

if __name__ == '__main__':
    main()

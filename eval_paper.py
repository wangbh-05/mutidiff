"""
Paper-quality DPP diversity guidance experiment.

Grid: gamma × h, K=8 fixed.
  gamma ∈ [0, 0.5, 1, 2, 3, 5, 7, 10]  (8 values)
  h     ∈ [0.5, 1.0, 2.0, 5.0]          (4 values)
  29 configs (gamma=0 shared baseline)

Plus ablations:
  Pure noise: gamma=0, eta=0.3
  OSCAR: gamma=3, 5

All configs: N_action=100, N_traj=30 (corrected K-parallel trajectory diversity)

Usage:  python eval_paper.py --launch 8   (auto-distribute across 8 GPUs)
"""
import sys, os, time, json, argparse, subprocess, numpy as np, torch, hydra
sys.path.insert(0, os.path.dirname(__file__))
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusion_policy.policy.diverse_guidance import DiverseGuidanceConfig
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

CKPT = "/home/wbh/shared/RTC_dp/diffusion_policy/data/pretrained/pusht_lowdim_dp_cnn/epoch=0550-test_mean_score=0.969.ckpt"
OUT = os.path.join(os.path.dirname(__file__), "eval_output_dg")
LEGACY, MAX_S, OBS_S, ACT_S, DDIM = True, 300, 2, 8, 16
N_ACT, N_TRAJ = 200, 200  # 200 action seeds + 200 trajectory seeds per config
N_RAW = N_TRAJ              # save ALL traj seeds' raw trajectories
TEST_SEED = 100000
K_FIXED = 8

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_policy(dev):
    pld = torch.load(CKPT, map_location=dev, pickle_module=__import__('dill'))
    sc = DDIMScheduler(num_train_timesteps=100, beta_start=0.0001, beta_end=0.02,
                       beta_schedule='squaredcos_cap_v2', clip_sample=True,
                       set_alpha_to_one=True, steps_offset=0, prediction_type='epsilon')
    p = hydra.utils.instantiate(pld['cfg'].policy.copy(), noise_scheduler=sc, num_inference_steps=DDIM)
    p.load_state_dict(pld['state_dicts']['model']); p.to(dev).eval()
    return p

def make_env(seed):
    kw = PushTKeypointsEnv.genenerate_keypoint_manager_params()
    return MultiStepWrapper(PushTKeypointsEnv(legacy=LEGACY, keypoint_visible_rate=1.0, **kw),
                            OBS_S, ACT_S, MAX_S)

# ---- Action diversity ----
def run_action(policy, seed, K):
    env = make_env(seed); policy.reset(); obs = env.reset(); mr = 0.0; dists = []
    for _ in range(MAX_S):
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
    return {'max_reward': mr, 'success': 1.0 if mr >= 0.9 else 0.0,
            'mean_pwd': float(np.mean(dists)) if dists else 0.0,
            'per_step_pwd': [float(d) for d in dists]}  # time series for analysis

# ---- Corrected trajectory diversity (returns metrics + optional raw trajectories) ----
def run_traj(policy, seed, K, save_raw=False):
    envs = [make_env(seed) for _ in range(K)]; policy.reset()
    obs_list = [env.reset() for env in envs]
    mrs, dones = [0.0]*K, [False]*K
    agent_trajs = [[] for _ in range(K)]
    block_trajs = [[] for _ in range(K)]
    for _ in range(MAX_S):
        obs_batch = np.stack(obs_list); Do = obs_batch.shape[-1] // 2
        with torch.no_grad():
            result = policy.predict_action({'obs': torch.from_numpy(obs_batch[:, :OBS_S, :Do].astype(np.float32)).to(policy.device)})
        for k in range(K):
            if dones[k]: continue
            obs_new, rew, done, info = envs[k].step(result['action'][k].cpu().numpy())
            obs_list[k] = obs_new
            rv = float(rew) if np.isscalar(rew) else float(np.asarray(rew).flat[0])
            mrs[k] = max(mrs[k], rv); dones[k] = done
            if obs_new.ndim == 2:
                agent_trajs[k].append(obs_new[1, 18:20].copy())
                block_trajs[k].append(obs_new[1, 0:18].reshape(9, 2).mean(axis=0).copy())
            else:
                agent_trajs[k].append(obs_new[18:20].copy())
                block_trajs[k].append(obs_new[0:18].reshape(9, 2).mean(axis=0).copy())
        if all(dones): break
    for env in envs: env.close()

    successes = [1.0 if mr >= 0.9 else 0.0 for mr in mrs]
    N_I = 100; interp = []
    for traj in agent_trajs:
        if len(traj) < 2: interp.append(np.zeros((N_I, 2))); continue
        arr = np.array(traj); t_old = np.linspace(0, 1, len(arr)); t_new = np.linspace(0, 1, N_I)
        ip = np.zeros((N_I, 2))
        for d in range(2): ip[:, d] = np.interp(t_new, t_old, arr[:, d])
        interp.append(ip)
    fds, mps, mxs = [], [], []
    for i in range(K):
        for j in range(i+1, K):
            sd = np.linalg.norm(interp[i] - interp[j], axis=1)
            fds.append(float(sd[-1])); mps.append(float(np.mean(sd))); mxs.append(float(np.max(sd)))
    metrics = {'success_rate': float(np.mean(successes)), 'success_std': float(np.std(successes)),
            'final_pwd': float(np.mean(fds)), 'mean_path_pwd': float(np.mean(mps)), 'max_path_pwd': float(np.mean(mxs))}
    if save_raw:
        raw = {'agent_trajs': [np.array(t) for t in agent_trajs],
               'block_trajs': [np.array(t) for t in block_trajs],
               'successes': successes, 'max_rewards': [float(mr) for mr in mrs]}
        return {'metrics': metrics, 'raw': raw}
    return {'metrics': metrics, 'raw': None}

# ---- Configs ----
def make_configs():
    cfgs = []
    gammas = [0, 1, 3, 5, 7, 10, 15]
    hs = [0.5, 1.0, 2.0, 5.0]
    Ks = [4, 8]
    for K in Ks:
        for g in gammas:
            for h in hs:
                if g == 0:
                    cfgs.append((f"K={K}_g=0", None, K, N_TRAJ))
                    break  # one baseline per K
                dc = DiverseGuidanceConfig(energy_type="dpp", gamma=g, dpp_h=h, ortho_coeff=0.95)
                cfgs.append((f"K={K}_DPP_g={g}_h={h}", dc, K, N_TRAJ))
    # t_gate sweep (gamma=5, h=2.0, K=8)
    for tgs in [0.7, 0.8, 0.9, 1.0]:
        dc = DiverseGuidanceConfig(energy_type="dpp", gamma=5.0, dpp_h=2.0, ortho_coeff=0.95, t_gate_start=tgs)
        cfgs.append((f"K=8_DPP_g=5_h=2_tgate={tgs}", dc, 8, N_TRAJ))
    # Ablations: pure noise + OSCAR
    for K in Ks:
        cfgs.append((f"K={K}_NOISE_eta=0.3",
                     DiverseGuidanceConfig(energy_type="dpp", gamma=0.0, eta_sde=0.3, ortho_coeff=0.95), K, N_TRAJ))
        cfgs.append((f"K={K}_OSCAR_g=3",
                     DiverseGuidanceConfig(energy_type="oscar", gamma=3, ortho_coeff=0.95), K, N_TRAJ))
        cfgs.append((f"K={K}_OSCAR_g=5",
                     DiverseGuidanceConfig(energy_type="oscar", gamma=5, ortho_coeff=0.95), K, N_TRAJ))
    return cfgs

# ---- Main ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--total_gpus', type=int, default=1)
    parser.add_argument('--launch', type=int, default=0)
    args = parser.parse_args()

    if args.launch > 0:
        script = os.path.abspath(__file__)
        procs = []
        for g in range(args.launch):
            lf = os.path.join(OUT, f"paper_gpu{g}.log")
            p = subprocess.Popen(f"/home/wbh/mambaforge/envs/robodiff/bin/python {script} --gpu {g} --total_gpus {args.launch}",
                                 shell=True, stdout=open(lf, 'w'), stderr=subprocess.STDOUT)
            procs.append(p); log(f"GPU {g} PID={p.pid} -> {lf}")
        log(f"{args.launch} GPUs launched"); [p.wait() for p in procs]; log("ALL DONE")
        return

    if args.gpu is None:
        parser.error("--gpu is required when not using --launch")
    device = f"cuda:{args.gpu}"
    all_cfgs = make_configs()
    my_cfgs = [c for i, c in enumerate(all_cfgs) if i % args.total_gpus == args.gpu]
    log(f"GPU {args.gpu}: {len(my_cfgs)}/{len(all_cfgs)} configs, N_act={N_ACT}, N_traj={N_TRAJ}")
    policy = load_policy(device)

    total_start = time.time(); all_res = {}
    for ci, (cfg_name, dc, K, n_traj) in enumerate(my_cfgs):
        policy.diverse_config = dc; t0 = time.time()
        log(f"\n[{ci+1}/{len(my_cfgs)}] {cfg_name}")

        act_res = []
        for i in range(N_ACT):
            torch.manual_seed(TEST_SEED + i)
            act_res.append(run_action(policy, TEST_SEED + i, K))
            if (i+1) % 50 == 0:
                s = sum(r['success'] for r in act_res)
                log(f"  action [{i+1}/{N_ACT}] succ={s/(i+1):.3f}")
        a_succ = sum(r['success'] for r in act_res) / N_ACT
        a_rew = float(np.mean([r['max_reward'] for r in act_res]))
        a_pwd = float(np.mean([r['mean_pwd'] for r in act_res if r['mean_pwd'] > 0]))
        # Per-seed action data (for statistical analysis)
        a_per_seed = [{'seed': TEST_SEED + i, 'success': r['success'],
                       'max_reward': r['max_reward'], 'mean_pwd': r['mean_pwd'],
                       'per_step_pwd': r['per_step_pwd']}
                      for i, r in enumerate(act_res)]

        traj_res = []
        raw_trajs_for_viz = []  # save first N_RAW seeds' raw trajectories for visualization
        for i in range(n_traj):
            torch.manual_seed(TEST_SEED + i)
            save_raw = (i < N_RAW)
            r = run_traj(policy, TEST_SEED + i, K, save_raw=save_raw)
            traj_res.append(r['metrics'])
            if save_raw:
                raw_trajs_for_viz.append(r['raw'])
            if (i+1) % 10 == 0 or i == n_traj-1:
                t = traj_res[-1]
                log(f"  traj [{i+1}/{n_traj}] succ={t['success_rate']:.2f} final={t['final_pwd']:.0f} mean_path={t['mean_path_pwd']:.0f}")

        t_succ = float(np.mean([t['success_rate'] for t in traj_res]))
        t_final = float(np.mean([t['final_pwd'] for t in traj_res]))
        t_mean = float(np.mean([t['mean_path_pwd'] for t in traj_res]))
        t_max = float(np.mean([t['max_path_pwd'] for t in traj_res]))
        # Per-seed trajectory data
        t_per_seed = [{'seed': TEST_SEED + i, **t} for i, t in enumerate(traj_res)]
        et = time.time() - t0
        log(f"  DONE {et:.0f}s: a_succ={a_succ:.3f} a_pwd={a_pwd:.1f} t_succ={t_succ:.3f} final={t_final:.0f} mean={t_mean:.0f} max={t_max:.0f}")

        # Full config record (everything needed for post-hoc plotting)
        cfg_params = {'energy_type': dc.energy_type if dc else 'none',
                      'gamma': dc.gamma if dc else 0, 'dpp_h': dc.dpp_h if dc else None,
                      'ortho_coeff': dc.ortho_coeff if dc else None,
                      'eta_sde': dc.eta_sde if dc else None,
                      't_gate_start': dc.t_gate_start if dc else None,
                      't_gate_end': dc.t_gate_end if dc else None,
                      'K': K}
        all_res[cfg_name] = {'config': cfg_params, 'K': K, 'elapsed': et,
            'action': {'success_rate': a_succ, 'avg_reward': a_rew, 'avg_pwd': a_pwd,
                       'per_seed': a_per_seed},
            'trajectory': {'success_rate': t_succ, 'final_pwd': t_final,
                           'mean_path_pwd': t_mean, 'max_path_pwd': t_max,
                           'per_seed': t_per_seed}}
        # Save raw trajectories for first 3 seeds (for visualization)
        if raw_trajs_for_viz:
            import pickle
            viz_path = os.path.join(OUT, f"raw_{cfg_name}_gpu{args.gpu}.pkl")
            with open(viz_path, 'wb') as f:
                pickle.dump(raw_trajs_for_viz, f)

        if (ci+1) % 3 == 0:
            ts = time.strftime('%Y%m%d_%H%M%S')
            with open(os.path.join(OUT, f"paper_ckpt_gpu{args.gpu}_{ts}.json"), 'w') as f:
                json.dump(all_res, f, indent=2)

    total_et = time.time() - total_start
    ts = time.strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(OUT, f"paper_final_gpu{args.gpu}_{ts}.json"), 'w') as f:
        json.dump({'meta': {'gpu': args.gpu, 'total_gpus': args.total_gpus,
                            'elapsed_s': total_et, 'n_action': N_ACT, 'n_traj': N_TRAJ,
                            'DDIM': DDIM, 'max_steps': MAX_S,
                            'legacy': LEGACY, 'test_seed_start': TEST_SEED,
                            'ckpt': CKPT},
                   'results': all_res}, f, indent=2)
    log(f"\nGPU {args.gpu} DONE {total_et:.0f}s. Saved.")

    # Summary
    log("\n" + "="*95)
    log(f"GPU {args.gpu} SUMMARY")
    log("="*95)
    hdr = f"{'Config':<25} {'ActSucc':>8} {'ActPWD':>8} {'TrajSucc':>9} {'Final':>7} {'MeanPath':>9} {'MaxPath':>8}"
    print(hdr); print("-"*len(hdr))
    for cfg_name, r in all_res.items():
        a = r['action']; t = r['trajectory']
        print(f"{cfg_name:<25} {a['success_rate']:>7.3f} {a['avg_pwd']:>8.1f} {t['success_rate']:>8.3f} {t['final_pwd']:>7.0f} {t['mean_path_pwd']:>9.0f} {t['max_path_pwd']:>8.0f}")
    log("DONE")

if __name__ == '__main__':
    main()

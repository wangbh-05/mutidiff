"""
Large-scale Independent Per-Env DPP experiment (6h, 8 GPU).

Method: K=4 parallel envs, each independently replicates its own obs K times
and runs DPP within that batch. env_k executes action[k]. This isolates the
PURE DPP effect — at every step, DPP works on K identical observations.

Saves: per-seed success, agent trajectories, block trajectories, action PWD.

Configs (16 total): baseline, DPP gamma×h, temperature, pure noise.
N=300 action seeds each. All raw trajectories for first 50 traj seeds.

Usage: python eval_independent_large.py --launch 8
"""
import sys, os, time, json, argparse, subprocess, pickle, numpy as np, torch, hydra
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusion_policy.policy.diverse_guidance import DiverseGuidanceConfig
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

CKPT = "/home/wbh/shared/RTC_dp/diffusion_policy/data/pretrained/pusht_lowdim_dp_cnn/epoch=0550-test_mean_score=0.969.ckpt"
OUT = os.path.join(os.path.dirname(__file__), "results")
LEGACY, MAX_S, OBS_S, ACT_S, DDIM = True, 300, 2, 8, 16
TEST_SEED = 100000

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

kw = PushTKeypointsEnv.genenerate_keypoint_manager_params()
pld = torch.load(CKPT, map_location='cpu', pickle_module=__import__('dill'))

def load_policy(dev, eta=0.0):
    sc = DDIMScheduler(num_train_timesteps=100, beta_start=0.0001, beta_end=0.02,
                       beta_schedule='squaredcos_cap_v2', clip_sample=True,
                       set_alpha_to_one=True, steps_offset=0, prediction_type='epsilon')
    p = hydra.utils.instantiate(pld['cfg'].policy.copy(), noise_scheduler=sc, num_inference_steps=DDIM)
    p.load_state_dict(pld['state_dicts']['model']); p.to(dev).eval()
    if eta != 0.0:
        orig_step = sc.step
        sc.step = lambda *a, **kw: orig_step(*a, **kw, eta=eta)
    return p

def make_env(seed):
    return MultiStepWrapper(PushTKeypointsEnv(legacy=LEGACY, keypoint_visible_rate=1.0, **kw), OBS_S, ACT_S, MAX_S)

def run_seed(policy, seed, save_raw, K):
    """Independent per-env DPP: each env replicates its own obs K times."""
    envs = [make_env(seed) for _ in range(K)]
    for env in envs: env.seed(seed)
    policy.reset(); obs_list = [env.reset() for env in envs]
    mrs, dones = [0.0]*K, [False]*K
    agent_trajs = [[] for _ in range(K)]
    block_trajs = [[] for _ in range(K)]
    action_pwds = []  # per-step PWD within each env (averaged across K envs)

    for _ in range(MAX_S):
        step_pwds = []
        for k in range(K):
            if dones[k]: continue
            obs_k = obs_list[k]
            if obs_k.ndim == 2: obs_k = obs_k[None, ...]
            Do = obs_k.shape[-1] // 2
            np_b = np.tile(obs_k[:, :OBS_S, :Do].astype(np.float32), (K, 1, 1))
            with torch.no_grad():
                result = policy.predict_action({'obs': torch.from_numpy(np_b).to(policy.device)})
            actions = result['action']  # (K, n_action_steps, Da)
            # Per-env PWD: within this env's K actions
            af = actions.reshape(K, -1).float()
            d = [torch.norm(af[i]-af[j]).item() for i in range(K) for j in range(i+1, K)]
            step_pwds.append(np.mean(d) if d else 0.0)

            a_k = actions[k].cpu().numpy()  # env_k executes action[k]
            obs_new, rew, done, info = envs[k].step(a_k)
            obs_list[k] = obs_new
            rv = float(rew) if np.isscalar(rew) else float(np.asarray(rew).flat[0])
            mrs[k] = max(mrs[k], rv); dones[k] = done
            if obs_new.ndim == 2:
                agent_trajs[k].append(obs_new[1, 18:20].copy())
                block_trajs[k].append(obs_new[1, 0:18].reshape(9, 2).mean(0).copy())
            else:
                agent_trajs[k].append(obs_new[18:20].copy())
                block_trajs[k].append(obs_new[0:18].reshape(9, 2).mean(0).copy())
        action_pwds.append(np.mean(step_pwds) if step_pwds else 0.0)
        if all(dones): break

    for env in envs: env.close()
    successes = [1.0 if mr >= 0.9 else 0.0 for mr in mrs]

    # Interpolate agent trajectories for pairwise metrics
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

    result = {
        'success_rate': float(np.mean(successes)),
        'successes': successes, 'max_rewards': [float(mr) for mr in mrs],
        'action_pwd': float(np.mean(action_pwds)) if action_pwds else 0.0,
        'final_pwd': float(np.mean(fds)), 'mean_path_pwd': float(np.mean(mps)), 'max_path_pwd': float(np.mean(mxs)),
    }
    if save_raw:
        result['agent_trajs'] = [np.array(t) for t in agent_trajs]
        result['block_trajs'] = [np.array(t) for t in block_trajs]
        result['action_pwds_per_step'] = action_pwds
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--total_gpus', type=int, default=1)
    parser.add_argument('--launch', type=int, default=0)
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--N', type=int, default=200)
    parser.add_argument('--gpu_offset', type=int, default=0)
    parser.add_argument('--tag', type=str, default='')
    args = parser.parse_args()

    K = args.K; N_SEEDS = args.N

    if args.launch > 0:
        script = os.path.abspath(__file__)
        procs = []
        for g in range(args.launch):
            gpu_id = g + args.gpu_offset
            tag_str = f"--tag {args.tag}" if args.tag else ""
            lf = os.path.join(OUT, f"indep_K{K}_{args.tag}_gpu{gpu_id}.log" if args.tag else f"indep_K{K}_gpu{gpu_id}.log")
            cmd = f"/home/wbh/mambaforge/envs/robodiff/bin/python {script} --gpu {gpu_id} --total_gpus {args.launch} --K {K} --N {N_SEEDS} {tag_str} --gpu_offset {args.gpu_offset}"
            p = subprocess.Popen(cmd, shell=True, stdout=open(lf, 'w'), stderr=subprocess.STDOUT)
            procs.append(p); log(f"GPU {gpu_id} PID={p.pid} -> {lf}")
        log(f"{args.launch} GPUs launched (K={K}, N={N_SEEDS}, offset={args.gpu_offset})")
        [p.wait() for p in procs]; log("ALL DONE")
        return

    if args.gpu is None: parser.error("--gpu required")
    device = f"cuda:{args.gpu}"

    cfgs = []
    # Baseline
    cfgs.append(("BASELINE_g=0", None, 0.0))
    # DPP
    for g in [3, 5, 7, 10]:
        for h in [1.0, 2.0]:
            dc = DiverseGuidanceConfig(energy_type="dpp", gamma=g, dpp_h=h, ortho_coeff=0.95)
            cfgs.append((f"DPP_g={g}_h={h}", dc, 0.0))
    # Temperature scaling (no DPP, vary DDIM eta)
    for eta in [0.2, 0.5, 0.8, 1.0]:
        cfgs.append((f"TEMP_eta={eta}", None, eta))
    # Pure noise
    for eta_sde in [0.3, 0.5, 1.0]:
        dc = DiverseGuidanceConfig(energy_type="dpp", gamma=0.0, eta_sde=eta_sde, ortho_coeff=0.95)
        cfgs.append((f"NOISE_eta={eta_sde}", dc, 0.0))

    my_cfgs = [c for i, c in enumerate(cfgs) if i % args.total_gpus == args.gpu]
    log(f"GPU {args.gpu}: {len(my_cfgs)}/{len(cfgs)} configs, N={N_SEEDS}, K={K}, raw=ALL")

    curr_eta = None; policy = None; all_res = {}; total_start = time.time()
    for ci, (cfg_name, dc, eta) in enumerate(my_cfgs):
        if eta != curr_eta:
            policy = load_policy(device, eta); curr_eta = eta
        policy.diverse_config = dc; t0 = time.time()
        log(f"\n[{ci+1}/{len(my_cfgs)}] {cfg_name}")

        per_seed = []; raw_list = []
        for i in range(N_SEEDS):
            torch.manual_seed(TEST_SEED + i)
            r = run_seed(policy, TEST_SEED + i, save_raw=True, K=K)
            per_seed.append(r)
            raw_list.append(r)
            if (i+1) % 50 == 0:
                succ = np.mean([s['success_rate'] for s in per_seed])
                pwd = np.mean([s['action_pwd'] for s in per_seed])
                log(f"  [{i+1}/{N_SEEDS}] succ={succ:.3f} actPWD={pwd:.1f}")

        succs = np.array([s['success_rate'] for s in per_seed])
        action_pwds = np.array([s['action_pwd'] for s in per_seed])
        final_pwds = np.array([s['final_pwd'] for s in per_seed])
        mean_paths = np.array([s['mean_path_pwd'] for s in per_seed])

        et = time.time() - t0
        log(f"  DONE {et:.0f}s: succ={succs.mean():.3f}±{succs.std()/np.sqrt(N_SEEDS):.3f} "
            f"actPWD={action_pwds.mean():.1f} finalPWD={final_pwds.mean():.0f} meanPath={mean_paths.mean():.0f}")

        all_res[cfg_name] = {
            'config': {'eta': eta, 'dc': str(dc)},
            'action': {'success_rate': float(succs.mean()), 'success_sem': float(succs.std()/np.sqrt(N_SEEDS)),
                       'action_pwd': float(action_pwds.mean()), 'action_pwd_sem': float(action_pwds.std()/np.sqrt(N_SEEDS)),
                       'per_seed': [{'seed': TEST_SEED+i, 'success_rate': s['success_rate'],
                                     'action_pwd': s['action_pwd'], 'final_pwd': s['final_pwd'],
                                     'mean_path_pwd': s['mean_path_pwd']} for i, s in enumerate(per_seed)]},
            'trajectory': {'final_pwd': float(final_pwds.mean()), 'mean_path_pwd': float(mean_paths.mean())},
        }

        if raw_list:
            raw_save = [{'agent_trajs': r['agent_trajs'], 'block_trajs': r['block_trajs'],
                         'successes': r['successes'], 'action_pwds_per_step': r['action_pwds_per_step']}
                        for r in raw_list]
            with open(os.path.join(OUT, f"indep_raw_{cfg_name}_gpu{args.gpu}.pkl"), 'wb') as f:
                pickle.dump(raw_save, f)

        if (ci+1) % 3 == 0:
            ts = time.strftime('%Y%m%d_%H%M%S')
            with open(os.path.join(OUT, f"indep_large_ckpt_gpu{args.gpu}_{ts}.json"), 'w') as f:
                json.dump(all_res, f, indent=2)

    ts = time.strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(OUT, f"indep_large_gpu{args.gpu}_{ts}.json"), 'w') as f:
        json.dump({'meta': {'gpu': args.gpu, 'N': N_SEEDS, 'K': K, 'elapsed': time.time()-total_start},
                   'results': all_res}, f, indent=2)
    log(f"\nGPU {args.gpu} DONE. Saved.")
    log("DONE")

if __name__ == '__main__':
    main()

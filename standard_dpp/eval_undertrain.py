"""
Test DPP diversity guidance on undertrained checkpoints.
Checkpoints: undertrain_match_pretrain_040139 @ epoch 100, 150, 160, 180, 300
Configs per checkpoint: baseline (gamma=0) + DPP gamma=7 h=2.0

Usage: python eval_undertrain.py --launch 4  (GPUs 4-7)
"""
import sys, os, time, json, argparse, subprocess, pickle, numpy as np, torch, hydra
sys.path.insert(0, os.path.dirname(__file__))
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusion_policy.policy.diverse_guidance import DiverseGuidanceConfig
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

BASE = "/home/wbh/shared/RTC_dp/diffusion_policy/data/outputs/2026.04.07/undertrain_match_pretrain_040139/checkpoints"
OUT = os.path.join(os.path.dirname(__file__), "eval_output_dg")
LEGACY, MAX_S, OBS_S, ACT_S, DDIM = True, 300, 2, 8, 16
K_FIXED, N_ACT, N_TRAJ = 8, 100, 200
TEST_SEED = 100000
EPOCHS = [160, 180]  # succ≈0.5, succ≈0.8

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_policy(dev, epoch):
    ckpt = f"{BASE}/epoch={epoch:04d}.ckpt"
    pld = torch.load(ckpt, map_location=dev, pickle_module=__import__('dill'))
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

def run_action(policy, seed, K):
    env = make_env(seed); policy.reset(); obs = env.reset(); mr = 0.0; dists = []
    for _ in range(MAX_S):
        if obs.ndim == 2: obs = obs[None, ...]
        Do = obs.shape[-1] // 2
        np_obs = obs[:, :OBS_S, :Do].astype(np.float32); np_b = np.tile(np_obs, (K, 1, 1))
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

def run_traj(policy, seed, K):
    envs = [make_env(seed) for _ in range(K)]; policy.reset()
    obs_list = [env.reset() for env in envs]; mrs = [0.0]*K; dones = [False]*K
    agent_trajs = [[] for _ in range(K)]
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
            agent_trajs[k].append(obs_new[1, 18:20].copy() if obs_new.ndim == 2 else obs_new[18:20].copy())
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
    return {'success_rate': float(np.mean(successes)), 'success_std': float(np.std(successes)),
            'final_pwd': float(np.mean(fds)), 'mean_path_pwd': float(np.mean(mps)), 'max_path_pwd': float(np.mean(mxs))}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--total_gpus', type=int, default=1)
    parser.add_argument('--launch', type=int, default=0)
    parser.add_argument('--gpu_offset', type=int, default=0, help='GPU ID offset (e.g. 4 for GPUs 4-7)')
    args = parser.parse_args()

    if args.launch > 0:
        script = os.path.abspath(__file__)
        procs = []
        for g in range(args.launch):
            gpu_id = g + args.gpu_offset
            lf = os.path.join(OUT, f"undertrain_gpu{gpu_id}.log")
            cmd = f"/home/wbh/mambaforge/envs/robodiff/bin/python {script} --gpu {gpu_id} --total_gpus {args.launch} --gpu_offset {args.gpu_offset}"
            p = subprocess.Popen(cmd, shell=True, stdout=open(lf, 'w'), stderr=subprocess.STDOUT)
            procs.append(p); log(f"GPU {gpu_id} PID={p.pid} -> {lf}")
        log(f"{args.launch} GPUs launched (offset={args.gpu_offset})"); [p.wait() for p in procs]; log("ALL DONE")
        return

    if args.gpu is None: parser.error("--gpu required")
    device = f"cuda:{args.gpu}"
    all_cfgs = []
    for ep in EPOCHS:
        all_cfgs.append((f"UT_ep={ep}_baseline", None, ep, N_TRAJ))
        all_cfgs.append((f"UT_ep={ep}_DPP_g=7_h=2", DiverseGuidanceConfig(energy_type="dpp", gamma=7.0, dpp_h=2.0, ortho_coeff=0.95), ep, N_TRAJ))

    my_cfgs = [c for i, c in enumerate(all_cfgs) if i % args.total_gpus == (args.gpu - args.gpu_offset)]
    log(f"GPU {args.gpu}: {len(my_cfgs)}/{len(all_cfgs)} configs, N_act={N_ACT}, N_traj={N_TRAJ}")

    curr_ep = None; policy = None; all_res = {}; total_start = time.time()
    for ci, (cfg_name, dc, epoch, n_traj) in enumerate(my_cfgs):
        if epoch != curr_ep:
            policy = load_policy(device, epoch); curr_ep = epoch
            log(f"  Loaded epoch={epoch}")
        policy.diverse_config = dc; t0 = time.time()
        log(f"\n[{ci+1}/{len(my_cfgs)}] {cfg_name} (epoch={epoch})")

        act_res = []
        for i in range(N_ACT):
            torch.manual_seed(TEST_SEED + i)
            act_res.append(run_action(policy, TEST_SEED + i, K_FIXED))
            if (i+1) % 50 == 0:
                log(f"  action [{i+1}/{N_ACT}] succ={sum(r['success'] for r in act_res)/(i+1):.3f}")

        a_succ = sum(r['success'] for r in act_res) / N_ACT
        a_pwd = float(np.mean([r['mean_pwd'] for r in act_res if r['mean_pwd'] > 0]))
        a_rew = float(np.mean([r['max_reward'] for r in act_res]))

        traj_res = []
        for i in range(n_traj):
            torch.manual_seed(TEST_SEED + i)
            traj_res.append(run_traj(policy, TEST_SEED + i, K_FIXED))
            if (i+1) % 100 == 0:
                t = traj_res[-1]
                log(f"  traj [{i+1}/{n_traj}] succ={t['success_rate']:.2f} final={t['final_pwd']:.0f} mean_path={t['mean_path_pwd']:.0f}")

        t_succ = float(np.mean([t['success_rate'] for t in traj_res]))
        t_final = float(np.mean([t['final_pwd'] for t in traj_res]))
        t_mean = float(np.mean([t['mean_path_pwd'] for t in traj_res]))
        et = time.time() - t0
        log(f"  DONE {et:.0f}s: a_succ={a_succ:.3f} a_pwd={a_pwd:.1f} t_succ={t_succ:.3f} final={t_final:.0f} mean={t_mean:.0f}")

        all_res[cfg_name] = {'epoch': epoch, 'elapsed': et,
            'action': {'success_rate': a_succ, 'avg_reward': a_rew, 'avg_pwd': a_pwd,
                       'per_seed': [{'seed': TEST_SEED+i, 'success': r['success'], 'max_reward': r['max_reward'], 'mean_pwd': r['mean_pwd']} for i, r in enumerate(act_res)]},
            'trajectory': {'success_rate': t_succ, 'final_pwd': t_final, 'mean_path_pwd': t_mean,
                           'per_seed': [{'seed': TEST_SEED+i, **t} for i, t in enumerate(traj_res)]}}

        if (ci+1) % 4 == 0:
            ts = time.strftime('%Y%m%d_%H%M%S')
            with open(os.path.join(OUT, f"undertrain_ckpt_gpu{args.gpu}_{ts}.json"), 'w') as f:
                json.dump(all_res, f, indent=2)

    total_et = time.time() - total_start
    ts = time.strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(OUT, f"undertrain_final_gpu{args.gpu}_{ts}.json"), 'w') as f:
        json.dump({'meta': {'gpu': args.gpu, 'elapsed_s': total_et, 'n_action': N_ACT, 'n_traj': N_TRAJ, 'K': K_FIXED},
                   'results': all_res}, f, indent=2)
    log(f"\nGPU {args.gpu} DONE {total_et:.0f}s")
    log("DONE")

if __name__ == '__main__':
    main()

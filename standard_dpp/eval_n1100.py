"""
Add +600 seeds to baseline and DPP gamma=7 h=2.0 → N=1100 total.
Usage: python eval_n1100.py --launch 2
"""
import sys, os, time, json, argparse, subprocess, numpy as np, torch, hydra
sys.path.insert(0, os.path.dirname(__file__))
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusion_policy.policy.diverse_guidance import DiverseGuidanceConfig
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

CKPT = "/home/wbh/shared/RTC_dp/diffusion_policy/data/pretrained/pusht_lowdim_dp_cnn/epoch=0550-test_mean_score=0.969.ckpt"
OUT = os.path.join(os.path.dirname(__file__), "eval_output_dg")
LEGACY, MAX_S, OBS_S, ACT_S, DDIM, K = True, 300, 2, 8, 16, 8
N_ACT, N_TRAJ, SEED_OFFSET = 600, 600, 100500  # start after paper (100000+200) + p0 (100200+300)

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
    return MultiStepWrapper(PushTKeypointsEnv(legacy=LEGACY, keypoint_visible_rate=1.0, **kw), OBS_S, ACT_S, MAX_S)

def run_action(policy, seed):
    env = make_env(seed); policy.reset(); obs = env.reset(); mr = 0.0; dists = []
    for _ in range(MAX_S):
        if obs.ndim == 2: obs = obs[None, ...]
        Do = obs.shape[-1] // 2; np_obs = obs[:, :OBS_S, :Do].astype(np.float32)
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

def run_traj(policy, seed):
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
            agent_trajs[k].append(obs_new[1, 18:20].copy() if obs_new.ndim==2 else obs_new[18:20].copy())
        if all(dones): break
    for env in envs: env.close()
    successes = [1.0 if mr >= 0.9 else 0.0 for mr in mrs]
    N_I = 100; interp = []
    for traj in agent_trajs:
        if len(traj) < 2: interp.append(np.zeros((N_I, 2))); continue
        arr = np.array(traj); t_old = np.linspace(0,1,len(arr)); t_new = np.linspace(0,1,N_I)
        ip = np.zeros((N_I, 2))
        for d in range(2): ip[:, d] = np.interp(t_new, t_old, arr[:, d])
        interp.append(ip)
    fds, mps = [], []
    for i in range(K):
        for j in range(i+1, K):
            sd = np.linalg.norm(interp[i]-interp[j], axis=1)
            fds.append(float(sd[-1])); mps.append(float(np.mean(sd)))
    return {'success_rate': float(np.mean(successes)), 'final_pwd': float(np.mean(fds)), 'mean_path_pwd': float(np.mean(mps))}

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
            lf = os.path.join(OUT, f"n1100_gpu{g}.log")
            cmd = f"/home/wbh/mambaforge/envs/robodiff/bin/python {script} --gpu {g} --total_gpus {args.launch}"
            p = subprocess.Popen(cmd, shell=True, stdout=open(lf, 'w'), stderr=subprocess.STDOUT)
            procs.append(p); log(f"GPU {g} PID={p.pid} -> {lf}")
        log(f"{args.launch} GPUs launched"); [p.wait() for p in procs]; log("ALL DONE")
        return

    if args.gpu is None: parser.error("--gpu required")
    device = f"cuda:{args.gpu}"
    configs = [
        ("N1100_baseline_extra2", None),
        ("N1100_DPP_g=7_h=2_extra2", DiverseGuidanceConfig(energy_type="dpp", gamma=7.0, dpp_h=2.0, ortho_coeff=0.95)),
    ]
    my_cfgs = [c for i, c in enumerate(configs) if i % args.total_gpus == args.gpu]
    log(f"GPU {args.gpu}: {len(my_cfgs)} configs, +{N_ACT} action + {N_TRAJ} traj seeds each")
    policy = load_policy(device)

    all_res = {}
    for ci, (cfg_name, dc) in enumerate(my_cfgs):
        policy.diverse_config = dc; t0 = time.time()
        log(f"\n[{ci+1}] {cfg_name}")

        act_res = []
        for i in range(N_ACT):
            torch.manual_seed(SEED_OFFSET + i)
            act_res.append(run_action(policy, SEED_OFFSET + i))
            if (i+1) % 200 == 0:
                log(f"  action [{i+1}/{N_ACT}] succ={sum(r['success'] for r in act_res)/(i+1):.3f}")

        a_succ = sum(r['success'] for r in act_res) / N_ACT
        a_pwd = float(np.mean([r['mean_pwd'] for r in act_res if r['mean_pwd'] > 0]))
        a_rew = float(np.mean([r['max_reward'] for r in act_res]))
        a_per_seed = [{'seed': SEED_OFFSET+i, 'success': r['success'], 'max_reward': r['max_reward'], 'mean_pwd': r['mean_pwd']} for i, r in enumerate(act_res)]

        traj_res = []
        for i in range(N_TRAJ):
            torch.manual_seed(SEED_OFFSET + i)
            traj_res.append(run_traj(policy, SEED_OFFSET + i))
            if (i+1) % 200 == 0:
                t = traj_res[-1]
                log(f"  traj [{i+1}/{N_TRAJ}] succ={t['success_rate']:.2f} final={t['final_pwd']:.0f}")

        t_succ = float(np.mean([t['success_rate'] for t in traj_res]))
        t_final = float(np.mean([t['final_pwd'] for t in traj_res]))
        t_mean = float(np.mean([t['mean_path_pwd'] for t in traj_res]))
        t_per_seed = [{'seed': SEED_OFFSET+i, **t} for i, t in enumerate(traj_res)]
        et = time.time() - t0
        log(f"  DONE {et:.0f}s: a_succ={a_succ:.3f} a_pwd={a_pwd:.1f} t_succ={t_succ:.3f}")

        all_res[cfg_name] = {'action': {'success_rate': a_succ, 'avg_reward': a_rew, 'avg_pwd': a_pwd, 'per_seed': a_per_seed},
                             'trajectory': {'success_rate': t_succ, 'final_pwd': t_final, 'mean_path_pwd': t_mean, 'per_seed': t_per_seed}}

    ts = time.strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(OUT, f"n1100_final_gpu{args.gpu}_{ts}.json"), 'w') as f:
        json.dump(all_res, f, indent=2)
    log(f"\nGPU {args.gpu} DONE. Saved.")
    log("DONE")

if __name__ == '__main__':
    main()

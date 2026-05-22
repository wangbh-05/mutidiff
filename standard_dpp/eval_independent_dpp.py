"""
Independent per-env DPP: each env replicates its own obs K times, runs DPP.
Compare baseline vs DPP gamma=7 h=2.0. K=4, N=50 seeds.

This isolates the PURE effect of DPP on trajectory diversity — at every step,
DPP works on K identical observations within each env.
"""
import sys, os, time, json, pickle, numpy as np, torch, hydra
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusion_policy.policy.diverse_guidance import DiverseGuidanceConfig
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

CKPT = "/home/wbh/shared/RTC_dp/diffusion_policy/data/pretrained/pusht_lowdim_dp_cnn/epoch=0550-test_mean_score=0.969.ckpt"
DEV, K, N_SEEDS, SEED = "cuda:0", 4, 50, 100000
RESULTS = os.path.join(os.path.dirname(__file__), "results")

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

kw = PushTKeypointsEnv.genenerate_keypoint_manager_params()
pld = torch.load(CKPT, map_location=DEV, pickle_module=__import__('dill'))
sc = DDIMScheduler(num_train_timesteps=100, beta_start=0.0001, beta_end=0.02,
                   beta_schedule='squaredcos_cap_v2', clip_sample=True,
                   set_alpha_to_one=True, steps_offset=0, prediction_type='epsilon')

def load_policy():
    p = hydra.utils.instantiate(pld['cfg'].policy.copy(), noise_scheduler=sc, num_inference_steps=16)
    p.load_state_dict(pld['state_dicts']['model']); p.to(DEV).eval()
    return p

def run_independent(policy, seed):
    """K parallel envs, each independently runs DPP on its own replicated obs."""
    envs = [MultiStepWrapper(PushTKeypointsEnv(legacy=True, keypoint_visible_rate=1.0, **kw), 2, 8, 300) for _ in range(K)]
    for env in envs: env.seed(seed)
    policy.reset()
    obs_list = [env.reset() for env in envs]; mrs, dones = [0.0]*K, [False]*K
    agent_trajs = [[] for _ in range(K)]

    for _ in range(300):
        for k in range(K):
            if dones[k]: continue
            obs_k = obs_list[k]
            if obs_k.ndim == 2: obs_k = obs_k[None, ...]
            Do = obs_k.shape[-1] // 2
            np_b = np.tile(obs_k[:, :2, :Do].astype(np.float32), (K, 1, 1))
            with torch.no_grad():
                result = policy.predict_action({'obs': torch.from_numpy(np_b).to(DEV)})
            a_k = result['action'][k].cpu().numpy()
            obs_new, rew, done, info = envs[k].step(a_k)
            obs_list[k] = obs_new
            rv = float(rew) if np.isscalar(rew) else float(np.asarray(rew).flat[0])
            mrs[k] = max(mrs[k], rv); dones[k] = done
            agent_trajs[k].append(obs_new[1, 18:20].copy() if obs_new.ndim==2 else obs_new[18:20].copy())
        if all(dones): break

    for env in envs: env.close()
    successes = [1.0 if mr >= 0.9 else 0.0 for mr in mrs]

    # Metrics: pairwise distances between K agent trajectories
    N_I = 100; interp = []
    for traj in agent_trajs:
        if len(traj) < 2: interp.append(np.zeros((N_I, 2))); continue
        arr = np.array(traj); t_old = np.linspace(0,1,len(arr)); t_new = np.linspace(0,1,N_I)
        ip = np.zeros((N_I, 2))
        for d in range(2): ip[:, d] = np.interp(t_new, t_old, arr[:, d])
        interp.append(ip)
    fds, mps, mxs = [], [], []
    for i in range(K):
        for j in range(i+1, K):
            sd = np.linalg.norm(interp[i]-interp[j], axis=1)
            fds.append(float(sd[-1])); mps.append(float(np.mean(sd))); mxs.append(float(np.max(sd)))

    return {
        'success_rate': float(np.mean(successes)),
        'final_pwd': float(np.mean(fds)), 'mean_path_pwd': float(np.mean(mps)), 'max_path_pwd': float(np.mean(mxs)),
        'agent_trajs': [np.array(t) for t in agent_trajs], 'successes': successes,
    }

log(f"Independent Per-Env DPP — K={K}, N={N_SEEDS}")
configs = [
    ("Baseline_gamma=0", None),
    ("DPP_g=7_h=2", DiverseGuidanceConfig(energy_type="dpp", gamma=7.0, dpp_h=2.0, ortho_coeff=0.95)),
]
all_results = {}
for cfg_name, dc in configs:
    p = load_policy(); p.diverse_config = dc
    per_seed = []; t0 = time.time()
    for i in range(N_SEEDS):
        torch.manual_seed(SEED + i)
        r = run_independent(p, SEED + i)
        per_seed.append(r)
        if (i+1) % 10 == 0:
            succ = np.mean([r['success_rate'] for r in per_seed])
            fp = np.mean([r['final_pwd'] for r in per_seed])
            mp = np.mean([r['mean_path_pwd'] for r in per_seed])
            log(f"  {cfg_name} [{i+1}/{N_SEEDS}] succ={succ:.2f} finalPWD={fp:.0f} meanPathPWD={mp:.0f}")

    elapsed = time.time() - t0
    succs = np.array([r['success_rate'] for r in per_seed])
    fps = np.array([r['final_pwd'] for r in per_seed])
    mps = np.array([r['mean_path_pwd'] for r in per_seed])
    log(f"  {cfg_name} DONE {elapsed:.0f}s: succ={succs.mean():.3f}±{succs.std()/np.sqrt(N_SEEDS):.3f} finalPWD={fps.mean():.0f} meanPathPWD={mps.mean():.0f}")

    all_results[cfg_name] = {'success_rate': float(succs.mean()), 'success_sem': float(succs.std()/np.sqrt(N_SEEDS)),
                             'final_pwd': float(fps.mean()), 'mean_path_pwd': float(mps.mean()),
                             'per_seed': [{'seed': SEED+i, 'success_rate': r['success_rate'],
                                           'final_pwd': r['final_pwd'], 'mean_path_pwd': r['mean_path_pwd']}
                                          for i, r in enumerate(per_seed)]}
    # Save raw trajectories for first 5 seeds for viz
    raw_for_viz = [{'agent_trajs': per_seed[i]['agent_trajs'], 'successes': per_seed[i]['successes']} for i in range(min(5, N_SEEDS))]
    with open(os.path.join(RESULTS, f"independent_raw_{cfg_name}.pkl"), 'wb') as f:
        pickle.dump(raw_for_viz, f)

ts = time.strftime('%Y%m%d_%H%M%S')
with open(os.path.join(RESULTS, f"independent_dpp_{ts}.json"), 'w') as f:
    json.dump(all_results, f, indent=2)
log(f"\nSaved. Comparing:")
from scipy import stats
bl = np.array([r['success_rate'] for r in all_results['Baseline_gamma=0']['per_seed']])
dp = np.array([r['success_rate'] for r in all_results['DPP_g=7_h=2']['per_seed']])
t, p = stats.ttest_rel(dp, bl)
log(f"  Succ: bl={bl.mean():.3f} dpp={dp.mean():.3f} Δ={dp.mean()-bl.mean():+.3f} p={p:.4f}")
bl_f = np.array([r['final_pwd'] for r in all_results['Baseline_gamma=0']['per_seed']])
dp_f = np.array([r['final_pwd'] for r in all_results['DPP_g=7_h=2']['per_seed']])
t2, p2 = stats.ttest_rel(dp_f, bl_f)
log(f"  FinalPWD: bl={bl_f.mean():.0f} dpp={dp_f.mean():.0f} Δ={dp_f.mean()-bl_f.mean():+.0f} p={p2:.4f}")
log("DONE")

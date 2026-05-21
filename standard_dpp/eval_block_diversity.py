"""Compare DPP vs NOISE: block final position diversity via K independent ActSucc rollouts."""
import sys, os, time, json, numpy as np, torch, hydra
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusion_policy.policy.diverse_guidance import DiverseGuidanceConfig
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

CKPT = "/home/wbh/shared/RTC_dp/diffusion_policy/data/pretrained/pusht_lowdim_dp_cnn/epoch=0550-test_mean_score=0.969.ckpt"
DEV = "cuda:0"; K = 8; N_SEEDS = 100
OUT = os.path.join(os.path.dirname(__file__), "results")

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

kw = PushTKeypointsEnv.genenerate_keypoint_manager_params()
pld = torch.load(CKPT, map_location=DEV, pickle_module=__import__('dill'))
sc = DDIMScheduler(num_train_timesteps=100, beta_start=0.0001, beta_end=0.02,
                   beta_schedule='squaredcos_cap_v2', clip_sample=True,
                   set_alpha_to_one=True, steps_offset=0, prediction_type='epsilon')
p = hydra.utils.instantiate(pld['cfg'].policy.copy(), noise_scheduler=sc, num_inference_steps=16)
p.load_state_dict(pld['state_dicts']['model']); p.to(DEV).eval()

def run_rollouts(policy, start_seed, K):
    block_finals, successes = [], []
    for k in range(K):
        torch.manual_seed(start_seed * 1000 + k)
        env = MultiStepWrapper(PushTKeypointsEnv(legacy=True, keypoint_visible_rate=1.0, **kw), 2, 8, 300)
        env.seed(start_seed); policy.reset(); obs = env.reset(); mr = 0.0; last_block = None
        for _ in range(300):
            if obs.ndim == 2: obs = obs[None, ...]
            Do = obs.shape[-1] // 2
            np_obs = obs[:, :2, :Do].astype(np.float32)
            np_b = np.tile(np_obs, (K, 1, 1))
            with torch.no_grad(): result = policy.predict_action({'obs': torch.from_numpy(np_b).to(DEV)})
            a0 = result['action'][0].cpu().numpy()
            obs, rew, done, _ = env.step(a0)
            mr = max(mr, float(rew) if np.isscalar(rew) else float(np.asarray(rew).flat[0]))
            if obs.ndim == 2: last_block = obs[1, 0:18].reshape(9,2).mean(0).copy()
            else: last_block = obs[0:18].reshape(9,2).mean(0).copy()
            if done: break
        env.close()
        block_finals.append(last_block if last_block is not None else np.zeros(2))
        successes.append(1.0 if mr >= 0.9 else 0.0)
    bf = np.array(block_finals)
    pwds = [np.linalg.norm(bf[i]-bf[j]) for i in range(K) for j in range(i+1, K)]
    return float(np.mean(pwds)), float(np.mean(successes)), [float(x) for x in successes], [bf[k].tolist() for k in range(K)]

configs = [
    ("NOISE_eta=0.3", DiverseGuidanceConfig(energy_type="dpp", gamma=0.0, eta_sde=0.3, ortho_coeff=0.95)),
    ("DPP_g=7_h=2", DiverseGuidanceConfig(energy_type="dpp", gamma=7.0, dpp_h=2.0, ortho_coeff=0.95)),
]

log(f"Block final position diversity — K={K} rollouts per seed, N={N_SEEDS} seeds")
all_results = {}
for name, dc in configs:
    p.diverse_config = dc
    per_seed = []; t0 = time.time()
    for i in range(N_SEEDS):
        pwd, succ, succs, finals = run_rollouts(p, 100000 + i, K)
        per_seed.append({'seed': 100000+i, 'block_final_pwd': pwd, 'success_rate': succ, 'successes': succs, 'block_finals': finals})
        if (i+1) % 25 == 0:
            log(f"  {name} [{i+1}/{N_SEEDS}] pwd={np.mean([s['block_final_pwd'] for s in per_seed]):.1f} succ={np.mean([s['success_rate'] for s in per_seed]):.3f}")
    pwds = np.array([s['block_final_pwd'] for s in per_seed])
    succs = np.array([s['success_rate'] for s in per_seed])
    log(f"  {name} DONE {time.time()-t0:.0f}s: BlockFinalPWD={pwds.mean():.1f}±{pwds.std()/np.sqrt(N_SEEDS):.1f} Succ={succs.mean():.3f}")
    all_results[name] = {'block_final_pwd_mean': float(pwds.mean()), 'block_final_pwd_sem': float(pwds.std()/np.sqrt(N_SEEDS)),
                         'success_rate': float(succs.mean()), 'success_sem': float(succs.std()/np.sqrt(N_SEEDS)),
                         'per_seed': per_seed}

ts = time.strftime('%Y%m%d_%H%M%S')
with open(os.path.join(OUT, f'block_diversity_{ts}.json'), 'w') as f:
    json.dump({'meta': {'K': K, 'N_seeds': N_SEEDS}, 'results': all_results}, f, indent=2)
log(f"Saved. Summary:")
for name, r in all_results.items():
    log(f"  {name}: BlockFinalPWD={r['block_final_pwd_mean']:.1f}±{r['block_final_pwd_sem']:.1f} Succ={r['success_rate']:.3f}")

# Significance
from scipy import stats
n_pwds = np.array([s['block_final_pwd'] for s in all_results['NOISE_eta=0.3']['per_seed']])
d_pwds = np.array([s['block_final_pwd'] for s in all_results['DPP_g=7_h=2']['per_seed']])
t, p = stats.ttest_rel(d_pwds, n_pwds)
log(f"  Paired t-test: t={t:.2f} p={p:.4f} {'***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'}")
log("DONE")

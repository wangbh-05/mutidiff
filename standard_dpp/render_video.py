"""Render videos: baseline vs DPP, 1 seed, K=4 trajectories."""
import sys, os, numpy as np, torch, hydra, imageio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusion_policy.policy.diverse_guidance import DiverseGuidanceConfig
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

CKPT = "/home/wbh/shared/RTC_dp/diffusion_policy/data/pretrained/pusht_lowdim_dp_cnn/epoch=0550-test_mean_score=0.969.ckpt"
DEV, K, SEED = "cuda:0", 4, 100000
RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

kw = PushTKeypointsEnv.genenerate_keypoint_manager_params()
pld = torch.load(CKPT, map_location=DEV, pickle_module=__import__('dill'))
sc = DDIMScheduler(num_train_timesteps=100, beta_start=0.0001, beta_end=0.02,
                   beta_schedule='squaredcos_cap_v2', clip_sample=True,
                   set_alpha_to_one=True, steps_offset=0, prediction_type='epsilon')
p = hydra.utils.instantiate(pld['cfg'].policy.copy(), noise_scheduler=sc, num_inference_steps=16)
p.load_state_dict(pld['state_dicts']['model']); p.to(DEV).eval()

def render_traj(policy, seed, K, output_path):
    """K parallel envs, render each step, save as video."""
    envs = [MultiStepWrapper(PushTKeypointsEnv(legacy=True, keypoint_visible_rate=1.0, draw_keypoints=True, **kw), 2, 8, 300) for _ in range(K)]
    for env in envs: env.seed(seed)
    policy.reset()
    obs_list = [env.reset() for env in envs]
    dones = [False]*K; frames = []

    for _ in range(300):
        obs_batch = np.stack(obs_list); Do = obs_batch.shape[-1] // 2
        with torch.no_grad():
            result = policy.predict_action({'obs': torch.from_numpy(obs_batch[:, :2, :Do].astype(np.float32)).to(DEV)})
        for k in range(K):
            if dones[k]: continue
            obs_new, _, done, _ = envs[k].step(result['action'][k].cpu().numpy())
            obs_list[k] = obs_new; dones[k] = done

        # Render all K envs side by side
        renders = [envs[k].render(mode='rgb_array') for k in range(K)]
        while len(renders) < K: renders.append(np.zeros_like(renders[0]))
        # Stack K renders horizontally
        row = np.hstack(renders)
        frames.append(row)
        if all(dones): break

    for env in envs: env.close()
    imageio.mimsave(output_path, frames, fps=10)
    print(f'Saved {output_path} ({len(frames)} frames)')

# Baseline
torch.manual_seed(SEED); p.diverse_config = None
render_traj(p, SEED, K, os.path.join(RESULTS, 'video_baseline.mp4'))

# DPP
torch.manual_seed(SEED)
p.diverse_config = DiverseGuidanceConfig(energy_type="dpp", gamma=7.0, dpp_h=2.0, ortho_coeff=0.95)
render_traj(p, SEED, K, os.path.join(RESULTS, 'video_dpp.mp4'))

print("Done!")

"""Compare rescue rates: Baseline vs NOISE vs DPP on same seeds."""
import sys, os, numpy as np, torch, hydra
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusion_policy.policy.diverse_guidance import DiverseGuidanceConfig
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

CKPT='/home/wbh/shared/RTC_dp/diffusion_policy/data/pretrained/pusht_lowdim_dp_cnn/epoch=0550-test_mean_score=0.969.ckpt'
DEV,K='cuda:0',8
kw=PushTKeypointsEnv.genenerate_keypoint_manager_params()
pld=torch.load(CKPT,map_location=DEV,pickle_module=__import__('dill'))
seeds = list(range(100000, 100200))  # 200 seeds

def run(dc, seeds):
    sc=DDIMScheduler(num_train_timesteps=100,beta_start=0.0001,beta_end=0.02,beta_schedule='squaredcos_cap_v2',clip_sample=True,set_alpha_to_one=True,steps_offset=0,prediction_type='epsilon')
    p=hydra.utils.instantiate(pld['cfg'].policy.copy(),noise_scheduler=sc,num_inference_steps=16)
    p.load_state_dict(pld['state_dicts']['model']);p.to(DEV).eval();p.diverse_config=dc
    results={}
    for i,seed in enumerate(seeds):
        torch.manual_seed(seed)
        envs=[MultiStepWrapper(PushTKeypointsEnv(legacy=True,keypoint_visible_rate=1.0,**kw),2,8,300) for _ in range(K)]
        for env in envs:env.seed(seed)
        p.reset();obs_list=[env.reset() for env in envs];dones=[False]*K;mrs=[0.0]*K
        for _ in range(300):
            for k in range(K):
                if dones[k]:continue
                obs_k=obs_list[k]
                if obs_k.ndim==2:obs_k=obs_k[None,...]
                Do=obs_k.shape[-1]//2
                np_b=np.tile(obs_k[:,:2,:Do].astype(np.float32),(K,1,1))
                with torch.no_grad():r=p.predict_action({'obs':torch.from_numpy(np_b).to(DEV)})
                obs_new,rew,done,info=envs[k].step(r['action'][k].cpu().numpy())
                obs_list[k]=obs_new;dones[k]=done
                mrs[k]=max(mrs[k],float(rew) if np.isscalar(rew) else float(np.asarray(rew).flat[0]))
            if all(dones):break
        for env in envs:env.close()
        results[seed]=[1.0 if mr>=0.9 else 0.0 for mr in mrs]
        if (i+1)%50==0: print(f'  {i+1}/{len(seeds)} done')
    del p; return results

cfgs = [
    ('Baseline', None),
    ('NOISE eta=0.5', DiverseGuidanceConfig(energy_type='dpp',gamma=0.0,eta_sde=0.5,ortho_coeff=0.95)),
    ('DPP gamma=7 h=2', DiverseGuidanceConfig(energy_type='dpp',gamma=7.0,dpp_h=2.0,ortho_coeff=0.95)),
]

all = {}
for name, dc in cfgs:
    print(f'Running {name}...')
    all[name] = run(dc, seeds)

total = len(seeds)*K
for label, dc_name in [('NOISE', 'NOISE eta=0.5'), ('DPP', 'DPP gamma=7 h=2')]:
    rescue = break_ = 0
    for seed in seeds:
        for k in range(K):
            bl_ok = all['Baseline'][seed][k] >= 1.0
            t_ok = all[dc_name][seed][k] >= 1.0
            if not bl_ok and t_ok: rescue += 1
            elif bl_ok and not t_ok: break_ += 1
    print(f'{label}: Rescue={rescue} ({rescue/total*100:.2f}%) Break={break_} ({break_/total*100:.2f}%) Net={rescue-break_:+d}')
print('Done')

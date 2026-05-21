"""All SUCCESSFUL trajectories: baseline vs DPP gamma=7 h=2.0 (N=500 combined)."""
import json, pickle, os, glob, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

OUT = os.path.join(os.path.dirname(__file__), "eval_output_dg")
BOARD, GOAL = 512, (256, 256)

def load_raw(name):
    """Load raw trajectory pickle files matching a config name."""
    all_data = []
    for f in sorted(glob.glob(os.path.join(OUT, f'raw_{name}_*.pkl'))):
        with open(f, 'rb') as fp:
            all_data.extend(pickle.load(fp))
    return all_data  # list of seed dicts

def fig_success(cfg_name, label, ax):
    """Plot all SUCCESSFUL agent trajectories."""
    raw = load_raw(cfg_name)
    total, ok, fail = 0, 0, 0
    for seed_data in raw:
        for k in range(len(seed_data['agent_trajs'])):
            agent = seed_data['agent_trajs'][k]
            succ = seed_data['successes'][k]
            total += 1
            if succ and len(agent) >= 2:
                ax.plot(agent[:, 0], agent[:, 1], '-', color='#1a5276', lw=1.0, alpha=0.12)
                ok += 1
            elif not succ:
                fail += 1

    ax.set_xlim(0, BOARD); ax.set_ylim(0, BOARD); ax.set_aspect('equal')
    ax.set_facecolor('white')
    ax.set_title(f'{label}\n{ok} successful / {total} total ({ok/total*100:.1f}%)',
                 fontsize=14, fontweight='bold')
    ax.add_patch(FancyBboxPatch((GOAL[0]-40, GOAL[1]-40), 80, 80,
                 boxstyle='round,pad=0.1', fill=True, facecolor='lightgreen',
                 edgecolor='green', lw=2, alpha=0.3))
    ax.annotate('GOAL', GOAL, fontsize=10, ha='center', va='center', fontweight='bold', color='darkgreen')
    ax.set_xticks([]); ax.set_yticks([])
    return ok, fail, total

# Find the exact config names from combined data
results = {}
for f in sorted(glob.glob(os.path.join(OUT, 'paper_final_gpu*.json'))):
    with open(f) as fp: results.update(json.load(fp)['results'])
for f in sorted(glob.glob(os.path.join(OUT, 'p0_final_gpu*.json'))):
    with open(f) as fp: results.update(json.load(fp)['results'])

bl_name = [c for c in results if c == 'K=8_g=0'][0]
dpp_name = [c for c in results if 'DPP_g=7_h=2.0' in c and 'K=8' in c and 'tgate' not in c and 'P0' not in c][0]

# Also include P0 extra data
bl_p0_name = 'P0_K=8_g=0_baseline_extra'
dpp_p0_name = 'P0_K=8_DPP_g=7_h=2.0_extra'

print(f'Baseline: {bl_name} + {bl_p0_name}')
print(f'DPP:      {dpp_name} + {dpp_p0_name}')

# ================================================================
# FIGURE: All SUCCESS trajectories side by side
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(20, 11))
fig.suptitle('All Successful Agent Trajectories — Baseline vs DPP (N=500 seeds, K=8 rollouts)',
             fontsize=16, fontweight='bold')

# Baseline (combine paper + P0)
bl_ok, bl_fail, bl_total = 0, 0, 0
for name in [bl_name, bl_p0_name]:
    o, f, t = fig_success(name, '', axes[0])  # dummy label
    bl_ok += o; bl_fail += f; bl_total += t

# Update title for baseline
a = results[bl_name]['action']
axes[0].set_title(f'Baseline (gamma=0)\n'
                  f'ActSucc={a["success_rate"]:.3f}  ActPWD={a["avg_pwd"]:.0f}\n'
                  f'{bl_ok} successful / {bl_total} total ({bl_ok/bl_total*100:.1f}%)',
                  fontsize=14, fontweight='bold')

# DPP (combine paper + P0)
dpp_ok, dpp_fail, dpp_total = 0, 0, 0
for name in [dpp_name, dpp_p0_name]:
    o, f, t = fig_success(name, '', axes[1])
    dpp_ok += o; dpp_fail += f; dpp_total += t

a = results[dpp_name]['action']
axes[1].set_title(f'DPP gamma=7 h=2.0\n'
                  f'ActSucc={a["success_rate"]:.3f}  ActPWD={a["avg_pwd"]:.0f}\n'
                  f'{dpp_ok} successful / {dpp_total} total ({dpp_ok/dpp_total*100:.1f}%)',
                  fontsize=14, fontweight='bold')

plt.tight_layout()
out_path = os.path.join(OUT, 'success_traj_all.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved {out_path}')
print(f'Baseline: {bl_ok} ok / {bl_total} total ({bl_ok/bl_total*100:.1f}%)')
print(f'DPP:      {dpp_ok} ok / {dpp_total} total ({dpp_ok/dpp_total*100:.1f}%)')

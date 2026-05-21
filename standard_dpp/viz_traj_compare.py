"""Compare ALL trajectories: baseline vs best DPP config."""
import json, pickle, os, glob, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

OUT = os.path.join(os.path.dirname(__file__), "eval_output_dg")
BOARD, GOAL = 512, (256, 256)

# Load results to find config names
results = {}
for f in sorted(glob.glob(os.path.join(OUT, 'paper_final_gpu*.json'))):
    with open(f) as fp: results.update(json.load(fp)['results'])

bl_name = [c for c in results if c == 'K=8_g=0'][0]
dpp_name = [c for c in results if 'DPP_g=7_h=2.0' in c and 'K=8' in c and 'tgate' not in c][0]

def load_raw(cfg_name):
    for f in sorted(glob.glob(os.path.join(OUT, f'raw_{cfg_name}_*.pkl'))):
        with open(f, 'rb') as fp: return pickle.load(fp)
    return None

bl_raw = load_raw(bl_name)
dpp_raw = load_raw(dpp_name)

print(f'Baseline: {len(bl_raw)} seeds x {len(bl_raw[0]["agent_trajs"])} rollouts')
print(f'DPP:      {len(dpp_raw)} seeds x {len(dpp_raw[0]["agent_trajs"])} rollouts')

# ============================================================================
# FIGURE: Side-by-side — ALL trajectories
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(20, 11))
fig.suptitle('All Agent Trajectories — 200 seeds x K=8 rollouts', fontsize=16, fontweight='bold')

COLOR_OK = '#1a5276'
COLOR_FAIL = '#c0392b'

for idx, (cfg_name, raw, r) in enumerate([
    (bl_name, bl_raw, results[bl_name]),
    (dpp_name, dpp_raw, results[dpp_name]),
]):
    ax = axes[idx]
    ax.set_xlim(0, BOARD); ax.set_ylim(0, BOARD); ax.set_aspect('equal')
    ax.set_facecolor('white')

    short = 'Baseline (gamma=0)' if 'g=0' in cfg_name else 'DPP gamma=7 h=2.0'
    ax.set_title(f'{short}\n'
                 f'ActSucc={r["action"]["success_rate"]:.3f}  '
                 f'ActPWD={r["action"]["avg_pwd"]:.0f}  '
                 f'TrajSucc={r["trajectory"]["success_rate"]:.3f}  '
                 f'MeanPathPWD={r["trajectory"]["mean_path_pwd"]:.0f}',
                 fontsize=13, fontweight='bold')

    # Goal region
    ax.add_patch(FancyBboxPatch((GOAL[0]-40, GOAL[1]-40), 80, 80,
                 boxstyle='round,pad=0.1', fill=True, facecolor='lightgreen',
                 edgecolor='green', lw=2, alpha=0.3))
    ax.annotate('GOAL', GOAL, fontsize=10, ha='center', va='center', fontweight='bold', color='darkgreen')
    ax.set_xticks([]); ax.set_yticks([])

    total_ok, total_fail = 0, 0
    for seed_idx in range(len(raw)):
        seed_data = raw[seed_idx]
        for k in range(len(seed_data['agent_trajs'])):
            agent = seed_data['agent_trajs'][k]
            succ = seed_data['successes'][k]
            if len(agent) < 2: continue
            if succ:
                ax.plot(agent[:, 0], agent[:, 1], '-', color=COLOR_OK, lw=1.0, alpha=0.12)
                total_ok += 1
            else:
                ax.plot(agent[:, 0], agent[:, 1], '-', color=COLOR_FAIL, lw=1.0, alpha=0.20)
                total_fail += 1

    print(f'{short}: {total_ok} OK / {total_fail} FAIL ({total_ok/(total_ok+total_fail)*100:.1f}%)')

# Legend
handles = [
    Line2D([0],[0], color=COLOR_OK, lw=2, label='Success'),
    Line2D([0],[0], color=COLOR_FAIL, lw=2, label='Failure'),
    plt.Rectangle((0,0),1,1, facecolor='lightgreen', edgecolor='green', alpha=0.3, label='Goal'),
]
axes[0].legend(handles=handles, loc='upper right', fontsize=10)

plt.tight_layout()
out_path = os.path.join(OUT, 'paper_traj_all.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved {out_path}')

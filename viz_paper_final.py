"""
Complete paper figures for DPP diversity guidance experiment.
"""
import json, pickle, os, glob, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

OUT = os.path.join(os.path.dirname(__file__), "eval_output_dg")
BOARD, GOAL = 512, (256, 256)

# Load
results = {}
for f in sorted(glob.glob(os.path.join(OUT, 'paper_final_gpu*.json'))):
    with open(f) as fp: results.update(json.load(fp)['results'])

def load_raw(cfg_name):
    for f in sorted(glob.glob(os.path.join(OUT, f'raw_{cfg_name}_*.pkl'))):
        with open(f, 'rb') as fp: return pickle.load(fp)
    return None

# Extract data grid
rows = []
for cfg_name, r in results.items():
    a = r['action']; t = r['trajectory']
    K = r['K']; g = r['config']['gamma']
    h = r['config'].get('dpp_h') or 0
    et = r['config'].get('energy_type','dpp')
    rows.append({'name': cfg_name, 'K': K, 'gamma': g, 'h': h, 'energy': et,
                 'act_succ': a['success_rate'], 'act_pwd': a['avg_pwd'],
                 'traj_succ': t['success_rate'], 'final_pwd': t['final_pwd'],
                 'mean_path': t['mean_path_pwd'], 'max_path': t['max_path_pwd']})

baseline_K4 = [r for r in rows if r['K']==4 and r['gamma']==0][0]
baseline_K8 = [r for r in rows if r['K']==8 and r['gamma']==0][0]

# ============================================================================
# FIGURE 1: Main result — gamma × h grid at K=8
# ============================================================================
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))
fig1.suptitle('DPP Diversity Guidance — K=8 Paper Results', fontsize=16, fontweight='bold')

dpp_k8 = [r for r in rows if r['K']==8 and r['gamma']>0 and 'DPP' in r['name'] and 'tgate' not in r['name']]
hs = sorted(set(r['h'] for r in dpp_k8))
COLORS_H = {0.5: 'red', 1.0: 'steelblue', 2.0: 'darkgreen', 5.0: 'gray'}

# Subplot 1: ActSucc vs ActPWD
ax = axes1[0,0]
for h in hs:
    pts = [r for r in dpp_k8 if r['h']==h]
    pts.sort(key=lambda r: r['gamma'])
    ax.plot([p['act_pwd'] for p in pts], [p['act_succ'] for p in pts], 'o-', lw=2, markersize=8, color=COLORS_H[h], label=f'h={h}')
    for p in pts:
        ax.annotate(f'{p["gamma"]:.0f}', (p['act_pwd']+1, p['act_succ']+0.005), fontsize=7, color=COLORS_H[h])
ax.axhline(baseline_K8['act_succ'], color='black', ls='--', alpha=0.4, lw=1)
ax.axvline(baseline_K8['act_pwd'], color='black', ls='--', alpha=0.4, lw=1)
ax.scatter([baseline_K8['act_pwd']], [baseline_K8['act_succ']], color='black', marker='D', s=120, zorder=10, label='Baseline (γ=0)')
ax.set_xlabel('Action PWD ↑', fontsize=11); ax.set_ylabel('Action Success Rate ↑', fontsize=11)
ax.set_title('Action Diversity'); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

# Subplot 2: TrajSucc vs ActPWD
ax = axes1[0,1]
for h in hs:
    pts = [r for r in dpp_k8 if r['h']==h]
    pts.sort(key=lambda r: r['gamma'])
    ax.plot([p['act_pwd'] for p in pts], [p['traj_succ'] for p in pts], 's--', lw=2, markersize=8, color=COLORS_H[h])
    for p in pts:
        ax.annotate(f'{p["gamma"]:.0f}', (p['act_pwd']+1, p['traj_succ']+0.005), fontsize=7, color=COLORS_H[h])
ax.axhline(baseline_K8['traj_succ'], color='black', ls='--', alpha=0.4, lw=1)
ax.scatter([baseline_K8['act_pwd']], [baseline_K8['traj_succ']], color='black', marker='D', s=120, zorder=10)
ax.set_xlabel('Action PWD ↑', fontsize=11); ax.set_ylabel('Trajectory Success Rate ↑', fontsize=11)
ax.set_title('Trajectory Diversity Success'); ax.grid(True, alpha=0.2)

# Subplot 3: ActSucc vs gamma (by h)
ax = axes1[1,0]
for h in hs:
    pts = [r for r in dpp_k8 if r['h']==h]
    pts.sort(key=lambda r: r['gamma'])
    ax.plot([p['gamma'] for p in pts], [p['act_succ'] for p in pts], 'o-', lw=2, markersize=8, color=COLORS_H[h], label=f'h={h}')
ax.axhline(baseline_K8['act_succ'], color='black', ls='--', alpha=0.4, lw=1, label=f'Baseline ({baseline_K8["act_succ"]:.3f})')
ax.set_xlabel('Gamma (γ)', fontsize=11); ax.set_ylabel('Action Success Rate ↑', fontsize=11)
ax.set_title('Success Rate vs Gamma'); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
ax.set_xscale('log')

# Subplot 4: ActPWD vs gamma (by h)
ax = axes1[1,1]
for h in hs:
    pts = [r for r in dpp_k8 if r['h']==h]
    pts.sort(key=lambda r: r['gamma'])
    ax.plot([p['gamma'] for p in pts], [p['act_pwd'] for p in pts], 's--', lw=2, markersize=8, color=COLORS_H[h])
ax.axhline(baseline_K8['act_pwd'], color='black', ls='--', alpha=0.4, lw=1, label=f'Baseline ({baseline_K8["act_pwd"]:.0f})')
ax.set_xlabel('Gamma (γ)', fontsize=11); ax.set_ylabel('Action PWD ↑', fontsize=11)
ax.set_title('Action Diversity vs Gamma'); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
ax.set_xscale('log')

plt.tight_layout()
fig1.savefig(os.path.join(OUT, 'paper_main.png'), dpi=150, bbox_inches='tight')
print('Saved paper_main.png')

# ============================================================================
# FIGURE 2: K-normalisation comparison + t_gate + ablation
# ============================================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle('K-Normalisation, t-gate, and Ablations', fontsize=14, fontweight='bold')

# Subplot 1: K-normalisation (PWD vs gamma for K=4 vs K=8 at h=1.0)
ax = axes2[0]
for K, color, marker in [(4, 'blue', 'o'), (8, 'red', 's')]:
    pts = [r for r in rows if r['K']==K and r['h']==1.0 and 'DPP' in r['name'] and 'tgate' not in r['name']]
    pts.sort(key=lambda r: r['gamma'])
    ax.plot([p['gamma'] for p in pts], [p['act_pwd'] for p in pts], marker+'-', lw=2, markersize=8, color=color, label=f'K={K}')
    bl = baseline_K4 if K==4 else baseline_K8
    ax.axhline(bl['act_pwd'], color=color, ls=':', alpha=0.3, lw=1)
ax.set_xlabel('Gamma (γ)', fontsize=11); ax.set_ylabel('Action PWD ↑', fontsize=11)
ax.set_title('K-Normalisation\n(K=4 vs K=8, h=1.0)'); ax.legend(fontsize=10); ax.grid(True, alpha=0.2)
ax.set_xscale('log')

# Subplot 2: t_gate sweep
ax = axes2[1]
tgate_pts = [r for r in rows if 'tgate' in r['name']]
tgate_pts.sort(key=lambda r: r['name'].split('tgate=')[1])
tgs_values = [float(r['name'].split('tgate=')[1]) for r in tgate_pts]
ax.bar(range(len(tgs_values)), [r['act_succ'] for r in tgate_pts], color='steelblue', edgecolor='black')
ax.set_xticks(range(len(tgs_values)))
ax.set_xticklabels([f'γ=5,h=2.0\ntgate={t}' for t in tgs_values], fontsize=9)
ax.axhline(baseline_K8['act_succ'], color='black', ls='--', lw=2, alpha=0.6, label=f'Baseline ({baseline_K8["act_succ"]:.3f})')
ax.set_ylabel('Action Success Rate ↑', fontsize=11)
ax.set_title('t_gate_start\n(close guidance at high noise)'); ax.legend(fontsize=9)
for i, r in enumerate(tgate_pts):
    ax.annotate(f'PWD={r["act_pwd"]:.0f}', (i, r['act_succ']-0.03), ha='center', fontsize=8, color='white', fontweight='bold')
ax.set_ylim(0.78, 0.95)

# Subplot 3: Pure noise and OSCAR
ax = axes2[2]
ablation_pts = [r for r in rows if 'NOISE' in r['name'] or 'OSCAR' in r['name']]
labels = [r['name'].replace('K=','').replace('_eta=0.3',' η=0.3').replace('_OSCAR_g=3',' OSCAR γ=3').replace('_OSCAR_g=5',' OSCAR γ=5') for r in ablation_pts]
colors_abl = ['gray', 'gray', 'orange', 'orange', 'darkred', 'darkred']
xs = np.arange(len(ablation_pts))
ax.bar(xs, [r['act_succ'] for r in ablation_pts], color=colors_abl, edgecolor='black')
ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=8)
ax.axhline(baseline_K8['act_succ'], color='black', ls='--', lw=2, alpha=0.6, label=f'Baseline')
for i, r in enumerate(ablation_pts):
    ax.annotate(f'PWD={r["act_pwd"]:.0f}', (i, max(r['act_succ']-0.1, 0.02)), ha='center', fontsize=7, color='white', fontweight='bold')
ax.set_ylabel('Action Success Rate ↑', fontsize=11); ax.set_title('Ablations'); ax.legend(fontsize=9)

plt.tight_layout()
fig2.savefig(os.path.join(OUT, 'paper_ablation.png'), dpi=150, bbox_inches='tight')
print('Saved paper_ablation.png')

# ============================================================================
# FIGURE 3: Trajectory visualization for best 4 configs
# ============================================================================
# Select: baseline K=8, DPP g=7 h=2.0 (winner), DPP g=5 h=1.0, tgate=0.9
viz_cfgs = [
    'K=8_g=0',
    [c for c in results if 'DPP_g=7_h=2.0' in c and 'K=8' in c and 'tgate' not in c][0],
    [c for c in results if 'DPP_g=5_h=1.0' in c and 'K=8' in c and 'tgate' not in c][0],
    [c for c in results if 'tgate=0.9' in c and 'K=8' in c][0],
]

fig3, axes3 = plt.subplots(1, 4, figsize=(24, 7))
fig3.suptitle('Agent Trajectories — K=8 Parallel Rollouts, seed=100000', fontsize=14, fontweight='bold')
COLORS = plt.cm.tab10(np.linspace(0, 1, 8))

for idx, cfg_name in enumerate(viz_cfgs):
    ax = axes3[idx]
    raw = load_raw(cfg_name)
    if raw is None: ax.text(0.5,0.5,'NO RAW DATA',ha='center',transform=ax.transAxes); continue
    seed_data = raw[0]
    agent_trajs = seed_data['agent_trajs']
    successes = seed_data['successes']
    K = len(agent_trajs)

    ax.set_xlim(0, BOARD); ax.set_ylim(0, BOARD); ax.set_aspect('equal')
    ax.set_facecolor('white')
    short = cfg_name.replace('K=8_DPP_','').replace('K=8_','')
    r = results[cfg_name]
    ax.set_title(f'{short}\nA_succ={r["action"]["success_rate"]:.3f} A_pwd={r["action"]["avg_pwd"]:.0f} T_succ={r["trajectory"]["success_rate"]:.3f}',
                 fontsize=10, fontweight='bold')
    ax.add_patch(FancyBboxPatch((GOAL[0]-40, GOAL[1]-40), 80, 80,
                 boxstyle='round,pad=0.1', fill=True, facecolor='lightgreen',
                 edgecolor='green', lw=2, alpha=0.3))
    ax.set_xticks([]); ax.set_yticks([])

    for k in range(K):
        agent = np.array(agent_trajs[k]); succ = successes[k]; c = COLORS[k]
        if len(agent) > 1:
            ax.plot(agent[:, 0], agent[:, 1], '-', color=c, lw=1.0, alpha=0.12)
            ax.scatter(agent[-1, 0], agent[-1, 1], color=c, marker='o' if succ else 'X', s=30, zorder=6, edgecolors='black', linewidth=0.5)

plt.tight_layout()
fig3.savefig(os.path.join(OUT, 'paper_trajectories.png'), dpi=150, bbox_inches='tight')
print('Saved paper_trajectories.png')

# ============================================================================
# FIGURE 4: BEST CONFIG summary table
# ============================================================================
fig4, ax4 = plt.subplots(figsize=(16, 8))
ax4.axis('off')

best_k4 = [r for r in rows if r['K']==4 and 'DPP' in r['name'] and 'tgate' not in r['name']]
best_k4.sort(key=lambda r: -(r['act_succ'] + r['traj_succ']))
best_k8 = [r for r in rows if r['K']==8 and 'DPP' in r['name'] and 'tgate' not in r['name']]
best_k8.sort(key=lambda r: -(r['act_succ'] + r['traj_succ']))

table = f"""
DPP DIVERSITY GUIDANCE — PAPER RESULTS SUMMARY
═══════════════════════════════════════════════════════════════════════════════════════

BASELINES:
  K=4 (γ=0):  ActSucc={baseline_K4['act_succ']:.3f}  ActPWD={baseline_K4['act_pwd']:.1f}  TrajSucc={baseline_K4['traj_succ']:.3f}  MeanPathPWD={baseline_K4['mean_path']:.0f}
  K=8 (γ=0):  ActSucc={baseline_K8['act_succ']:.3f}  ActPWD={baseline_K8['act_pwd']:.1f}  TrajSucc={baseline_K8['traj_succ']:.3f}  MeanPathPWD={baseline_K8['mean_path']:.0f}

BEST DPP CONFIGS (K=8):
"""
for i, r in enumerate(best_k8[:3]):
    table += f"  #{i+1} γ={r['gamma']:.0f} h={r['h']:.1f}:  ActSucc={r['act_succ']:.3f}  ActPWD={r['act_pwd']:.1f}  TrajSucc={r['traj_succ']:.3f}  FinalPWD={r['final_pwd']:.0f}\n"

table += f"""
BEST DPP CONFIGS (K=4):
"""
for i, r in enumerate(best_k4[:3]):
    table += f"  #{i+1} γ={r['gamma']:.0f} h={r['h']:.1f}:  ActSucc={r['act_succ']:.3f}  ActPWD={r['act_pwd']:.1f}  TrajSucc={r['traj_succ']:.3f}\n"

tgate_best = [r for r in rows if 'tgate=0.9' in r['name']]
if tgate_best:
    r = tgate_best[0]
    table += f"""
T_GATE BEST (γ=5, h=2.0, K=8, tgate_start=0.9):
  ActSucc={r['act_succ']:.3f}  ActPWD={r['act_pwd']:.1f}  TrajSucc={r['traj_succ']:.3f}
"""

table += f"""
KEY FINDINGS:
  1. MeanPathPWD is ~{baseline_K8['mean_path']:.0f} for ALL configs — DDIM noise alone produces
     substantial trajectory diversity. Guidance changes action-space diversity
     and final position spread.
  2. K-normalisation (γ/(K-1)) greatly improves consistency across K values.
  3. h=2.0 is the sweet spot for DPP kernel bandwidth.
  4. t_gate_start=0.9 (close guidance at highest 10% noise) gives best success.
  5. OSCAR needs separate gamma tuning for K-normalised regime.
  6. Pure orthogonal noise (η=0.3, γ=0) does NOT produce meaningful diversity.
  7. ActPWD can be increased to ~70-90 while maintaining or exceeding baseline
     success rate (0.840 baseline vs 0.900 best DPP).
"""

ax4.text(0.02, 0.98, table, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace')
fig4.savefig(os.path.join(OUT, 'paper_summary.png'), dpi=150, bbox_inches='tight')
print('Saved paper_summary.png')
print('\nAll figures saved to', OUT)

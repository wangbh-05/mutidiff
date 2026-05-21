"""Paper-quality visualization — reads experiment outputs directly."""
import json, pickle, os, glob, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(__file__), "eval_output_dg")
BOARD, GOAL = 512, (256, 256)

# ---- Load aggregated metrics ----
def load_metrics():
    results = {}
    for f in sorted(glob.glob(os.path.join(OUT, "paper_final_gpu*.json"))):
        with open(f) as fp:
            results.update(json.load(fp)['results'])
    return results

# ---- Load raw trajectories for a config ----
def load_raw(cfg_name):
    for f in sorted(glob.glob(os.path.join(OUT, f"raw_{cfg_name}_*.pkl"))):
        with open(f, 'rb') as fp:
            return pickle.load(fp)
    return None

# ---- FIGURE 1: Success-Diversity trade-off ----
def fig_tradeoff(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("DPP Diversity Guidance: Success Rate vs Diversity", fontsize=14, fontweight="bold")

    # Parse gamma and h from config names
    pts = []
    for cfg, r in results.items():
        if not cfg.startswith("DPP_g="): continue
        parts = cfg.split("_")
        g = float(parts[0].split("=")[1])
        h = float(parts[1].split("=")[1])
        a = r['action']; t = r['trajectory']
        pts.append((g, h, a['success_rate'], a['avg_pwd'], t['success_rate'],
                    t['final_pwd'], t['mean_path_pwd'], t['max_path_pwd']))

    gammas = sorted(set(p[0] for p in pts))
    hs = sorted(set(p[1] for p in pts))

    # Subplot 1: ActSucc vs ActPWD
    ax = axes[0]
    for h in hs:
        hp = [p for p in pts if p[1] == h]
        hp.sort(key=lambda x: x[0])
        xs = [p[3] for p in hp]; ys = [p[2] for p in hp]
        ax.plot(xs, ys, 'o-', lw=2, markersize=8, label=f'h={h}')
        for p in hp:
            ax.annotate(f'{p[0]:.0f}', (p[3]+1, p[2]+0.005), fontsize=7)
    baseline = results.get('DPP_g=0', results.get(list(results.keys())[0]))
    bl_s = baseline['action']['success_rate']; bl_p = baseline['action']['avg_pwd']
    ax.axhline(bl_s, color='gray', ls='--', alpha=0.3)
    ax.axvline(bl_p, color='gray', ls='--', alpha=0.3)
    ax.scatter([bl_p], [bl_s], color='black', marker='D', s=100, zorder=10, label='baseline')
    ax.set_xlabel('Action PWD'); ax.set_ylabel('Action Success')
    ax.set_title('Action Diversity'); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    # Subplot 2: TrajSucc vs MeanPathPWD
    ax = axes[1]
    for h in hs:
        hp = [p for p in pts if p[1] == h]
        hp.sort(key=lambda x: x[0])
        xs = [p[6] for p in hp]; ys = [p[4] for p in hp]
        ax.plot(xs, ys, 's--', lw=2, markersize=8, label=f'h={h}')
        for p in hp:
            ax.annotate(f'{p[0]:.0f}', (p[6]+1, p[4]+0.005), fontsize=7)
    bl_t = baseline['trajectory']['success_rate']; bl_mp = baseline['trajectory']['mean_path_pwd']
    ax.axhline(bl_t, color='gray', ls='--', alpha=0.3)
    ax.scatter([bl_mp], [bl_t], color='black', marker='D', s=100, zorder=10, label='baseline')
    ax.set_xlabel('MeanPathPWD'); ax.set_ylabel('Trajectory Success')
    ax.set_title('Trajectory Diversity'); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    # Subplot 3: ActPWD vs MeanPathPWD
    ax = axes[2]
    for h in hs:
        hp = [p for p in pts if p[1] == h]
        hp.sort(key=lambda x: x[0])
        xs = [p[3] for p in hp]; ys = [p[6] for p in hp]
        ax.plot(xs, ys, 'D-', lw=2, markersize=8, label=f'h={h}')
        for p in hp:
            ax.annotate(f'{p[0]:.0f}', (p[3]+1, p[6]+1), fontsize=7)
    ax.scatter([bl_p], [bl_mp], color='black', marker='D', s=100, zorder=10, label='baseline')
    ax.set_xlabel('Action PWD'); ax.set_ylabel('MeanPathPWD')
    ax.set_title('Action → Trajectory Diversity'); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "paper_tradeoff.png"), dpi=150, bbox_inches="tight")
    print("Saved paper_tradeoff.png")

# ---- FIGURE 2: Trajectory visualization (for a chosen config) ----
def fig_trajectories(cfg_name, seed_idx=0):
    raw = load_raw(cfg_name)
    if raw is None: print(f"No raw data for {cfg_name}"); return
    seed_data = raw[seed_idx]  # first traj seed
    agent_trajs = seed_data['agent_trajs']
    successes = seed_data['successes']
    K = len(agent_trajs)
    COLORS = plt.cm.tab10(np.linspace(0, 1, K))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, BOARD); ax.set_ylim(0, BOARD); ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.set_title(f"{cfg_name} — K={K} parallel rollouts\n{int(sum(successes))}/{K} succeeded", fontsize=12, fontweight="bold")
    ax.add_patch(FancyBboxPatch((GOAL[0]-40, GOAL[1]-40), 80, 80,
                 boxstyle="round,pad=0.1", fill=True, facecolor="lightgreen",
                 edgecolor="green", lw=2, alpha=0.3))
    ax.annotate("GOAL", GOAL, fontsize=10, ha="center", va="center", fontweight="bold", color="darkgreen")
    ax.set_xticks([]); ax.set_yticks([])

    for k in range(K):
        agent = np.array(agent_trajs[k]); succ = successes[k]; c = COLORS[k]
        if len(agent) > 1:
            ax.plot(agent[:, 0], agent[:, 1], "-", color=c, lw=1.0, alpha=0.12)
            ax.scatter(agent[-1, 0], agent[-1, 1], color=c, marker="o" if succ else "X", s=40, zorder=6, edgecolors="black", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, f"traj_{cfg_name}.png"), dpi=150, bbox_inches="tight")
    print(f"Saved traj_{cfg_name}.png")

if __name__ == '__main__':
    results = load_metrics()
    if results:
        print(f"Loaded {len(results)} configs")
        fig_tradeoff(results)
        # Plot trajectories for baseline and best DPP
        for cfg in ["DPP_g=0", "DPP_g=5_h=2.0", "DPP_g=5_h=1.0"]:
            if cfg in results:
                fig_trajectories(cfg)
    else:
        print("No results found. Run eval_paper.py first.")

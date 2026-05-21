"""
Toy example: DPP diversity gradient in 2D.
Demonstrates:
  1. DPP pushes crowded samples apart, leaves isolated samples alone
  2. Orthogonal projection preserves the main direction
  3. K-normalization (E/(K-1)) makes gamma K-invariant
  4. DPP vs random noise — structured vs unstructured repulsion
"""
import os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS, exist_ok=True)
np.random.seed(42)

def pairwise_sq_distances(Z):
    Z_norm_sq = (Z**2).sum(axis=1, keepdims=True)
    return Z_norm_sq + Z_norm_sq.T - 2 * Z @ Z.T

def dpp_energy_and_grad(Z, h=1.0, eps=1e-6):
    """DPP energy E = [logdet(L+I)-logdet(L)]/(K-1), returns E and dE/dZ."""
    K = Z.shape[0]
    D = pairwise_sq_distances(Z)
    off_diag = D[~np.eye(K, dtype=bool)]
    med = np.median(off_diag).clip(min=eps)
    D = D / med
    L = np.exp(-h * D)
    I = np.eye(K)
    L_reg = L + eps * I
    L_plus_I = L_reg + I

    E = (np.linalg.slogdet(L_plus_I)[1] - np.linalg.slogdet(L_reg)[1])
    if K > 1: E = E / (K - 1)

    # Gradient via finite differences for simplicity
    grad = np.zeros_like(Z)
    delta = 1e-4
    for i in range(K):
        for d in range(2):
            Z[i, d] += delta
            D2 = pairwise_sq_distances(Z) / med
            L2 = np.exp(-h * D2)
            L2_reg = L2 + eps * I
            E2 = (np.linalg.slogdet(L2_reg + I)[1] - np.linalg.slogdet(L2_reg)[1])
            if K > 1: E2 = E2 / (K - 1)
            grad[i, d] = (E2 - E) / delta
            Z[i, d] -= delta
    return E, grad

def orthogonal_projection(g, direction):
    """Project g away from direction."""
    g_flat = g.reshape(g.shape[0], -1)
    d_flat = direction.reshape(direction.shape[0], -1)
    dot = (g_flat * d_flat).sum(axis=1)
    norm_sq = (d_flat * d_flat).sum(axis=1) + 1e-8
    alpha = (dot / norm_sq).reshape(-1, 1)
    g_parallel = alpha * direction
    return g - 0.95 * g_parallel

# ================================================================
# FIGURE 1: DPP gradient field — adaptive repulsion
# ================================================================
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))
fig1.suptitle('DPP Diversity Gradient — Adaptive Repulsion', fontsize=14, fontweight='bold')

for idx, (title, Z0) in enumerate([
    ('Crowded cluster', np.array([[0.0, 0.0], [0.3, 0.1], [-0.2, 0.2], [0.1, -0.2]])),
    ('One isolated + cluster', np.array([[0.0, 0.0], [0.3, 0.1], [-0.2, 0.2], [3.0, 3.0]])),
    ('Well-separated', np.array([[0.0, 0.0], [1.5, 1.5], [3.0, 0.0], [0.0, 3.0]])),
]):
    ax = axes1[idx]
    Z = Z0.copy()
    E, grad = dpp_energy_and_grad(Z)

    # Plot gradient arrows
    for k in range(len(Z)):
        ax.arrow(Z[k, 0], Z[k, 1], grad[k, 0]*0.5, grad[k, 1]*0.5,
                 head_width=0.1, head_length=0.15, fc='red', ec='red', alpha=0.8, width=0.03)

    ax.scatter(Z[:, 0], Z[:, 1], s=150, c='steelblue', edgecolors='black', linewidth=1, zorder=5)
    for k in range(len(Z)): ax.annotate(f'{k}', (Z[k,0]+0.1, Z[k,1]+0.1), fontsize=12, fontweight='bold')

    ax.set_xlim(-1, 5); ax.set_ylim(-1, 5); ax.set_aspect('equal')
    ax.set_title(f'{title}\nE={E:.2f}', fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)

plt.tight_layout()
fig1.savefig(os.path.join(RESULTS, 'toy_dpp_gradient.png'), dpi=150, bbox_inches='tight')
print(f'Saved toy_dpp_gradient.png')

# ================================================================
# FIGURE 2: Gradient descent evolution — DPP vs Random Noise
# ================================================================
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle('Evolution Under DPP vs Random Noise (K=4, 100 steps)', fontsize=14, fontweight='bold')

Z_dpp = np.array([[0.0, 0.0], [0.2, 0.1], [-0.1, 0.1], [0.0, -0.1]])
Z_noise = Z_dpp.copy()
traj_dpp, traj_noise = [Z_dpp.copy()], [Z_noise.copy()]
gamma = 0.5

for _ in range(100):
    _, grad = dpp_energy_and_grad(Z_dpp)
    Z_dpp = Z_dpp - gamma * grad
    traj_dpp.append(Z_dpp.copy())

    noise = np.random.randn(*Z_noise.shape)
    Z_noise = Z_noise - gamma * noise * 0.3
    traj_noise.append(Z_noise.copy())

for idx, (ax, trajs, title, color) in enumerate([
    (axes2[0], traj_dpp, 'DPP Gradient Descent', 'steelblue'),
    (axes2[1], traj_noise, 'Random Noise (no structure)', 'gray'),
]):
    for k in range(4):
        pts = np.array([t[k] for t in trajs])
        ax.plot(pts[:, 0], pts[:, 1], '-', color=color, lw=1.5, alpha=0.7)
        ax.scatter(pts[0, 0], pts[0, 1], color='black', marker='s', s=50, zorder=10)
        ax.scatter(pts[-1, 0], pts[-1, 1], color=color, marker='o', s=80, zorder=10, edgecolors='black')

    # Compute final PWD
    Z_final = trajs[-1]
    pwd = np.mean([np.linalg.norm(Z_final[i]-Z_final[j]) for i in range(4) for j in range(i+1, 4)])
    ax.set_title(f'{title}\nFinal PWD={pwd:.2f}  (start: {np.mean([np.linalg.norm(Z_dpp[i]-Z_dpp[j]) for i in range(4) for j in range(i+1, 4)]):.2f})', fontsize=12)
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect('equal'); ax.grid(True, alpha=0.2)

plt.tight_layout()
fig2.savefig(os.path.join(RESULTS, 'toy_evolution.png'), dpi=150, bbox_inches='tight')
print(f'Saved toy_evolution.png')

# ================================================================
# FIGURE 3: K-normalization — same gamma, different K
# ================================================================
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))
fig3.suptitle('K-Normalization: E/(K−1) Makes Gamma K-Invariant', fontsize=14, fontweight='bold')

for idx, (K, title) in enumerate([(2, 'K=2'), (8, 'K=8'), (32, 'K=32')]):
    ax = axes3[idx]
    # With K-normalization (our fix)
    Z_with = np.random.randn(K, 2) * 0.5
    # Without K-normalization (old bug)
    Z_without = Z_with.copy()

    traj_with, traj_without = [Z_with.copy()], [Z_without.copy()]

    for _ in range(50):
        _, grad_with = dpp_energy_and_grad(Z_with)  # E already has /(K-1)
        Z_with = Z_with - 0.3 * grad_with

        # Old bug: no /(K-1) → effective gradient K-1 times larger
        E_raw, grad_raw = dpp_energy_and_grad(Z_without)
        # Manually remove the K-1 normalization to simulate old bug
        grad_simulated_old = grad_raw * (K - 1)
        Z_without = Z_without - 0.3 * grad_simulated_old

        traj_with.append(Z_with.copy())
        traj_without.append(Z_without.copy())

    pwd_with = np.mean([np.linalg.norm(Z_with[i]-Z_with[j]) for i in range(K) for j in range(i+1, K)])
    pwd_without = np.mean([np.linalg.norm(Z_without[i]-Z_without[j]) for i in range(K) for j in range(i+1, K)])

    # Plot final positions
    ax.scatter(Z_with[:, 0], Z_with[:, 1], s=50, c='steelblue', label=f'With K-norm (PWD={pwd_with:.1f})', zorder=5)
    ax.scatter(Z_without[:, 0], Z_without[:, 1], s=50, c='red', marker='x', label=f'Without (PWD={pwd_without:.1f})', zorder=5)
    ax.set_title(f'{title}', fontsize=13)
    ax.set_xlim(-8, 8); ax.set_ylim(-8, 8); ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9)

plt.tight_layout()
fig3.savefig(os.path.join(RESULTS, 'toy_k_norm.png'), dpi=150, bbox_inches='tight')
print(f'Saved toy_k_norm.png')

# ================================================================
# FIGURE 4: Orthogonal projection explanation
# ================================================================
fig4, ax4 = plt.subplots(figsize=(10, 8))
fig4.suptitle('Orthogonal Projection: Preserving Task Quality', fontsize=14, fontweight='bold')

# Simulate: main direction (DDIM step) = upward, diversity gradient = diagonal
main_dir = np.array([0.0, 1.0])     # denoising direction
g_d = np.array([0.4, 0.6])          # raw diversity gradient (has parallel + orthogonal components)

# Projection
alpha = np.dot(g_d, main_dir) / np.dot(main_dir, main_dir)
g_parallel = alpha * main_dir
g_orthogonal = g_d - 0.95 * g_parallel

# Plot
origin = np.array([0.0, 0.0])
ax4.arrow(*origin, *main_dir*3, head_width=0.05, head_length=0.1, fc='black', ec='black', width=0.02, label='DDIM direction (task quality)', lw=2)
ax4.arrow(*origin, *g_d*3, head_width=0.05, head_length=0.1, fc='red', ec='red', width=0.02, alpha=0.7, label='Raw diversity gradient g_d')
ax4.arrow(*origin, *g_parallel*3, head_width=0.05, head_length=0.1, fc='gray', ec='gray', width=0.02, alpha=0.5, label=f'Parallel component (removed)\nα={alpha:.2f} × main_dir', ls='--')
ax4.arrow(*origin, *g_orthogonal*3, head_width=0.05, head_length=0.1, fc='green', ec='green', width=0.03, label=f'g_⊥ = g_d - 0.95·g_∥\n(diversity retained)', lw=2)

# Annotate
ax4.annotate('Quality', (0.3, 2.5), fontsize=11, fontweight='bold', color='black')
ax4.annotate('Diversity', (g_orthogonal[0]*3+0.2, g_orthogonal[1]*3-0.2), fontsize=11, fontweight='bold', color='green')

ax4.set_xlim(-0.5, 2); ax4.set_ylim(-0.5, 3.5); ax4.set_aspect('equal')
ax4.legend(fontsize=10, loc='upper left'); ax4.grid(True, alpha=0.2)

plt.tight_layout()
fig4.savefig(os.path.join(RESULTS, 'toy_orthogonal.png'), dpi=150, bbox_inches='tight')
print(f'Saved toy_orthogonal.png')
print('\nAll toy example figures saved to results/')

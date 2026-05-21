"""
Diverse Guidance for Diffusion Policy sampling.

Implements OSCAR-style diversity injection via orthogonal projection of the
diversity gradient, with support for two energy functions:

  - DPP  (Gaussian kernel, original DiverseFlow formulation):
      LL = log det(L) - log det(L+I)          (diversity score, maximise)
      E  = log det(L+I) - log det(L)          (energy to minimise)

  - OSCAR (Gram volume, from the original OSCAR paper):
      E = -0.5 * log det(I + tau * Z Z^T + eps * I)   (energy to minimise)

Key fixes vs. the reference documents:
  - Uses Tweedie's formula (not flow-matching Heun extrapolation) for endpoint prediction
  - Normalizes time by num_train_timesteps (diffusion t is integer, not [0,1])
  - Correctly unpacks torch.autograd.grad tuple
  - Handles K<2 edge case
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch


@dataclass
class DiverseGuidanceConfig:
    """Configuration for diversity guidance during diffusion sampling."""

    # Diversity energy type
    energy_type: Literal["dpp", "oscar"] = "dpp"

    # Diversity strength (0 = disabled)
    gamma: float = 0.1

    # DPP Gaussian kernel bandwidth
    dpp_h: float = 1.0

    # OSCAR volume signal-scaling coefficient
    oscar_tau: float = 1.0

    # Orthogonal projection coefficient.
    #   1.0 = completely orthogonal (remove all parallel component)
    #   0.95 = OSCAR paper recommendation (retain 5% of parallel component)
    ortho_coeff: float = 0.95

    # Orthogonal stochastic noise strength (0 = deterministic)
    eta_sde: float = 0.0

    # Time gate (normalised [0,1]; 1 = pure noise, 0 = clean).
    # Diversity is applied while t_norm in (t_gate_end, t_gate_start].
    t_gate_start: float = 1.0
    t_gate_end: float = 0.05

    # Small constant for numerical stability
    eps: float = 1e-6

    # Use median normalisation for the DPP kernel
    dpp_median_norm: bool = True


# ---------------------------------------------------------------------------
# Helper: pairwise distances
# ---------------------------------------------------------------------------

def _pairwise_sq_distances(Z: torch.Tensor) -> torch.Tensor:
    """Squared L2 distances.  Z: (K, D) -> (K, K)."""
    Z_norm_sq = (Z**2).sum(dim=-1, keepdim=True)  # (K, 1)
    D = Z_norm_sq + Z_norm_sq.T - 2.0 * (Z @ Z.T)
    return D.clamp(min=0.0)


# ---------------------------------------------------------------------------
# Energy functions
# ---------------------------------------------------------------------------

def compute_dpp_energy(
    Z: torch.Tensor,
    h: float = 1.0,
    eps: float = 1e-6,
    use_median_norm: bool = True,
) -> torch.Tensor:
    """DPP diversity energy (original formulation from DiverseFlow / DPP).

        LL = log det(L) - log det(L+I)          ← diversity score (higher = more diverse)
        E  = log det(L+I) - log det(L)          ← energy to *minimise*

        L_{ij} = exp(-h * ||Z_i - Z_j||^2 / median(||·||^2))

    The two terms pull in opposite directions:
      -log det(L)     pushes samples apart (dominates when they are close)
      +log det(L+I)   provides a restoring force (prevents unbounded divergence)

    Minimising E maximises the DPP volume spanned by the feature vectors.

    Args:
        Z: (K, D) feature matrix (flattened estimated clean samples).
        h: kernel bandwidth (> 0).
        eps: diagonal regularisation for numerical stability.
        use_median_norm: divide distances by their median (scale invariance).

    Returns:
        scalar energy (lower = more diverse).
    """
    K = Z.shape[0]
    D = _pairwise_sq_distances(Z)  # (K, K)

    if use_median_norm:
        off_diag = D[~torch.eye(K, dtype=torch.bool, device=Z.device)]
        med = off_diag.median().clamp(min=eps)
        D = D / med

    L = torch.exp(-h * D)  # (K, K)  Gaussian kernel

    # Regularise both matrices separately for numerical stability
    I_K = torch.eye(K, device=Z.device, dtype=Z.dtype)
    L_reg = L + eps * I_K
    L_plus_I_reg = L_reg + I_K  # = L + I + eps*I

    # Original DPP diversity:  LL = log det(L) - log det(L+I)   (maximise)
    # Energy to minimise:      E  = log det(L+I) - log det(L)   (minimise)
    # Divide by (K-1) so that the effective guidance strength is
    # independent of batch size K.  Without this, larger K produces
    # larger determinant → larger gradient → effectively larger γ.
    energy = torch.logdet(L_plus_I_reg) - torch.logdet(L_reg)
    if K > 1:
        energy = energy / (K - 1)
    return energy


def compute_oscar_energy(
    Z: torch.Tensor,
    tau: float = 1.0,
    eps_tr: float | None = None,
) -> torch.Tensor:
    """OSCAR regularised volume energy.

        E = -0.5 * log det(I + tau * Z Z^T + eps_tr * I)

    Args:
        Z: (K, D) feature matrix.
        tau: signal-scaling coefficient.
        eps_tr: ridge regularisation (auto-scaled if None).

    Returns:
        scalar energy.
    """
    K = Z.shape[0]
    Gram = Z @ Z.T  # (K, K)

    if eps_tr is None:
        eps_tr = 1e-6 * torch.trace(Gram) / K

    I_K = torch.eye(K, device=Z.device, dtype=Z.dtype)
    M = I_K + tau * Gram + eps_tr * I_K
    return -0.5 * torch.logdet(M)


# ---------------------------------------------------------------------------
# Orthogonal projection
# ---------------------------------------------------------------------------

def orthogonal_projection(
    g: torch.Tensor,
    direction: torch.Tensor,
    coeff: float = 1.0,
) -> torch.Tensor:
    """Project *g* away from *direction* (per-sample).

        g_ortho = g - coeff * (⟨g, dir⟩ / ‖dir‖²) * dir

    Args:
        g: gradient tensor (K, ...).
        direction: reference direction, same shape as *g*.
        coeff: 1.0 = fully orthogonal; 0.95 = OSCAR default.

    Returns:
        projected gradient, same shape as *g*.
    """
    g_flat = g.reshape(g.shape[0], -1)
    d_flat = direction.reshape(direction.shape[0], -1)

    dot = (g_flat * d_flat).sum(dim=-1)  # (K,)
    norm_sq = (d_flat * d_flat).sum(dim=-1) + 1e-8  # (K,)

    alpha = (dot / norm_sq).view(-1, *([1] * (g.dim() - 1)))
    g_parallel = alpha * direction
    return g - coeff * g_parallel


# ---------------------------------------------------------------------------
# Main step
# ---------------------------------------------------------------------------

def diverse_guidance_step(
    sample: torch.Tensor,
    model_output: torch.Tensor,
    prev_sample: torch.Tensor,
    alpha_prod_t: torch.Tensor,
    t_norm: float,
    config: DiverseGuidanceConfig,
    prediction_type: str = "epsilon",
) -> torch.Tensor:
    """Apply diversity guidance to one diffusion sampling step.

    Call this **after** ``scheduler.step()`` to modify the DDIM/DDPM output.

    Args:
        sample: current noisy sample A_t  (B, ...).
        model_output: raw model prediction (noise or x_0).
        prev_sample: base scheduler output A_{t-1} (before guidance).
        alpha_prod_t: cumulative alpha-product at timestep *t*.
        t_norm: normalised timestep in [0, 1] where 1 = pure noise.
        config: diversity guidance settings.
        prediction_type: ``"epsilon"`` or ``"sample"``.

    Returns:
        Modified *prev_sample* with diversity guidance applied.
    """
    B = sample.shape[0]

    # ---- guard: skip when disabled or batch too small ----------------------
    if B < 2 or config.gamma <= 0.0:
        return prev_sample

    # ---- time gate --------------------------------------------------------
    if not (config.t_gate_end < t_norm <= config.t_gate_start):
        return prev_sample

    # ---- 1. Tweedie / x_0 estimate ----------------------------------------
    # Ensure alpha_prod_t is on the same device as sample
    alpha_prod_t = alpha_prod_t.to(device=sample.device, dtype=sample.dtype)
    if prediction_type == "sample":
        # Model predicts x_0 directly; we compute diversity on x_0 itself.
        x_0_hat = model_output
    elif prediction_type == "epsilon":
        # Tweedie:  x_0 = (x_t - sqrt(1-α̅_t) * ε) / sqrt(α̅_t)
        sqrt_alpha = alpha_prod_t.sqrt()
        sqrt_one_minus = (1.0 - alpha_prod_t).sqrt()
        while sqrt_alpha.dim() < sample.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
        x_0_hat = (sample - sqrt_one_minus * model_output) / sqrt_alpha
    else:
        raise ValueError(f"Unknown prediction_type: {prediction_type}")

    # ---- 2. Diversity energy & gradient -----------------------------------
    with torch.enable_grad():
        sample_grad = sample.detach().requires_grad_(True)

        # Recompute x_0_hat with gradient w.r.t. sample
        if prediction_type == "sample":
            # x_0_hat doesn't depend on sample through the model,
            # so the gradient through sample is zero.
            # Instead, diversity gradient is computed directly on x_0_hat.
            x_0_grad = x_0_hat.detach().requires_grad_(True)
            Z = x_0_grad.reshape(B, -1)
            if config.energy_type == "dpp":
                energy = compute_dpp_energy(
                    Z, h=config.dpp_h, eps=config.eps,
                    use_median_norm=config.dpp_median_norm,
                )
            else:
                energy = compute_oscar_energy(Z, tau=config.oscar_tau)
            # Gradient w.r.t. x_0 (not sample)
            g_d = torch.autograd.grad(energy, x_0_grad)[0]
        else:
            sqrt_alpha = alpha_prod_t.sqrt()
            sqrt_one_minus = (1.0 - alpha_prod_t).sqrt()
            while sqrt_alpha.dim() < sample_grad.dim():
                sqrt_alpha = sqrt_alpha.unsqueeze(-1)
                sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
            x_0_hat_grad = (
                sample_grad - sqrt_one_minus * model_output.detach()
            ) / sqrt_alpha

            Z = x_0_hat_grad.reshape(B, -1)
            if config.energy_type == "dpp":
                energy = compute_dpp_energy(
                    Z, h=config.dpp_h, eps=config.eps,
                    use_median_norm=config.dpp_median_norm,
                )
            else:
                energy = compute_oscar_energy(Z, tau=config.oscar_tau)

            # Gradient of energy w.r.t. current noisy sample
            g_d = torch.autograd.grad(energy, sample_grad)[0]

    # ---- 3. Base update direction -----------------------------------------
    base_direction = prev_sample - sample  # (B, ...)

    # ---- 4. Orthogonal projection -----------------------------------------
    g_orthogonal = orthogonal_projection(g_d, base_direction, config.ortho_coeff)

    # ---- 5. Time-dependent scaling (linear decay towards t_gate_end) ------
    frac = (t_norm - config.t_gate_end) / (config.t_gate_start - config.t_gate_end)
    frac = max(0.0, min(1.0, frac))
    gamma_eff = config.gamma * frac

    # ---- 6. Orthogonal stochastic noise (optional) ------------------------
    if config.eta_sde > 0.0:
        noise = torch.randn_like(sample)
        noise_orthogonal = orthogonal_projection(
            noise, base_direction, config.ortho_coeff
        )
    else:
        noise_orthogonal = torch.zeros_like(sample)

    # ---- 7. Apply guidance ------------------------------------------------
    prev_sample = (
        prev_sample
        - gamma_eff * g_orthogonal
        + config.eta_sde * t_norm * noise_orthogonal
    )

    return prev_sample.detach()


# ---------------------------------------------------------------------------
# Convenience: create config & integrate into a scheduler loop
# ---------------------------------------------------------------------------

def make_diverse_config(
    gamma: float = 0.1,
    energy_type: Literal["dpp", "oscar"] = "dpp",
    ortho_coeff: float = 0.95,
    eta_sde: float = 0.0,
    t_gate_end: float = 0.05,
    **kwargs,
) -> DiverseGuidanceConfig:
    """Create a DiverseGuidanceConfig with common defaults."""
    return DiverseGuidanceConfig(
        gamma=gamma,
        energy_type=energy_type,
        ortho_coeff=ortho_coeff,
        eta_sde=eta_sde,
        t_gate_end=t_gate_end,
        **kwargs,
    )

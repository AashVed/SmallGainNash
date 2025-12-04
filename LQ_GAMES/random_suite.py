"""
Random LQ ensemble for SGN robustness.

This module constructs a suite of random 2-player LQ games and computes,
for a small set of coupling values λ,

  - Euclidean monotonicity margins,
  - true and SGN monotonicity margins in a fixed SGN metric,
  - Lipschitz constants and SGN-based RK4 step bounds,
  - approximate true RK4 stability thresholds.

The results can be aggregated to show that the qualitative pattern observed
in the clean LQ game (SGN extends beyond Euclidean and yields conservative
but nontrivial step bounds) holds generically.
"""

from __future__ import annotations

import pathlib
from typing import Tuple

import numpy as np

from .config import GAME_CONFIG, LAMBDA_GRID, WEIGHT_SEARCH
from .compute_sgn_metrics import (
    WeightConfig,
    build_H,
    build_metric_matrix,
    euclidean_monotonicity_margin,
    lipschitz_constant,
    sgn_margin,
    true_monotonicity_margin,
)
from .compute_stability import rk4_step_matrix, spectral_radius


def generate_random_game(seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a random 2-player LQ game instance: Q_1, Q_2, and R.

    Q_i = R_i^T R_i + mu0 I, with R_i random Gaussian.
    R is a normalized cross-player base matrix with spectral norm 1.
    """

    cfg = GAME_CONFIG
    rng = np.random.default_rng(seed)

    R1 = rng.standard_normal((cfg.d1, cfg.d1))
    R2 = rng.standard_normal((cfg.d2, cfg.d2))
    Q1 = R1.T @ R1 + cfg.mu0 * np.eye(cfg.d1)
    Q2 = R2.T @ R2 + cfg.mu0 * np.eye(cfg.d2)

    R_base = rng.standard_normal((cfg.d1, cfg.d2))
    u, s, vt = np.linalg.svd(R_base, full_matrices=False)
    spectral_norm = s[0]
    if spectral_norm == 0:
        raise RuntimeError("Sampled cross-player base matrix has zero norm")
    R = R_base / spectral_norm

    return Q1, Q2, R


def approximate_rk4_stability_threshold(H: np.ndarray, beta: float, num_h_points: int = 80) -> float:
    """
    Approximate the RK4 stability threshold h_stab for x' = -H x.

    We scan a grid in h between 0 and h_max and return the largest h for which
    the spectral radius of the RK4 one-step matrix is < 1.
    """

    if beta <= 0:
        return 0.0

    # Conservative upper bound for h: something proportional to 1/beta.
    h_max = 4.0 / beta
    h_grid = np.linspace(0.0, h_max, num_h_points)
    rho_vals = []
    for h in h_grid:
        T = rk4_step_matrix(H, h)
        rho = spectral_radius(T)
        rho_vals.append(rho)
    rho_vals = np.asarray(rho_vals)
    stable_mask = rho_vals < 1.0
    if np.any(stable_mask):
        return float(h_grid[stable_mask].max())
    return 0.0


def run_random_suite(
    num_seeds: int = 30,
    lambda_indices: Tuple[int, ...] | None = None,
    out_path: pathlib.Path | str = "LQ_GAMES/data/random_suite.npz",
) -> pathlib.Path:
    """
    Run the random LQ ensemble and save aggregated metrics.

    Parameters
    ----------
    num_seeds : int
        Number of random game instances.
    lambda_indices : tuple of int, optional
        Indices into the global λ grid at which to evaluate metrics.  If None,
        we choose three representative indices (low, mid, high).
    out_path : path-like
        Output .npz file to write.
    """

    cfg = GAME_CONFIG
    lambdas_full = LAMBDA_GRID.values()

    if lambda_indices is None:
        # Choose multiple λ-indices across the grid (including both
        # Euclidean-monotone and SGN-only regimes if available).
        n = lambdas_full.size
        lambda_indices = tuple(
            sorted(
                set(
                    int(round(idx))
                    for idx in np.linspace(0, n - 1, num=7)
                )
            )
        )

    lambda_indices = tuple(sorted(set(lambda_indices)))
    lambda_subset = lambdas_full[list(lambda_indices)]

    S = num_seeds
    L = len(lambda_indices)

    # Arrays indexed by (seed, lambda_index).
    gamma_euc = np.zeros((S, L))
    alpha_true = np.zeros((S, L))
    alpha_sgn = np.zeros((S, L))
    beta_vals = np.zeros((S, L))
    h_sgn_rk4 = np.zeros((S, L))
    h_stab_rk4 = np.zeros((S, L))

    w_cfg = WeightConfig(mode="balanced")
    w = w_cfg.weights(a=cfg.a, b=cfg.b)

    for s in range(S):
        seed = cfg.seed + 1000 * s
        Q1, Q2, R = generate_random_game(seed)

        # Own-player curvature parameters via smallest eigenvalues of Q_i.
        mu1 = float(np.linalg.eigvalsh(Q1)[0])
        mu2 = float(np.linalg.eigvalsh(Q2)[0])

        d1, d2 = Q1.shape[0], Q2.shape[0]
        M = build_metric_matrix(w, d1=d1, d2=d2)

        for j, idx in enumerate(lambda_indices):
            lam = float(lambdas_full[idx])
            H = build_H(Q1, Q2, R, lam=lam, a=cfg.a, b=cfg.b)

            # Euclidean margin
            gamma_euc[s, j] = euclidean_monotonicity_margin(H)

            # SGN margin and true margin
            L12 = lam * cfg.a
            L21 = lam * cfg.b
            alpha_star = sgn_margin(mu1, mu2, L12, L21, w=w)
            alpha_sgn[s, j] = alpha_star

            alpha_t = true_monotonicity_margin(H, M)
            alpha_true[s, j] = alpha_t

            # Lipschitz and step bounds for RK4
            beta = lipschitz_constant(H, M)
            beta_vals[s, j] = beta

            if alpha_star > 0.0 and beta > 0.0:
                h_sgn_rk4[s, j] = WEIGHT_SEARCH.rk4_c4 / beta
            else:
                h_sgn_rk4[s, j] = 0.0

            # Approximate true RK4 stability threshold.
            if beta > 0.0:
                h_stab = approximate_rk4_stability_threshold(H, beta)
            else:
                h_stab = 0.0
            h_stab_rk4[s, j] = h_stab

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        lambdas_full=lambdas_full,
        lambda_indices=np.array(lambda_indices, dtype=int),
        lambdas=lambda_subset,
        gamma_euc=gamma_euc,
        alpha_true=alpha_true,
        alpha_sgn=alpha_sgn,
        beta=beta_vals,
        h_sgn_rk4=h_sgn_rk4,
        h_stab_rk4=h_stab_rk4,
    )

    return out_path


def main() -> None:
    """
    Command-line entry point.

    Runs the random LQ ensemble and saves metrics to `LQ_GAMES/data/random_suite.npz`.
    """

    path = run_random_suite()
    print(f"Saved random LQ ensemble metrics to {path}")


if __name__ == "__main__":
    main()

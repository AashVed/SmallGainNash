"""
Compute discrete-time stability regions for Euler and RK4.

Given the game matrices and the precomputed SGN metrics, this script:

  - builds the Euler and RK4 one-step matrices T(λ, h),
  - evaluates their spectral radii over a grid of step sizes,
  - approximates the true stability thresholds h_stab(λ) where ρ(T) = 1,
  - stores the spectral radius grids and thresholds for later analysis
    and plotting.
"""

from __future__ import annotations

import pathlib
from typing import Dict, Tuple

import numpy as np
from numpy.linalg import eigvals

from .config import LAMBDA_GRID
from .compute_sgn_metrics import build_H, load_game


def euler_step_matrix(H: np.ndarray, h: float) -> np.ndarray:
    """One-step matrix for forward Euler: T = I - h H."""

    d = H.shape[0]
    return np.eye(d) - h * H


def rk4_step_matrix(H: np.ndarray, h: float) -> np.ndarray:
    """
    One-step matrix for RK4 on the linear system x' = -H x.

    For a linear system, RK4 is equivalent to applying the stability
    polynomial P(-h H) with

        P(z) = 1 + z + z^2/2 + z^3/6 + z^4/24.
    """

    d = H.shape[0]
    I = np.eye(d)
    H1 = -h * H
    H2 = H1 @ H1
    H3 = H2 @ H1
    H4 = H3 @ H1
    return I + H1 + 0.5 * H2 + (1.0 / 6.0) * H3 + (1.0 / 24.0) * H4


def spectral_radius(T: np.ndarray) -> float:
    """Return the spectral radius ρ(T)."""

    eigs = eigvals(T)
    return float(np.max(np.abs(eigs)))


def compute_stability_grid(
    metrics_path: pathlib.Path | str = "LQ_GAMES/data/metrics_balanced.npz",
    game_path: pathlib.Path | str = "LQ_GAMES/data/game.npz",
    out_dir: pathlib.Path | str = "LQ_GAMES/data",
    num_h_points: int = 400,
) -> pathlib.Path:
    """
    Compute spectral radii over a grid of step sizes for Euler and RK4.

    For each λ in the predefined grid and each method, we construct H(λ),
    and then evaluate ρ(T(λ, h)) on a linearly spaced grid in h between 0 and
    a conservative upper bound.  We then approximate the true stability
    threshold h_stab(λ) as the largest h for which ρ(T(λ, h)) < 1.
    """

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = np.load(metrics_path)
    game = load_game(game_path)

    lambdas = metrics["lambdas"]
    a = float(metrics["a"])
    b = float(metrics["b"])

    Q1 = game["Q1"]
    Q2 = game["Q2"]
    R = game["R"]

    # Choose a conservative global upper bound for step sizes by looking at
    # the SGN-based bounds and backing off by a factor, then extending a bit.
    euler_sgn = metrics["euler_step_sgn"]
    rk4_sgn = metrics["rk4_step_sgn"]

    euler_h_max_global = float(np.nanmax(euler_sgn) if np.any(euler_sgn > 0) else 1.0)
    rk4_h_max_global = float(np.nanmax(rk4_sgn) if np.any(rk4_sgn > 0) else 1.0)

    # Allow exploration somewhat beyond the SGN bound.
    euler_h_max_global *= 4.0
    rk4_h_max_global *= 4.0

    euler_h_grid = np.linspace(0.0, euler_h_max_global, num_h_points)
    rk4_h_grid = np.linspace(0.0, rk4_h_max_global, num_h_points)

    # Arrays to store spectral radii and stability thresholds.
    rho_euler = np.zeros((lambdas.size, euler_h_grid.size))
    rho_rk4 = np.zeros((lambdas.size, rk4_h_grid.size))
    h_stab_euler = np.zeros(lambdas.size)
    h_stab_rk4 = np.zeros(lambdas.size)

    for i, lam in enumerate(lambdas):
        H = build_H(Q1, Q2, R, lam=lam, a=a, b=b)

        # Euler spectral radii
        for j, h in enumerate(euler_h_grid):
            T = euler_step_matrix(H, h)
            rho_euler[i, j] = spectral_radius(T)

        # Approximate Euler stability threshold with grid + local refinement.
        stable_mask = rho_euler[i, :] < 1.0
        if np.any(stable_mask):
            h_lo = euler_h_grid[stable_mask].max()
            unstable_mask = rho_euler[i, :] >= 1.0
            h_hi_candidates = euler_h_grid[unstable_mask]
            h_hi = h_hi_candidates[h_hi_candidates > h_lo].min() if h_hi_candidates.size else euler_h_grid[-1]
            # Bisection refinement around the transition.
            for _ in range(20):
                h_mid = 0.5 * (h_lo + h_hi)
                rho_mid = spectral_radius(euler_step_matrix(H, h_mid))
                if rho_mid < 1.0:
                    h_lo = h_mid
                else:
                    h_hi = h_mid
            h_stab_euler[i] = h_lo
        else:
            h_stab_euler[i] = 0.0

        # RK4 spectral radii
        for j, h in enumerate(rk4_h_grid):
            T = rk4_step_matrix(H, h)
            rho_rk4[i, j] = spectral_radius(T)

        stable_mask = rho_rk4[i, :] < 1.0
        if np.any(stable_mask):
            h_lo = rk4_h_grid[stable_mask].max()
            unstable_mask = rho_rk4[i, :] >= 1.0
            h_hi_candidates = rk4_h_grid[unstable_mask]
            h_hi = h_hi_candidates[h_hi_candidates > h_lo].min() if h_hi_candidates.size else rk4_h_grid[-1]
            for _ in range(20):
                h_mid = 0.5 * (h_lo + h_hi)
                rho_mid = spectral_radius(rk4_step_matrix(H, h_mid))
                if rho_mid < 1.0:
                    h_lo = h_mid
                else:
                    h_hi = h_mid
            h_stab_rk4[i] = h_lo
        else:
            h_stab_rk4[i] = 0.0

    out_path = out_dir / "stability.npz"
    np.savez_compressed(
        out_path,
        lambdas=lambdas,
        euler_h_grid=euler_h_grid,
        rk4_h_grid=rk4_h_grid,
        rho_euler=rho_euler,
        rho_rk4=rho_rk4,
        h_stab_euler=h_stab_euler,
        h_stab_rk4=h_stab_rk4,
    )

    return out_path


def main() -> None:
    """
    Command-line entry point.

    Computes stability grids and thresholds for Euler and RK4 and saves them
    to `LQ_GAMES/data/stability.npz`.
    """

    path = compute_stability_grid()
    print(f"Saved discrete-time stability data to {path}")


if __name__ == "__main__":
    main()

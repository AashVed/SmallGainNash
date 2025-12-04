"""
Simulate discrete-time Euler and RK4 dynamics for visualization.

While stability is classified exactly via spectral radii, it is useful to
visualize actual trajectories and estimate empirical contraction factors.
This script:

  - loads the game, metrics, and stability data,
  - selects a subset of (λ, h) pairs in the stable region,
  - runs Euler and RK4 updates on random initial conditions,
  - logs norms in the SGN metric and empirical per-step contraction factors.
"""

from __future__ import annotations

import pathlib
from typing import Dict, Tuple

import numpy as np

from .config import GAME_CONFIG, LAMBDA_GRID, SIM_CONFIG
from .compute_sgn_metrics import build_H, build_metric_matrix, load_game
from .compute_stability import euler_step_matrix, rk4_step_matrix


def simulate_trajectories(
    metrics_path: pathlib.Path | str = "LQ_GAMES/data/metrics_balanced.npz",
    stability_path: pathlib.Path | str = "LQ_GAMES/data/stability.npz",
    game_path: pathlib.Path | str = "LQ_GAMES/data/game.npz",
    out_dir: pathlib.Path | str = "LQ_GAMES/data",
    max_lambda_indices: int = 30,
) -> pathlib.Path:
    """
    Simulate trajectories for a subset of (λ, h) pairs.

    We pick up to `max_lambda_indices` λ-values spread across the grid, and
    for each we simulate a few step sizes:

      - one well inside the SGN region,
      - one near the SGN bound,
      - one between SGN and the true stability boundary.
    """

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = np.load(metrics_path)
    stability = np.load(stability_path)
    game = load_game(game_path)

    lambdas = metrics["lambdas"]
    w = metrics["w"]
    a = float(metrics["a"])
    b = float(metrics["b"])

    euler_step_sgn = metrics["euler_step_sgn"]
    rk4_step_sgn = metrics["rk4_step_sgn"]

    euler_h_grid = stability["euler_h_grid"]
    rk4_h_grid = stability["rk4_h_grid"]
    rho_euler = stability["rho_euler"]
    rho_rk4 = stability["rho_rk4"]

    Q1 = game["Q1"]
    Q2 = game["Q2"]
    R = game["R"]

    d1, d2 = Q1.shape[0], Q2.shape[0]
    M = build_metric_matrix(w, d1=d1, d2=d2)

    rng = np.random.default_rng(GAME_CONFIG.seed + 1)

    # Choose a small set of λ indices spaced across the grid.
    if lambdas.size <= max_lambda_indices:
        lambda_indices = np.arange(lambdas.size)
    else:
        lambda_indices = np.linspace(0, lambdas.size - 1, max_lambda_indices).astype(int)

    # Container for outputs
    records = []

    for idx in lambda_indices:
        lam = float(lambdas[idx])
        H = build_H(Q1, Q2, R, lam=lam, a=a, b=b)

        # Euler step sizes: pick a richer set of representatives spanning
        # well inside SGN, near the SGN bound, and between the SGN and
        # true stability thresholds.
        euler_candidates: list[float] = []
        if euler_step_sgn[idx] > 0:
            e_sgn = float(euler_step_sgn[idx])
            for mult in (0.2, 0.35, 0.5, 0.65, 0.8, 0.9):
                euler_candidates.append(mult * e_sgn)
        # Add steps between SGN bound and approximate stability threshold if possible.
        stable_mask = rho_euler[idx, :] < 1.0
        if np.any(stable_mask):
            h_stab = euler_h_grid[stable_mask].max()
            if h_stab > euler_step_sgn[idx] > 0:
                mid = 0.5 * (euler_step_sgn[idx] + h_stab)
                euler_candidates.append(mid)
                euler_candidates.append(0.9 * h_stab)

        # RK4 step sizes similarly.
        rk4_candidates: list[float] = []
        if rk4_step_sgn[idx] > 0:
            r_sgn = float(rk4_step_sgn[idx])
            for mult in (0.2, 0.35, 0.5, 0.65, 0.8, 0.9):
                rk4_candidates.append(mult * r_sgn)
        stable_mask = rho_rk4[idx, :] < 1.0
        if np.any(stable_mask):
            h_stab = rk4_h_grid[stable_mask].max()
            if h_stab > rk4_step_sgn[idx] > 0:
                mid = 0.5 * (rk4_step_sgn[idx] + h_stab)
                rk4_candidates.append(mid)
                rk4_candidates.append(0.9 * h_stab)

        # Deduplicate and keep positive step sizes.
        euler_candidates = sorted({h for h in euler_candidates if h > 0})
        rk4_candidates = sorted({h for h in rk4_candidates if h > 0})

        for method, h_list in (("euler", euler_candidates), ("rk4", rk4_candidates)):
            for h in h_list:
                # Determine one-step matrix
                if method == "euler":
                    T = euler_step_matrix(H, h)
                else:
                    T = rk4_step_matrix(H, h)

                # Stability check
                # (we only simulate for numerically stable configurations)
                eigs = np.linalg.eigvals(T)
                rho = float(np.max(np.abs(eigs)))
                if rho >= 1.0:
                    continue

                # Simulate multiple seeds and estimate empirical contraction factor.
                seed_results = []
                for s in range(SIM_CONFIG.num_seeds):
                    x0 = rng.normal(loc=0.0, scale=SIM_CONFIG.sigma0, size=(d1 + d2,))
                    norms = []
                    x = x0.copy()
                    for k in range(SIM_CONFIG.max_steps):
                        # M-norm: sqrt(x^T M x)
                        norm_M = float(np.sqrt(x.T @ (M @ x)))
                        norms.append(norm_M)
                        x = T @ x
                    norms = np.array(norms)
                    # Fit a line to log norms over the last half of the trajectory.
                    eps = 1e-12
                    log_norms = np.log(norms + eps)
                    ks = np.arange(SIM_CONFIG.max_steps)
                    start = SIM_CONFIG.max_steps // 4
                    A = np.vstack([ks[start:], np.ones_like(ks[start:])]).T
                    slope, intercept = np.linalg.lstsq(A, log_norms[start:], rcond=None)[0]
                    rho_emp = float(np.exp(slope))
                    seed_results.append((norms[0], norms[-1], rho_emp))

                seed_results = np.array(seed_results)
                rho_emp_median = float(np.median(seed_results[:, 2]))
                record = {
                    "lambda": lam,
                    "lambda_index": int(idx),
                    "method": method,
                    "h": float(h),
                    "rho_spectral": rho,
                    "rho_emp_median": rho_emp_median,
                }
                records.append(record)

    # Save the simulation summary as a numpy structured array.
    dtype = [
        ("lambda", float),
        ("lambda_index", int),
        ("method", "U8"),
        ("h", float),
        ("rho_spectral", float),
        ("rho_emp_median", float),
    ]
    out = np.zeros(len(records), dtype=dtype)
    for i, rec in enumerate(records):
        out[i]["lambda"] = rec["lambda"]
        out[i]["lambda_index"] = rec["lambda_index"]
        out[i]["method"] = rec["method"]
        out[i]["h"] = rec["h"]
        out[i]["rho_spectral"] = rec["rho_spectral"]
        out[i]["rho_emp_median"] = rec["rho_emp_median"]

    out_path = pathlib.Path(out_dir) / "simulation_summary.npy"
    np.save(out_path, out)
    return out_path


def main() -> None:
    """
    Command-line entry point.

    Runs discrete-time simulations for a subset of (λ, h) pairs and saves a
    summary of empirical contraction factors to
    `LQ_GAMES/data/simulation_summary.npy`.
    """

    path = simulate_trajectories()
    print(f"Saved discrete-time simulation summary to {path}")


if __name__ == "__main__":
    main()

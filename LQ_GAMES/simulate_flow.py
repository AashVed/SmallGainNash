"""
Simulate the continuous-time flow x' = -H(λ) x for selected λ.

For a few representative values of the coupling parameter λ, this script:

  - integrates the ODE using a small-step RK4 scheme in the SGN metric,
  - logs ‖x(t)‖_{M(w)} for multiple random initial conditions,
  - fits exponential decay rates from the time series,
  - compares empirical decay rates to the true monotonicity margin α_true
    and the SGN bound α_*.

The primary purpose is to provide sanity checks and optional figures for the
continuous-time contraction story in the paper.
"""

from __future__ import annotations

import pathlib
from typing import Dict, Tuple

import numpy as np

from .config import FLOW_CONFIG, GAME_CONFIG
from .compute_sgn_metrics import build_H, build_metric_matrix, load_game


def rk4_step(G, x, dt):
    """One step of RK4 for x' = G(x)."""

    k1 = G(x)
    k2 = G(x + 0.5 * dt * k1)
    k3 = G(x + 0.5 * dt * k2)
    k4 = G(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_flow_for_lambda(
    lam: float,
    Q1: np.ndarray,
    Q2: np.ndarray,
    R: np.ndarray,
    w: np.ndarray,
    M: np.ndarray,
    alpha_true_lam: float,
    alpha_sgn_lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate the flow for a given λ and return times and median norms.
    """

    d1, d2 = Q1.shape[0], Q2.shape[0]
    H = build_H(Q1, Q2, R, lam=lam, a=float(a_global), b=float(b_global))

    def G(x):
        return -H @ x

    dt = FLOW_CONFIG.dt
    num_steps = int(FLOW_CONFIG.horizon / dt)
    times = np.linspace(0.0, FLOW_CONFIG.horizon, num_steps + 1)

    rng = np.random.default_rng(GAME_CONFIG.seed + int(100 * lam))

    norms_all = []
    for s in range(FLOW_CONFIG.num_seeds):
        x = rng.normal(loc=0.0, scale=1.0, size=(d1 + d2,))
        norms = []
        for k in range(num_steps + 1):
            norms.append(float(np.sqrt(x.T @ (M @ x))))
            if k < num_steps:
                x = rk4_step(G, x, dt)
        norms_all.append(norms)

    norms_all = np.asarray(norms_all)
    median_norms = np.median(norms_all, axis=0)
    return times, median_norms


def simulate_flows(
    metrics_path: pathlib.Path | str = "LQ_GAMES/data/metrics_balanced.npz",
    game_path: pathlib.Path | str = "LQ_GAMES/data/game.npz",
    out_dir: pathlib.Path | str = "LQ_GAMES/data",
    figs_dir: pathlib.Path | str = "figs/LQ_GAME",
) -> pathlib.Path:
    """
    Run continuous-time simulations for a few selected λ values and save
    summary data and an optional figure.
    """

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figs_dir = pathlib.Path(figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)

    metrics = np.load(metrics_path)
    game = load_game(game_path)

    lambdas = metrics["lambdas"]
    alpha_sgn = metrics["alpha_sgn"]
    alpha_true = metrics["alpha_true"]
    w = metrics["w"]
    global a_global, b_global
    a_global = float(metrics["a"])
    b_global = float(metrics["b"])

    Q1 = game["Q1"]
    Q2 = game["Q2"]
    R = game["R"]
    d1, d2 = Q1.shape[0], Q2.shape[0]
    M = build_metric_matrix(w, d1=d1, d2=d2)

    # Choose three λ values: Euclidean monotone, SGN-only, and outside SGN (if available).
    gamma_euc = metrics["gamma_euc"]
    idx_euc = np.where(gamma_euc > 0.0)[0]
    idx_sgn_only = np.where((alpha_sgn > 0.0) & (gamma_euc < 0.0))[0]
    idx_outside = np.where(alpha_sgn <= 0.0)[0]

    chosen_indices = []
    if idx_euc.size > 0:
        chosen_indices.append(idx_euc[-1])  # largest λ still Euclidean monotone
    if idx_sgn_only.size > 0:
        chosen_indices.append(idx_sgn_only[len(idx_sgn_only) // 2])
    if idx_outside.size > 0:
        chosen_indices.append(idx_outside[0])

    # Deduplicate and limit to at most three.
    chosen_indices = sorted(set(chosen_indices))[:3]
    if not chosen_indices:
        # No interesting separation; fall back to first, middle, last λ.
        chosen_indices = [0, len(lambdas) // 2, len(lambdas) - 1]

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    records = []
    fig, axes = plt.subplots(
        1, len(chosen_indices), figsize=(5 * len(chosen_indices), 4), sharey=True
    )
    if len(chosen_indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, chosen_indices):
        lam = float(lambdas[idx])
        times, median_norms = simulate_flow_for_lambda(
            lam,
            Q1,
            Q2,
            R,
            w,
            M,
            alpha_true_lam=float(alpha_true[idx]),
            alpha_sgn_lam=float(alpha_sgn[idx]),
        )

        # Fit an empirical decay rate from log norms.
        eps = 1e-12
        log_norms = np.log(median_norms + eps)
        # Fit over the second half of the time horizon.
        start = len(times) // 4
        A = np.vstack([times[start:], np.ones_like(times[start:])]).T
        slope, intercept = np.linalg.lstsq(A, log_norms[start:], rcond=None)[0]
        alpha_emp = -float(slope)

        records.append(
            (lam, float(alpha_true[idx]), float(alpha_sgn[idx]), alpha_emp)
        )

        # Median norm trajectory.
        ax.plot(
            times,
            median_norms,
            color="C0",
            linewidth=1.0,
            label=r"median $\|x(t)\|_{M(w)}$",
        )
        # Reference line for the true metric margin.
        ax.plot(
            times,
            median_norms[0] * np.exp(-alpha_true[idx] * (times - times[0])),
            color="k",
            linestyle="-",
            linewidth=2.0,
            label=r"$e^{-\alpha_{\mathrm{true}} t}$",
        )
        # Reference line for the SGN bound (when it certifies).
        if alpha_sgn[idx] > 0.0:
            ax.plot(
                times,
                median_norms[0] * np.exp(-alpha_sgn[idx] * (times - times[0])),
                color="tab:orange",
                linestyle="--",
                linewidth=2.0,
                label=r"$e^{-\alpha_* t}$",
            )
        else:
            # Make it explicit when SGN does not certify a positive margin.
            ax.text(
                0.05,
                0.05,
                r"$\alpha_* \leq 0$ (no SGN bound)",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=10,
                color="tab:orange",
            )

        ax.set_xlabel("Time $t$", fontsize=11)
        ax.set_title(rf"$\lambda = {lam:.2g}$", fontsize=12)
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(labelsize=10)

    axes[0].set_ylabel(r"$\|x(t)\|_{M(w)}$", fontsize=11)

    # Use a single, shared legend for all panels.
    has_sgn = any(alpha_sgn[idx] > 0.0 for idx in chosen_indices)
    legend_handles = [
        Line2D([], [], color="C0", linewidth=1.4, label=r"median $\|x(t)\|_{M(w)}$"),
        Line2D([], [], color="k", linestyle="--", linewidth=1.4, label=r"$e^{-\alpha_{\mathrm{true}} t}$"),
    ]
    if has_sgn:
        legend_handles.append(
            Line2D(
                [],
                [],
                color="tab:orange",
                linestyle="-.",
                linewidth=1.4,
                label=r"$e^{-\alpha_* t}$",
            )
        )
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        fontsize=11,
        frameon=True,
        framealpha=0.9,
    )

    fig.suptitle("Continuous-time SGN flow: median norms vs time")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_fig = figs_dir / "lq_flow_timeseries.pdf"
    fig.savefig(out_fig)
    fig.savefig(out_fig.with_suffix(".png"), dpi=300)
    plt.close(fig)

    # Save summary of decay rates.
    dtype = [
        ("lambda", float),
        ("alpha_true", float),
        ("alpha_sgn", float),
        ("alpha_emp", float),
    ]
    arr = np.zeros(len(records), dtype=dtype)
    for i, (lam, at, asgn, aemp) in enumerate(records):
        arr[i]["lambda"] = lam
        arr[i]["alpha_true"] = at
        arr[i]["alpha_sgn"] = asgn
        arr[i]["alpha_emp"] = aemp

    out_path = out_dir / "flow_summary.npy"
    np.save(out_path, arr)
    return out_path


def main() -> None:
    """
    Command-line entry point.

    Runs continuous-time simulations for a few λ values and saves a summary
    of empirical decay rates, as well as a time-series figure for inclusion
    in the paper.
    """

    path = simulate_flows()
    print(f"Saved continuous-time flow summary to {path}")


if __name__ == "__main__":
    main()

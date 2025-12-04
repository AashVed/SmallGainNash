"""
Unified plotting utilities for the LQ SGN experiments.

This module consolidates the plotting logic that was previously split across
multiple scripts. The single entry point `plot_all_figures` regenerates every
figure used in the paper:

  - Timescale band vs weight ratio r (Experiment A).
  - Phase diagrams and margin curves (Experiment B).
  - Random ensemble summaries.
  - Structured noise robustness (ε-sweep).
"""

from __future__ import annotations

import pathlib
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .compute_sgn_metrics import (
    build_H,
    build_metric_matrix,
    load_game,
    sgn_margin,
    true_monotonicity_margin,
    two_player_band,
)
from .compute_stability import euler_step_matrix, rk4_step_matrix
from .config import GAME_CONFIG, NOISE_CONFIG, WEIGHT_SEARCH
from .random_suite import run_random_suite


def _ensure_dir(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_timescale_band(
    metrics_path: pathlib.Path | str = "LQ_GAMES/data/metrics_balanced.npz",
    figs_dir: pathlib.Path | str = "figs/LQ_GAME",
) -> None:
    """
    Plot α_true(r) and α_*(r) at λ = lambda_timescale on a log-scale r axis.
    """

    figs_dir = _ensure_dir(pathlib.Path(figs_dir))
    metrics = np.load(metrics_path)
    ratios = metrics["ratio_grid"]
    alpha_true_grid = metrics["alpha_true_grid"]
    alpha_sgn_grid = metrics["alpha_sgn_grid"]
    idx_ts = int(metrics["timescale_index"])
    lam_ts = float(metrics["timescale_lambda"])
    r_minus = float(metrics["timescale_r_minus"])
    r_plus = float(metrics["timescale_r_plus"])

    alpha_true = alpha_true_grid[:, idx_ts]
    alpha_sgn = alpha_sgn_grid[:, idx_ts]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    ax.plot(ratios, alpha_true, label=r"True margin $\alpha_{\mathrm{true}}(r)$", color="#1f77b4")
    ax.plot(ratios, alpha_sgn, label=r"SGN margin $\alpha_*(r)$", color="#2ca02c", linestyle="--")
    ax.axhline(0.0, color="gray", linewidth=1.0)
    if np.isfinite(r_minus) and np.isfinite(r_plus):
        ax.axvspan(r_minus, r_plus, color="#FFDDC1", alpha=0.5, label="SGN feasibility band (α=0)")
        ax.axvline(r_minus, color="#FF7F0E", linestyle=":", linewidth=1.2)
        ax.axvline(r_plus, color="#FF7F0E", linestyle=":", linewidth=1.2)

    ax.set_xscale("log")
    ax.set_xlabel(r"Weight ratio $r = w_2 / w_1$")
    ax.set_ylabel("Margin")
    ax.set_title(rf"Timescale band at $\lambda = {lam_ts:.2g}$")
    ax.legend(loc="best")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_pdf = figs_dir / "lq_timescale_band.pdf"
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_phase_diagrams_and_margins(
    metrics_path: pathlib.Path | str = "LQ_GAMES/data/metrics_balanced.npz",
    stability_path: pathlib.Path | str = "LQ_GAMES/data/stability.npz",
    sim_summary_path: Optional[pathlib.Path | str] = "LQ_GAMES/data/simulation_summary.npy",
    figs_dir: pathlib.Path | str = "figs/LQ_GAME",
) -> None:
    """
    Recreate the (λ, h) phase diagrams and λ → margin curves.
    """

    figs_dir = _ensure_dir(pathlib.Path(figs_dir))
    metrics = np.load(metrics_path)
    stability = np.load(stability_path)

    lambdas = metrics["lambdas"]
    euler_step_sgn = metrics["euler_step_sgn"]
    rk4_step_sgn = metrics["rk4_step_sgn"]
    gamma_euc = metrics["gamma_euc"]

    euler_h_grid = stability["euler_h_grid"]
    rk4_h_grid = stability["rk4_h_grid"]
    rho_euler = stability["rho_euler"]
    rho_rk4 = stability["rho_rk4"]
    h_stab_euler = stability["h_stab_euler"]
    h_stab_rk4 = stability["h_stab_rk4"]

    sim_data = None
    if sim_summary_path is not None and pathlib.Path(sim_summary_path).exists():
        sim_data = np.load(sim_summary_path)

    def _plot_panel(method: str) -> None:
        if method == "euler":
            h_grid = euler_h_grid
            rho = rho_euler
            h_stab = h_stab_euler
            h_sgn = euler_step_sgn
            fname = "lq_phase_euler.pdf"
        else:
            h_grid = rk4_h_grid
            rho = rho_rk4
            h_stab = h_stab_rk4
            h_sgn = rk4_step_sgn
            fname = "lq_phase_rk4.pdf"

        stable_mask = rho < 1.0
        rho_plot = np.full_like(rho, np.nan, dtype=float)
        rho_plot[stable_mask] = np.log(rho[stable_mask])
        vmin = float(np.nanmin(rho_plot)) if np.any(np.isfinite(rho_plot)) else -1.0

        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        im = ax.pcolormesh(
            lambdas,
            h_grid,
            rho_plot.T,
            cmap="viridis",
            shading="auto",
            vmin=vmin,
            vmax=0.0,
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r"$\log \rho(T)$ for $\rho<1$")

        ax.plot(
            lambdas,
            h_sgn,
            color="#FF7F0E",
            linestyle="--",
            linewidth=1.8,
            label=r"SGN bound",
        )
        ax.plot(
            lambdas,
            h_stab,
            color="#D62728",
            linestyle="-",
            linewidth=2.2,
            label="True stability",
        )

        if sim_data is not None:
            mask = sim_data["method"] == method
            if np.any(mask):
                subset = sim_data[mask]
                rho_emp = subset["rho_emp_median"]
                rho_spec = subset["rho_spectral"]
                stable = rho_spec < 1.0
                unstable = ~stable
                if np.any(stable):
                    ax.scatter(
                        subset["lambda"][stable],
                        subset["h"][stable],
                        color="#2ca02c",
                        edgecolors="k",
                        s=14,
                        marker="o",
                        label="simulated (stable)",
                        linewidths=0.6,
                    )
                if np.any(unstable):
                    ax.scatter(
                        subset["lambda"][unstable],
                        subset["h"][unstable],
                        color="#d62728",
                        edgecolors="k",
                        s=22,
                        marker="X",
                        label="simulated (unstable)",
                        linewidths=0.6,
                    )

        mono_mask = gamma_euc >= 0.0
        if np.any(mono_mask):
            lam_min = float(lambdas[mono_mask].min())
            lam_max = float(lambdas[mono_mask].max())
            # Clearly highlight the Euclidean strongly-monotone region.
            span = ax.axvspan(
                lam_min,
                lam_max,
                facecolor="#f0f0f0",
                edgecolor="#555555",
                alpha=0.5,
                linewidth=1.2,
            )

        ax.set_xlabel(r"Coupling $\lambda$")
        ax.set_ylabel(r"Step size $h$")
        # Focus on the practically relevant region.
        ax.set_xlim(0.0, 1.75)
        ax.set_ylim(0.0, 3.0)
        ax.set_title(f"LQ {method.capitalize()} phase diagram")

        # Axes-level legend placed inside the plot in the top-right corner.
        handles, labels = ax.get_legend_handles_labels()
        if np.any(mono_mask):
            from matplotlib.patches import Patch

            handles.append(
                Patch(
                    facecolor="white",
                    edgecolor="gray",
                    alpha=0.14,
                    label="Euclidean monotone region",
                )
            )
            labels.append("Euclidean monotone region")

        ax.legend(
            handles,
            labels,
            loc="upper right",
            frameon=True,
            framealpha=0.9,
            fontsize=8,
        )

        fig.tight_layout()
        out_pdf = figs_dir / fname
        fig.savefig(out_pdf)
        fig.savefig(out_pdf.with_suffix(".png"), dpi=300)
        plt.close(fig)

    _plot_panel("euler")
    _plot_panel("rk4")

    # Margin curves
    alpha_true = metrics["alpha_true"]
    alpha_sgn = metrics["alpha_sgn"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lambdas, np.maximum(gamma_euc, 0.0), "k--", label=r"Euclidean $\gamma_{\min}$")
    ax.plot(lambdas, alpha_true, "b-", label=r"True $\alpha_{\mathrm{true}}$")
    ax.plot(lambdas, alpha_sgn, "g-.", label=r"SGN bound $\alpha_*$")
    ax.axhline(0.0, color="gray", linewidth=1.0)
    ax.set_xlabel(r"Coupling $\lambda$")
    ax.set_ylabel("Margin")
    ax.set_title("Monotonicity margins vs coupling")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_pdf = figs_dir / "lq_margins.pdf"
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_time_series_slices(
    metrics_path: pathlib.Path | str = "LQ_GAMES/data/metrics_balanced.npz",
    stability_path: pathlib.Path | str = "LQ_GAMES/data/stability.npz",
    game_path: pathlib.Path | str = "LQ_GAMES/data/game.npz",
    figs_dir: pathlib.Path | str = "figs/LQ_GAME",
    num_steps: int = 300,
) -> None:
    """
    Plot representative norm trajectories at a few (λ, h) pairs.
    """

    figs_dir = _ensure_dir(pathlib.Path(figs_dir))
    metrics = np.load(metrics_path)
    stability = np.load(stability_path)
    game = load_game(game_path)

    lambdas = metrics["lambdas"]
    alpha_sgn = metrics["alpha_sgn"]
    euler_step_sgn = metrics["euler_step_sgn"]
    rk4_step_sgn = metrics["rk4_step_sgn"]
    w = metrics["w"]
    a = float(metrics["a"])
    b = float(metrics["b"])

    euler_h_grid = stability["euler_h_grid"]
    rk4_h_grid = stability["rk4_h_grid"]
    rho_euler = stability["rho_euler"]
    rho_rk4 = stability["rho_rk4"]

    Q1 = game["Q1"]
    Q2 = game["Q2"]
    R = game["R"]
    d1, d2 = Q1.shape[0], Q2.shape[0]
    M = build_metric_matrix(w, d1=d1, d2=d2)

    rng = np.random.default_rng(2024)

    positive_indices = np.where(alpha_sgn > 0.0)[0]
    if positive_indices.size == 0:
        return
    chosen = np.unique(
        np.round(
            np.linspace(0, positive_indices.size - 1, min(3, positive_indices.size))
        ).astype(int)
    )
    lambda_indices = positive_indices[chosen]

    for method in ("euler", "rk4"):
        fig, axes = plt.subplots(
            1, len(lambda_indices), figsize=(5 * len(lambda_indices), 4), sharey=True
        )
        if len(lambda_indices) == 1:
            axes = [axes]

        for ax, idx in zip(axes, lambda_indices):
            lam = float(lambdas[idx])
            H = build_H(Q1, Q2, R, lam=lam, a=a, b=b)

            if method == "euler":
                h_sgn = euler_step_sgn[idx]
                h_grid = euler_h_grid
                rho_grid = rho_euler
                step_fn = euler_step_matrix
            else:
                h_sgn = rk4_step_sgn[idx]
                h_grid = rk4_h_grid
                rho_grid = rho_rk4
                step_fn = rk4_step_matrix

            rho_row = rho_grid[idx, :]
            stable_mask = rho_row < 1.0
            h_stab = h_grid[stable_mask].max() if np.any(stable_mask) else 0.0

            step_sizes = []
            if h_sgn > 0.0:
                step_sizes.append(0.5 * h_sgn)
                step_sizes.append(0.9 * h_sgn)
            if h_stab > h_sgn > 0.0:
                step_sizes.append(0.5 * (h_sgn + h_stab))

            step_sizes = sorted({h for h in step_sizes if h > 0.0})

            for h in step_sizes:
                T = step_fn(H, h)
                x = rng.normal(loc=0.0, scale=1.0, size=(d1 + d2,))
                norms = []
                for k in range(num_steps):
                    norms.append(float(np.sqrt(x.T @ (M @ x))))
                    x = T @ x
                ks = np.arange(num_steps)
                label = rf"$h = {h:.3g}$"
                ax.plot(ks, norms, label=label)

            ax.set_xlabel("Iteration $k$")
            ax.set_title(rf"$\lambda = {lam:.2g}$")
            ax.set_yscale("log")
            ax.grid(True, linestyle="--", alpha=0.3)

        axes[0].set_ylabel(r"$\|x_k\|_{M(w)}$")
        fig.suptitle(f"LQ {method.capitalize()} trajectories at representative step sizes")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = figs_dir / f"lq_timeseries_{method}.pdf"
        fig.savefig(out_path)
        fig.savefig(out_path.with_suffix(".png"), dpi=300)
        plt.close(fig)


def compute_noise_robustness(
    metrics_path: pathlib.Path | str = "LQ_GAMES/data/metrics_balanced.npz",
    game_path: pathlib.Path | str = "LQ_GAMES/data/game.npz",
    out_path: pathlib.Path | str = "LQ_GAMES/data/noise_robustness.npz",
) -> pathlib.Path:
    """
    Compute α_true/α_* under structured noise perturbations of both couplings.
    """

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = np.load(metrics_path)
    game = load_game(game_path)

    w = metrics["w"]
    mu1 = float(metrics["mu1"])
    mu2 = float(metrics["mu2"])
    a = float(metrics["a"])
    b = float(metrics["b"])

    Q1 = game["Q1"]
    Q2 = game["Q2"]
    R_struct = game["R"]
    d1, d2 = Q1.shape[0], Q2.shape[0]
    M = build_metric_matrix(w, d1=d1, d2=d2)

    eps_grid = np.array(NOISE_CONFIG.eps_grid, dtype=float)
    lambda_val = float(NOISE_CONFIG.lambda_value)
    rng = np.random.default_rng(GAME_CONFIG.seed + NOISE_CONFIG.seed_offset)

    alpha_true_mean = []
    alpha_sgn_mean = []
    ratio_mean = []

    for eps in eps_grid:
        alpha_true_trials = []
        alpha_sgn_trials = []

        for _ in range(NOISE_CONFIG.num_trials):
            noise12 = rng.standard_normal((d1, d2))
            noise21 = rng.standard_normal((d2, d1))

            u, s, vh = np.linalg.svd(noise12, full_matrices=False)
            noise12 = noise12 / max(s[0], 1e-8)
            u, s, vh = np.linalg.svd(noise21, full_matrices=False)
            noise21 = noise21 / max(s[0], 1e-8)

            A12 = (1.0 - eps) * (lambda_val * a * R_struct) + eps * (lambda_val * a * noise12)
            A21 = (1.0 - eps) * (lambda_val * b * R_struct.T) + eps * (lambda_val * b * noise21)
            H = np.block([[Q1, A12], [A21, Q2]])

            # True margin in metric M.
            alpha_true_trials.append(true_monotonicity_margin(H, M))

            # SGN bound uses spectral norms of the perturbed couplings.
            L12 = float(np.linalg.svd(A12, compute_uv=False)[0])
            L21 = float(np.linalg.svd(A21, compute_uv=False)[0])
            alpha_sgn_trials.append(sgn_margin(mu1, mu2, L12, L21, w=w))

        alpha_true_avg = float(np.mean(alpha_true_trials))
        alpha_sgn_avg = float(np.mean(alpha_sgn_trials))
        alpha_true_mean.append(alpha_true_avg)
        alpha_sgn_mean.append(alpha_sgn_avg)
        ratio_mean.append(alpha_true_avg / alpha_sgn_avg if alpha_sgn_avg > 0 else np.nan)

    np.savez_compressed(
        out_path,
        eps_grid=eps_grid,
        alpha_true=np.array(alpha_true_mean),
        alpha_sgn=np.array(alpha_sgn_mean),
        ratio=np.array(ratio_mean),
    )
    return out_path


def plot_noise_robustness(
    data_path: pathlib.Path | str = "LQ_GAMES/data/noise_robustness.npz",
    figs_dir: pathlib.Path | str = "figs/LQ_GAME",
) -> None:
    """Plot α_true/α_* vs ε for the structured noise experiment."""

    figs_dir = _ensure_dir(pathlib.Path(figs_dir))
    data = np.load(data_path)
    eps_grid = data["eps_grid"]
    ratio = data["ratio"]

    fig, ax = plt.subplots(figsize=(6, 4))
    line_curve, = ax.plot(
        eps_grid,
        ratio,
        "o-",
        color="#9467bd",
        label=r"mean $\alpha_{\mathrm{true}}/\alpha_*$",
    )
    line_ref = ax.axhline(
        1.0,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label=r"ideal ratio $=1$",
    )
    ax.set_xlabel(r"Noise fraction $\epsilon$")
    ax.set_ylabel(r"Conservatism ratio $\alpha_{\mathrm{true}}/\alpha_*$")
    ax.set_title("Structured coupling noise robustness")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=8)
    fig.tight_layout()
    out_pdf = figs_dir / "lq_noise_robustness.pdf"
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_random_suite_figures(
    suite_path: pathlib.Path | str = "LQ_GAMES/data/random_suite.npz",
    figs_dir: pathlib.Path | str = "figs/LQ_GAME",
) -> None:
    """
    Produce the three standard random-ensemble figures (margins, steps, certification rates).
    """

    figs_dir = _ensure_dir(pathlib.Path(figs_dir))
    data = np.load(suite_path)
    lambdas = data["lambdas"]
    alpha_true = data["alpha_true"]
    alpha_sgn = data["alpha_sgn"]
    h_sgn = data["h_sgn_rk4"]
    h_stab = data["h_stab_rk4"]
    gamma_euc = data["gamma_euc"]

    ratios = np.zeros_like(alpha_true)
    ratios[:] = np.nan
    mask = alpha_true > 0
    ratios[mask] = alpha_sgn[mask] / alpha_true[mask]

    # Reverted to larger size but with optimized density.
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    positions = np.arange(len(lambdas))
    for j, lam in enumerate(lambdas):
        vals = ratios[:, j]
        vals = vals[~np.isnan(vals)]
        jitter = (np.random.rand(vals.size) - 0.5) * 0.08
        ax.scatter(
            np.full(vals.size, lam) + jitter,
            vals,
            color="#1f77b4",
            alpha=0.6,
            s=30,
            edgecolors="k",
            linewidths=0.6,
        )
        if vals.size:
            med = np.median(vals)
            q1, q3 = np.percentile(vals, [25, 75])
            ax.plot([lam, lam], [q1, q3], color="#ff7f0e", linewidth=3.0)
            ax.scatter(lam, med, color="#ff7f0e", marker="_", s=100, linewidths=3.0)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_xticks(lambdas)
    ax.set_xticklabels([f"{lam:.2g}" for lam in lambdas])
    ax.set_xlabel(r"Coupling $\lambda$", fontsize=11)
    ax.set_ylabel(r"Ratio $\alpha_*/\alpha_{\mathrm{true}}$", fontsize=11)
    ax.set_title("Margin ratios", fontsize=12)
    ax.set_ylim(-0.25, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(labelsize=10)

    # Legend: distinguish individual games vs median/IQR summary.
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color="#1f77b4",
            markeredgecolor="k",
            markersize=5.0,
            label="games",
        ),
        Line2D(
            [],
            [],
            color="#ff7f0e",
            linewidth=3.0,
            label="median",
        ),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, framealpha=0.9, fontsize=11)

    fig.tight_layout(pad=0.3)
    out_pdf = figs_dir / "lq_random_margin_ratios.pdf"
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=300)
    plt.close(fig)

    ratios_log = np.zeros_like(h_sgn)
    ratios_log[:] = np.nan
    mask = (h_sgn > 0) & (h_stab > 0)
    ratios_log[mask] = np.log10(h_sgn[mask] / h_stab[mask])

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for j, lam in enumerate(lambdas):
        vals = ratios_log[:, j]
        vals = vals[~np.isnan(vals)]
        jitter = (np.random.rand(vals.size) - 0.5) * 0.08
        ax.scatter(
            np.full(vals.size, lam) + jitter,
            vals,
            color="#9467bd",
            alpha=0.6,
            s=30,
            edgecolors="k",
            linewidths=0.6,
        )
        if vals.size:
            med = np.median(vals)
            q1, q3 = np.percentile(vals, [25, 75])
            ax.plot([lam, lam], [q1, q3], color="#ff7f0e", linewidth=3.0)
            ax.scatter(lam, med, color="#ff7f0e", marker="_", s=100, linewidths=3.0)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xticks(lambdas)
    ax.set_xticklabels([f"{lam:.2g}" for lam in lambdas])
    ax.set_xlabel(r"Coupling $\lambda$", fontsize=11)
    ax.set_ylabel(r"$\log_{10}(h_{\mathrm{SGN}}/h_{\mathrm{stab}})$", fontsize=11)
    ax.set_title("RK4 step bounds", fontsize=12)
    all_vals = np.concatenate([vals for vals in [ratios_log[:, j][~np.isnan(ratios_log[:, j])] for j in range(len(lambdas))] if vals.size > 0])
    ymin = float(np.floor(all_vals.min() - 0.5)) if all_vals.size else -2.0
    ax.set_ylim(ymin, 0.2)
    ax.set_yticks(np.arange(int(ymin), 1))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(labelsize=10)

    # Legend: individual games vs median/IQR summary.
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color="#9467bd",
            markeredgecolor="k",
            markersize=5.0,
            label="games",
        ),
        Line2D(
            [],
            [],
            color="#ff7f0e",
            linewidth=3.0,
            label="median",
        ),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, framealpha=0.9, fontsize=11)

    fig.tight_layout(pad=0.3)
    out_pdf = figs_dir / "lq_random_step_ratios.pdf"
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=300)
    plt.close(fig)

    S, L = gamma_euc.shape
    frac_euc = (gamma_euc > 0.0).sum(axis=0) / S
    frac_sgn = (alpha_sgn > 0.0).sum(axis=0) / S

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(lambdas, frac_euc, "k--o", label="Euclidean", markersize=6)
    ax.plot(lambdas, frac_sgn, "g-o", label="SGN", markersize=6)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"Coupling $\lambda$", fontsize=11)
    ax.set_ylabel("Fraction of seeds", fontsize=11)
    ax.set_title("Certification rates", fontsize=12)
    ax.legend(loc="best", fontsize=11)
    ax.tick_params(labelsize=10)
    fig.tight_layout(pad=0.3)
    out_pdf = figs_dir / "lq_random_certification_rates.pdf"
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_all_figures(
    metrics_path: pathlib.Path | str = "LQ_GAMES/data/metrics_balanced.npz",
    stability_path: pathlib.Path | str = "LQ_GAMES/data/stability.npz",
    sim_summary_path: pathlib.Path | str = "LQ_GAMES/data/simulation_summary.npy",
    random_suite_path: pathlib.Path | str = "LQ_GAMES/data/random_suite.npz",
    game_path: pathlib.Path | str = "LQ_GAMES/data/game.npz",
    figs_dir: pathlib.Path | str = "figs/LQ_GAME",
) -> None:
    """Run every plotting routine and save outputs to figs/."""

    plot_timescale_band(metrics_path, figs_dir=figs_dir)
    plot_phase_diagrams_and_margins(
        metrics_path=metrics_path,
        stability_path=stability_path,
        sim_summary_path=sim_summary_path,
        figs_dir=figs_dir,
    )

    # Random ensemble figures (compute if missing).
    suite_path = pathlib.Path(random_suite_path)
    if not suite_path.exists():
        run_random_suite(out_path=suite_path)
    plot_random_suite_figures(suite_path, figs_dir=figs_dir)

    # Structured noise robustness.
    noise_path = pathlib.Path("LQ_GAMES/data/noise_robustness.npz")
    compute_noise_robustness(metrics_path=metrics_path, game_path=game_path, out_path=noise_path)
    plot_noise_robustness(data_path=noise_path, figs_dir=figs_dir)

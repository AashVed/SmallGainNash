"""
Plotting script for Markov Game Experiment.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def plot_results():
    """
    Recreate the Markov game figures:
      1) Lyapunov decay for NPG vs EPG.
      2) Stability region as a function of step-size multiplier.
      3) Timescale band: SGN margin α_*(r) vs empirical success.
    """

    path = Path("markov/results.npz")
    if not path.exists():
        print("No results found.")
        return
        
    data = np.load(path, allow_pickle=True)
    
    # Create figs dir
    figs_dir = Path("figs/markov")
    figs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Lyapunov Descent
    plt.figure(figsize=(6.5, 4.0))

    # Helper for median/IQR envelopes on log V
    def plot_shades(curves, label, color):
        min_len = min(len(c) for c in curves)
        arr = np.array([c[:min_len] for c in curves])
        
        log_arr = np.log(arr + 1e-16)
        
        med = np.median(log_arr, axis=0)
        q25 = np.percentile(log_arr, 25, axis=0)
        q75 = np.percentile(log_arr, 75, axis=0)
        
        x = np.arange(min_len)
        plt.plot(x, med, label=label, color=color)
        plt.fill_between(x, q25, q75, color=color, alpha=0.2)

    plot_shades(data["npg_curves_lyap"], "Natural PG (SGN metric)", "#1f77b4")
    plot_shades(data["epg_curves_lyap"], "Euclidean PG", "#d62728")

    plt.xlabel("Iteration $k$")
    plt.ylabel(r"$\log V_k$  (Bregman Lyapunov)")
    plt.title("Markov game: Lyapunov decay in mirror geometry")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    out = figs_dir / "lyapunov.pdf"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"), dpi=300)
    plt.close()
    
    # 2. Stability Sweep
    plt.figure(figsize=(6.5, 4.0))
    mults = data["sweep_mults"]

    plt.semilogx(
        mults,
        data["sweep_probs_npg"],
        "o-",
        color="#1f77b4",
        label="Natural PG (SGN metric)",
    )
    plt.semilogx(
        mults,
        data["sweep_probs_epg"],
        "s-",
        color="#d62728",
        label="Euclidean PG",
    )

    plt.axvline(1.0, color="k", linestyle="--", linewidth=1.2, label=r"SGN bound ($\eta=\eta_{\max}$)")

    plt.xlabel(r"Step-size multiplier $\eta / \eta_{\max}$")
    plt.ylabel("Convergence probability")
    plt.title("Markov game: stability vs step size")
    plt.legend(loc="best")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    out = figs_dir / "stability.pdf"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"), dpi=300)
    plt.close()
    
    # 3. Timescale Band
    fig, ax1 = plt.subplots(figsize=(6.5, 4.0))
    
    r_grid = data["band_r_grid"]
    alphas = data["band_alphas"]
    success = data["band_success"]

    # Theoretical SGN margin α_*(r)
    ax1.set_xlabel(r"Weight ratio $r = w_2 / w_1$")
    ax1.set_ylabel(r"SGN margin $\alpha_*(r)$", color="#2ca02c")
    ax1.set_xscale("log")
    ax1.semilogx(
        r_grid,
        alphas,
        color="#2ca02c",
        linewidth=2.0,
        linestyle="--",
        label=r"SGN margin $\alpha_*(r)$",
    )
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)

    # Highlight the SGN feasibility band where α_*(r) ≥ 0
    if np.any(alphas >= 0.0):
        mask = alphas >= 0.0
        r_pos = r_grid[mask]
        r_min = float(r_pos.min())
        r_max = float(r_pos.max())
        ax1.axvspan(
            r_pos.min(),
            r_pos.max(),
            facecolor="#FFDDC1",
            alpha=0.4,
            label="SGN feasibility band (α≥0)",
        )

    plt.title("Markov game: SGN Certification Band (Theory)")
    plt.legend(loc="best")
    plt.tight_layout()
    out = figs_dir / "timescale.pdf"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"), dpi=300)
    plt.close()
    
    print("Plots generated in figs/markov/")

if __name__ == "__main__":
    plot_results()
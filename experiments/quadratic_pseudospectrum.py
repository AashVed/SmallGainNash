"""
Visualizes the pseudospectra of the linearized quadratic game (Section 5.1).

This script computes the resolvent norm ||(J - lambda I)^-1|| in the complex plane
for two geometries:
1. The standard Euclidean metric (where non-normality is large).
2. The SGN metric M(w) (where the operator is closer to normal/dissipative).

It produces Figure 2 of the paper. The output is saved to `figs/mini-toy/quadratic_pseudospectrum.pdf`.
"""
import os

import matplotlib.pyplot as plt
import numpy as np


# Paper-style defaults.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "text.usetex": False,
        "axes.unicode_minus": False,
        "figure.dpi": 300,
        "axes.unicode_minus": False,
    }
)


def compute_smin_grid(J, xlim, ylim, resolution=321):
    """
    Compute the smallest singular value of J - λI on a grid in the complex plane.
    """
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    smin = np.empty_like(X, dtype=float)

    eye = np.eye(J.shape[0], dtype=complex)
    Jc = J.astype(complex)
    for i in range(Y.shape[0]):
        for j in range(X.shape[1]):
            lam = X[i, j] + 1j * Y[i, j]
            A = Jc - lam * eye
            sv = np.linalg.svd(A, compute_uv=False)
            smin[i, j] = sv[-1]
    return X, Y, smin


def main():
    # Quadratic example parameters from the paper.
    mu1 = 1.0
    mu2 = 1.0
    a = 10.0
    b = 0.05

    J = np.array([[mu1, a],
                  [b,   mu2]], dtype=float)
    eigvals = np.linalg.eigvals(J)

    # SGN-balanced metric weights: w1/w2 = L21/L12 = 0.05/10 = 1/200.
    w1 = 1.0
    w2 = 200.0
    sqrt_w = np.sqrt(np.array([w1, w2]))
    M_sqrt = np.diag(sqrt_w)
    M_inv_sqrt = np.diag(1.0 / sqrt_w)
    # Similarity transform corresponding to the M(w)-geometry.
    J_sgn = M_sqrt @ J @ M_inv_sqrt

    # Common window around the eigenvalues.
    re_min, re_max = eigvals.real.min(), eigvals.real.max()
    pad_re = 1.5
    pad_im = 1.2
    xlim = (re_min - pad_re, re_max + pad_re)
    ylim = (-pad_im, pad_im)

    X, Y, smin_eucl = compute_smin_grid(J, xlim=xlim, ylim=ylim, resolution=321)
    _, _, smin_sgn = compute_smin_grid(J_sgn, xlim=xlim, ylim=ylim, resolution=321)

    # Use log10 of the resolvent norm for a smooth color scale.
    eps = 1e-6
    log_res_eucl = np.log10(1.0 / np.clip(smin_eucl, eps, None))
    log_res_sgn = np.log10(1.0 / np.clip(smin_sgn, eps, None))
    vmin = min(np.nanmin(log_res_eucl), np.nanmin(log_res_sgn))
    vmax = max(np.nanmax(log_res_eucl), np.nanmax(log_res_sgn))

    # Quantitative summary: how much of the region has large resolvent norm.
    thresholds = [0.5, 1.0, 1.5]
    frac_summary = {}
    for thr in thresholds:
        frac_e = float((log_res_eucl >= thr).mean())
        frac_s = float((log_res_sgn >= thr).mean())
        frac_summary[thr] = (frac_e, frac_s)
        print(
            f"Fraction of grid with log10||resolvent|| >= {thr:.1f}: "
            f"Euclidean={frac_e:.3f}, M(w)={frac_s:.3f}"
        )

    # Use constrained_layout to keep the colorbar from overlapping axes.
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.3), sharex=True, sharey=True, constrained_layout=True)

    levels = np.linspace(vmin, vmax, 25)

    # Panel (a): Euclidean pseudospectra of J.
    cf0 = axes[0].contourf(
        X,
        Y,
        log_res_eucl,
        levels=levels,
        cmap="viridis",
    )
    axes[0].contour(
        X,
        Y,
        log_res_eucl,
        levels=levels[::4],
        colors="k",
        linewidths=0.4,
        alpha=0.8,
    )
    axes[0].scatter(eigvals.real, eigvals.imag, marker="x", color="red", s=16, linewidths=1.0)
    axes[0].set_title(r"(a) Euclidean geometry")

    # Panel (b): pseudospectra in the SGN metric via the similarity J_sgn.
    cf1 = axes[1].contourf(
        X,
        Y,
        log_res_sgn,
        levels=levels,
        cmap="viridis",
    )
    axes[1].contour(
        X,
        Y,
        log_res_sgn,
        levels=levels[::4],
        colors="k",
        linewidths=0.4,
        alpha=0.8,
    )
    axes[1].scatter(eigvals.real, eigvals.imag, marker="x", color="red", s=16, linewidths=1.0)
    axes[1].set_title(r"(b) $M(w)$ geometry, $w_1/w_2=1/200$")

    for ax in axes:
        ax.set_xlabel(r"$\Re(\lambda)$")
        ax.axhline(0.0, color="0.6", linewidth=0.5, alpha=0.7)
        ax.axvline(0.0, color="0.6", linewidth=0.5, alpha=0.7)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=7, width=0.6, length=2.6)

    axes[0].set_ylabel(r"$\Im(\lambda)$")

    # Shared colorbar, placed to the right of both panels.
    cbar = fig.colorbar(cf0, ax=axes, shrink=0.9, pad=0.05)
    cbar.set_label(r"$\log_{10}\|(J-\lambda I)^{-1}\|_2$")
    cbar.ax.tick_params(labelsize=7, width=0.6, length=2.6)

    # Save into figs/mini-toy with a stable relative path.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    figs_dir = os.path.join(base_dir, "..", "figs", "mini-toy")
    os.makedirs(figs_dir, exist_ok=True)
    pdf_path = os.path.join(figs_dir, "quadratic_pseudospectrum.pdf")
    png_path = os.path.join(figs_dir, "quadratic_pseudospectrum.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    print(f"Saved pseudospectrum figure to {pdf_path} and {png_path}")


if __name__ == "__main__":
    main()

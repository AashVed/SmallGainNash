"""
Compute Euclidean and SGN metrics for the 2-player LQ game.

Given Q_1, Q_2 and the normalized base coupling matrix R, this script:

  - builds the joint pseudo-gradient matrix H(λ) for each λ,
  - computes Euclidean monotonicity margins γ_min^Euc(λ),
  - constructs SGN matrices C(w, 0) and SGN margins α_*(w, λ),
  - constructs the SGN metric M(w) and the true monotonicity margin
    α_true(w, λ) in that metric,
  - computes Lipschitz constants β(w, λ) in ‖·‖_{M(w)},
  - computes SGN-based Euler and RK4 step bounds for each λ.

The outputs are stored in a compressed .npz file and reused by other scripts.
"""

from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass

import numpy as np
from numpy.linalg import eigvalsh

from .config import GAME_CONFIG, LAMBDA_GRID, WEIGHT_SEARCH


@dataclass
class WeightConfig:
    """
    Description of the SGN weight choice.

    Attributes
    ----------
    mode : str
        Either 'ones' for w = (1,1) or 'balanced' for w_1/w_2 = b/a.
    """

    mode: str = "balanced"

    def weights(self, a: float, b: float) -> np.ndarray:
        """
        Return the weight vector w for the given coupling scales a, b.
        """

        if self.mode == "ones":
            return np.array([1.0, 1.0])
        if self.mode == "balanced":
            # w1 / w2 = b / a ⇒ choose w2 = 1, w1 = b/a.
            return np.array([b / a, 1.0])
        raise ValueError(f"Unknown weight mode '{self.mode}'")


def load_game(path: pathlib.Path | str = "LQ_GAMES/data/game.npz") -> Dict[str, np.ndarray]:
    """
    Load Q_1, Q_2, R and basic parameters from disk.
    """

    data = np.load(path)
    return data


def build_H(Q1: np.ndarray, Q2: np.ndarray, R: np.ndarray, lam: float, a: float, b: float) -> np.ndarray:
    """
    Construct the joint pseudo-gradient matrix H(λ) for the 2-player game.

    H(λ) has block structure

      H(λ) = [[Q1, A12(λ)],
              [A21(λ), Q2]],

    with A12(λ) = λ a R and A21(λ) = λ b R^T.
    """

    d1, d2 = Q1.shape[0], Q2.shape[0]
    H = np.zeros((d1 + d2, d1 + d2))

    H[:d1, :d1] = Q1
    H[d1:, d1:] = Q2
    H[:d1, d1:] = lam * a * R
    H[d1:, :d1] = lam * b * R.T
    return H


def euclidean_monotonicity_margin(H: np.ndarray) -> float:
    """
    Compute the Euclidean monotonicity margin γ_min^Euc(H).
    """

    Hs = 0.5 * (H + H.T)
    eigs = eigvalsh(Hs)
    return float(eigs[0])


def build_metric_matrix(w: np.ndarray, d1: int, d2: int) -> np.ndarray:
    """
    Construct M(w) = diag(w_1 I_{d1}, w_2 I_{d2}).
    """

    w1, w2 = w.tolist()
    return np.diag(np.concatenate([w1 * np.ones(d1), w2 * np.ones(d2)]))


def sgn_margin(
    mu1: float,
    mu2: float,
    L12: float,
    L21: float,
    w: np.ndarray,
) -> float:
    """
    Compute just the SGN margin α_*(w) for the 2-player case.
    """

    w1, w2 = w.tolist()
    C = np.array(
        [
            [2.0 * w1 * mu1, -(w1 * L12 + w2 * L21)],
            [-(w1 * L12 + w2 * L21), 2.0 * w2 * mu2],
        ]
    )
    D_inv_sqrt = np.diag(1.0 / np.sqrt(w))
    C_tilde = D_inv_sqrt @ C @ D_inv_sqrt
    eigs = eigvalsh(C_tilde)
    return 0.5 * float(eigs[0])


def true_monotonicity_margin(H: np.ndarray, M: np.ndarray) -> float:
    """
    Compute the true strong monotonicity margin α_true in the metric M.

    α_true is the smallest eigenvalue of M^{-1/2} J_M M^{-1/2}, where
    J_M = (M H + H^T M) / 2.
    """

    # M is diagonal with positive entries; work with its diagonal to avoid
    # numerical issues when inverting zeros off the diagonal.
    diag_M = np.diag(M)
    diag_sqrt = np.sqrt(diag_M)
    M_sqrt = np.diag(diag_sqrt)
    M_inv_sqrt = np.diag(1.0 / diag_sqrt)
    # J_M = 0.5 * (M H + H^T M)
    J_M = 0.5 * (M @ H + H.T @ M)
    A = M_inv_sqrt @ J_M @ M_inv_sqrt
    eigs = eigvalsh(A)
    return float(eigs[0])


def lipschitz_constant(H: np.ndarray, M: np.ndarray) -> float:
    """
    Compute the Lipschitz constant β of G(x) = -F(x) = -H x in the metric M.

    In the M-norm, the operator norm of H is the spectral norm of

        H_M = M^{1/2} H M^{-1/2}.
    """

    diag_M = np.diag(M)
    diag_sqrt = np.sqrt(diag_M)
    M_sqrt = np.diag(diag_sqrt)
    M_inv_sqrt = np.diag(1.0 / diag_sqrt)
    H_M = M_sqrt @ H @ M_inv_sqrt
    # Largest singular value of H_M.
    svals = np.linalg.svd(H_M, compute_uv=False)
    return float(svals[0])


def ratio_grid(cfg=WEIGHT_SEARCH) -> np.ndarray:
    """Log-spaced grid of weight ratios r = w2 / w1."""

    return np.geomspace(cfg.ratio_min, cfg.ratio_max, cfg.num_points)


def two_player_band(mu1: float, mu2: float, L12: float, L21: float, alpha: float = 0.0) -> tuple[float, float]:
    """
    Closed-form SGN feasibility band (r_-, r_+) for N = 2 and a target margin α.
    """

    det_term = (mu1 - alpha) * (mu2 - alpha) - L12 * L21
    if det_term <= 0 or L21 == 0:
        return (math.nan, math.nan)
    base = 2 * (mu1 - alpha) * (mu2 - alpha) - L12 * L21
    radical = 2 * math.sqrt((mu1 - alpha) * (mu2 - alpha) * det_term)
    denom = L21 ** 2
    r_minus = (base - radical) / denom
    r_plus = (base + radical) / denom
    return (r_minus, r_plus)


def _metrics_for_weight(
    w: np.ndarray,
    lambdas: np.ndarray,
    Q1: np.ndarray,
    Q2: np.ndarray,
    R: np.ndarray,
    a: float,
    b: float,
    mu1: float,
    mu2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Helper computing alpha_sgn, alpha_true, beta, euler/rk4 bounds for a fixed weight."""

    d1, d2 = Q1.shape[0], Q2.shape[0]
    alpha_sgn = []
    alpha_true = []
    beta_vals = []
    euler_step = []
    rk4_step = []

    M = build_metric_matrix(w, d1=d1, d2=d2)

    for lam in lambdas:
        H = build_H(Q1, Q2, R, lam=lam, a=a, b=b)
        L12 = lam * a
        L21 = lam * b

        alpha_star = sgn_margin(mu1, mu2, L12, L21, w=w)
        alpha_sgn.append(alpha_star)

        alpha_t = true_monotonicity_margin(H, M)
        alpha_true.append(alpha_t)

        beta = lipschitz_constant(H, M)
        beta_vals.append(beta)

        if alpha_star > 0.0 and beta > 0.0:
            euler_step.append(2.0 * alpha_star / (beta ** 2))
            rk4_step.append(WEIGHT_SEARCH.rk4_c4 / beta)
        else:
            euler_step.append(0.0)
            rk4_step.append(0.0)

    return (
        np.asarray(alpha_sgn),
        np.asarray(alpha_true),
        np.asarray(beta_vals),
        np.asarray(euler_step),
        np.asarray(rk4_step),
    )


def compute_metrics(
    w_cfg: WeightConfig = WeightConfig(mode="balanced"),
    game_path: pathlib.Path | str = "LQ_GAMES/data/game.npz",
    out_dir: pathlib.Path | str = "LQ_GAMES/data",
) -> pathlib.Path:
    """
    Compute Euclidean and SGN metrics, including a weight-ratio sweep.

    This function evaluates α_*, α_true, β, and SGN step bounds over:
      - the predefined λ grid, for the nominal weight choice (w_cfg),
      - a log-spaced grid of weight ratios r = w2 / w1 used to optimize
        α_*(r)/β(r)^2 (Euler) and α_*(r)/β(r) (RK4).
    The results, along with the recommended ratios and the closed-form
    two-player SGN band, are saved to an .npz file.
    """

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_game(game_path)
    Q1 = data["Q1"]
    Q2 = data["Q2"]
    R = data["R"]
    a = float(data["a"])
    b = float(data["b"])

    d1, d2 = Q1.shape[0], Q2.shape[0]
    lambdas = LAMBDA_GRID.values()

    mu1 = float(eigvalsh(Q1)[0])
    mu2 = float(eigvalsh(Q2)[0])

    # Euclidean margins do not depend on the metric.
    gamma_euc = np.zeros_like(lambdas)
    for i, lam in enumerate(lambdas):
        H = build_H(Q1, Q2, R, lam=lam, a=a, b=b)
        gamma_euc[i] = euclidean_monotonicity_margin(H)

    ratios = ratio_grid()
    lam_ts = WEIGHT_SEARCH.lambda_timescale
    idx_ts = int(np.argmin(np.abs(lambdas - lam_ts)))
    alpha_sgn_grid = np.zeros((ratios.size, lambdas.size))
    alpha_true_grid = np.zeros_like(alpha_sgn_grid)
    beta_grid = np.zeros_like(alpha_sgn_grid)

    # Sweep over ratios.
    for j, r in enumerate(ratios):
        w_r = np.array([1.0, r])
        M_r = build_metric_matrix(w_r, d1=d1, d2=d2)
        for i, lam in enumerate(lambdas):
            H = build_H(Q1, Q2, R, lam=lam, a=a, b=b)
            L12 = lam * a
            L21 = lam * b
            alpha_sgn_grid[j, i] = sgn_margin(mu1, mu2, L12, L21, w=w_r)
            alpha_true_grid[j, i] = true_monotonicity_margin(H, M_r)
            beta_grid[j, i] = lipschitz_constant(H, M_r)

    # Objectives for picking a single r.
    eps = 1e-12
    obj_euler = np.where(alpha_sgn_grid > 0, alpha_sgn_grid / (beta_grid ** 2 + eps), np.nan)
    obj_rk4 = np.where(alpha_sgn_grid > 0, alpha_sgn_grid / (beta_grid + eps), np.nan)
    focus_euler = obj_euler[:, idx_ts]
    focus_rk4 = obj_rk4[:, idx_ts]

    if np.all(np.isnan(focus_euler)):
        agg_euler = np.nanmedian(obj_euler, axis=1)
    else:
        agg_euler = focus_euler
    if np.all(np.isnan(focus_rk4)):
        agg_rk4 = np.nanmedian(obj_rk4, axis=1)
    else:
        agg_rk4 = focus_rk4

    if np.all(np.isnan(agg_euler)):
        best_idx_euler = 0
    else:
        best_idx_euler = int(np.nanargmax(agg_euler))
    if np.all(np.isnan(agg_rk4)):
        best_idx_rk4 = 0
    else:
        best_idx_rk4 = int(np.nanargmax(agg_rk4))

    best_ratio_euler = float(ratios[best_idx_euler])
    best_ratio_rk4 = float(ratios[best_idx_rk4])

    # Nominal weight (balanced by default) used for the main figures.
    w_nominal = w_cfg.weights(a=a, b=b)
    (
        alpha_sgn_nom,
        alpha_true_nom,
        beta_nom,
        euler_step_nom,
        rk4_step_nom,
    ) = _metrics_for_weight(w_nominal, lambdas, Q1, Q2, R, a, b, mu1, mu2)

    # Two-player SGN band for λ = lambda_timescale (default 1.0).
    L12_ts = float(lambdas[idx_ts] * a)
    L21_ts = float(lambdas[idx_ts] * b)
    r_minus, r_plus = two_player_band(mu1, mu2, L12_ts, L21_ts, alpha=0.0)

    out_path = out_dir / "metrics_balanced.npz"
    np.savez_compressed(
        out_path,
        lambdas=lambdas,
        mu1=mu1,
        mu2=mu2,
        a=a,
        b=b,
        w=w_nominal,
        gamma_euc=gamma_euc,
        alpha_sgn=alpha_sgn_nom,
        alpha_true=alpha_true_nom,
        beta=beta_nom,
        euler_step_sgn=euler_step_nom,
        rk4_step_sgn=rk4_step_nom,
        ratio_grid=ratios,
        alpha_sgn_grid=alpha_sgn_grid,
        alpha_true_grid=alpha_true_grid,
        beta_grid=beta_grid,
        objective_euler=obj_euler,
        objective_rk4=obj_rk4,
        best_ratio_euler=best_ratio_euler,
        best_ratio_rk4=best_ratio_rk4,
        timescale_lambda=float(lambdas[idx_ts]),
        timescale_index=idx_ts,
        timescale_r_minus=r_minus,
        timescale_r_plus=r_plus,
    )
    return out_path


def main() -> None:
    """Entry point computing metrics and saving them under LQ_GAMES/data/."""

    path = compute_metrics()
    print(f"Saved SGN and Euclidean metrics to {path}")


if __name__ == "__main__":
    main()

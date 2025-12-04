"""
Generate a high-dimensional 2-player linear–quadratic (LQ) game.

The main "clean" game used in the paper has the structure described in
plan_LQ.md:

  - Two players (i = 1, 2) with block dimensions d1 and d2 (d1 = d2).
  - Own-player curvature matrices

        Q_1 = mu0 * I_{d1},   Q_2 = mu0 * I_{d2},

    so the curvature parameters are exactly μ_1 = μ_2 = μ_0.
  - A shared cross-player coupling matrix R = I (or any orthonormal block),
    and asymmetric cross-player couplings

        A_12(λ) = λ a R,   A_21(λ) = λ b R^T,

    where a ≫ b > 0 mirror the scalar example in the paper.

This script constructs Q_1, Q_2 and R once (for a fixed configuration) and
stores them in an .npz file for reuse by the metric and simulation scripts.
"""

from __future__ import annotations

import pathlib
from typing import Tuple

import numpy as np
from numpy.linalg import norm

from .config import GAME_CONFIG


def generate_game_matrices(
    cfg=GAME_CONFIG,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Q_1, Q_2 and the base coupling matrix R for the clean LQ game.

    Parameters
    ----------
    cfg : GameConfig
        Game configuration defining dimensions, base curvature and couplings.

    Returns
    -------
    Q1 : ndarray of shape (d1, d1)
        Own-player Hessian for player 1.
    Q2 : ndarray of shape (d2, d2)
        Own-player Hessian for player 2.
    R : ndarray of shape (d1, d2)
        Normalized cross-player base matrix with spectral norm 1.
    """

    # Clean isotropic curvature: Q_i = mu0 * I
    Q1 = cfg.mu0 * np.eye(cfg.d1)
    Q2 = cfg.mu0 * np.eye(cfg.d2)

    # Clean orthonormal coupling: draw a random orthogonal block with ‖R‖_2 = 1.
    if cfg.d1 != cfg.d2:
        raise ValueError("Clean LQ generator requires d1 == d2")
    rng = np.random.default_rng(cfg.seed)
    gaussian_block = rng.standard_normal((cfg.d1, cfg.d2))
    Q_raw, upper = np.linalg.qr(gaussian_block)
    # Ensure a deterministic orientation by flipping the sign of columns whose
    # R diagonal entry is negative.
    signs = np.sign(np.diag(upper))
    signs[signs == 0] = 1.0
    R = Q_raw * signs

    return Q1, Q2, R


def save_game(
    Q1: np.ndarray,
    Q2: np.ndarray,
    R: np.ndarray,
    cfg=GAME_CONFIG,
    out_dir: pathlib.Path | str = "LQ_GAMES/data",
) -> pathlib.Path:
    """
    Save the generated game matrices and configuration as a compressed .npz file.

    Parameters
    ----------
    Q1, Q2, R : ndarray
        Game matrices as returned by :func:`generate_game_matrices`.
    cfg : GameConfig
        Game configuration used to generate the matrices.
    out_dir : path-like
        Directory in which to store the game file.

    Returns
    -------
    path : pathlib.Path
        Path to the saved .npz file.
    """

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "game.npz"

    np.savez_compressed(
        path,
        Q1=Q1,
        Q2=Q2,
        R=R,
        d_total=cfg.d_total,
        d1=cfg.d1,
        d2=cfg.d2,
        mu0=cfg.mu0,
        a=cfg.a,
        b=cfg.b,
        seed=cfg.seed,
    )

    return path


def main() -> None:
    """
    Command-line entry point.

    Generates a single 2-player LQ game instance using the default configuration
    and stores it under `LQ_GAMES/data/game.npz`.
    """

    Q1, Q2, R = generate_game_matrices()
    path = save_game(Q1, Q2, R)
    print(f"Saved LQ game to {path}")


if __name__ == "__main__":
    main()

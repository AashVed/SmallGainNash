"""
Configuration for the LQ_GAMES validation experiment.

This module centralizes all numeric hyperparameters used to construct the
linear–quadratic game, define the SGN metric, and sweep over coupling and
step-size grids.  The defaults are chosen to:

  - produce a clearly nontrivial dimension (d = 64),
  - mirror the asymmetric couplings from the scalar example in the paper
    (a = 10, b = 0.05),
  - give a reasonably fine grid in the coupling parameter λ, and
  - keep all computations cheap enough to run on a laptop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class GameConfig:
    """
    Parameters defining the high-dimensional 2-player LQ game.

    Attributes
    ----------
    d_total : int
        Total dimension d of the joint strategy space.
    mu0 : float
        Base curvature parameter for each player's Hessian.  In the canonical
        flagship game we take Q_i = mu0 * I so that the block curvature
        parameters are exactly μ_i = mu0.  In the random ensemble we instead
        use Q_i = R_i^T R_i + mu0 * I to generate heterogeneous but
        well-conditioned curvature.
    a : float
        Scaling factor for the (1 -> 2) coupling A_12(λ) = λ a R.
    b : float
        Scaling factor for the (2 -> 1) coupling A_21(λ) = λ b R^T.
    seed : int
        Global random seed for reproducibility of Q_i and R.
    """

    d_total: int = 64
    # Base curvature for each player's quadratic cost.  We choose mu0 = 1.0
    # to mirror the canonical scalar example in the paper (mu_1 = mu_2 = 1),
    # while keeping cross-player couplings (a = 10, b = 0.05) strong enough
    # to induce Euclidean non-monotonicity for moderate λ.
    mu0: float = 1.0
    a: float = 10.0
    b: float = 0.05
    seed: int = 12345

    @property
    def d1(self) -> int:
        """Dimension of player 1's block."""

        return self.d_total // 2

    @property
    def d2(self) -> int:
        """Dimension of player 2's block."""

        return self.d_total - self.d1


@dataclass(frozen=True)
class LambdaGrid:
    """
    Grid of coupling parameters λ.

    Attributes
    ----------
    lambda_min : float
        Smallest λ in the grid (typically 0).
    lambda_max : float
        Largest λ in the grid.  Should be large enough to make both the
        Euclidean and SGN conditions fail for some λ.
    num_points : int
        Number of grid points between lambda_min and lambda_max (inclusive).
    """

    lambda_min: float = 0.0
    lambda_max: float = 2.5
    num_points: int = 61

    def values(self):
        """Return the λ grid as a NumPy array."""

        import numpy as np

        return np.linspace(self.lambda_min, self.lambda_max, self.num_points)


@dataclass(frozen=True)
class StepSizeGrid:
    """
    Step-size multipliers used relative to the SGN step bound.

    The actual step sizes are defined per-(λ, method) as

      h = c * h_SGN(λ)

    for c in step_multipliers.  We use the same multipliers for Euler and
    RK4; the underlying SGN bound differs by method.
    """

    # Multipliers relative to the SGN step bound.
    # Values < 1 are strictly inside the certified region,
    # values > 1 probe beyond the SGN guarantee.
    step_multipliers: Tuple[float, ...] = (
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
        4.0,
    )


@dataclass(frozen=True)
class SimulationConfig:
    """
    Configuration for discrete-time simulations used to visualize trajectories.

    Attributes
    ----------
    num_seeds : int
        Number of random initial conditions per (λ, h, method) combination.
    sigma0 : float
        Standard deviation of the Gaussian used to sample initial conditions.
    max_steps : int
        Maximum number of steps per trajectory.
    """

    num_seeds: int = 16
    sigma0: float = 1.0
    max_steps: int = 800


@dataclass(frozen=True)
class WeightSearchConfig:
    """
    Grid and constants used to optimize the SGN weights.

    Attributes
    ----------
    ratio_min : float
        Minimum ratio r = w2 / w1 explored in the search.
    ratio_max : float
        Maximum ratio r = w2 / w1 explored in the search.
    num_points : int
        Number of ratios sampled on a log scale between ratio_min and ratio_max.
    lambda_timescale : float
        Coupling value used for the timescale-band visualization.
    rk4_c4 : float
        Constant C_4 used in the RK4 stability guideline h <= C_4 / beta.
    """

    ratio_min: float = 1e-1
    ratio_max: float = 1e4
    num_points: int = 161
    lambda_timescale: float = 1.0
    rk4_c4: float = 2.5


@dataclass(frozen=True)
class NoiseSuiteConfig:
    """
    Configuration for the structured noise robustness sweep.

    Attributes
    ----------
    lambda_value : float
        Coupling used for the perturbation experiment.
    eps_grid : tuple of float
        Fractions ε in [0, 1] mixing the structured coupling with random noise.
    num_trials : int
        Number of independent noise draws per ε for averaging.
    seed_offset : int
        Offset added to cfg.seed for reproducible perturbations.
    """

    lambda_value: float = 1.0
    eps_grid: tuple[float, ...] = tuple(np.linspace(0.0, 1.0, 11))
    num_trials: int = 12
    seed_offset: int = 777


@dataclass(frozen=True)
class ContinuousTimeConfig:
    """
    Configuration for continuous-time flow simulations.

    Attributes
    ----------
    dt : float
        Time step for the RK4 integrator used to approximate the flow.
    horizon : float
        Total integration time.
    num_seeds : int
        Number of random initial conditions per λ.
    """

    dt: float = 1e-3
    horizon: float = 5.0
    num_seeds: int = 8


# Default global configs used by the scripts in this package.

GAME_CONFIG = GameConfig()
LAMBDA_GRID = LambdaGrid()
STEP_GRID = StepSizeGrid()
SIM_CONFIG = SimulationConfig()
FLOW_CONFIG = ContinuousTimeConfig()
WEIGHT_SEARCH = WeightSearchConfig()
NOISE_CONFIG = NoiseSuiteConfig()

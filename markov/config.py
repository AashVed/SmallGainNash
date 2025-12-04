"""
Configuration for the Markov Game SGN Experiment.
"""
from dataclasses import dataclass

@dataclass
class MarkovConfig:
    # Game Dimensions
    n_players: int = 2
    n_states: int = 2
    n_actions: int = 2  # Binary actions {0, 1}
    
    # Dynamics
    # Discount factor for the Markov game. We choose a relatively
    # large value so that long-horizon effects and the environment
    # gain are visible, matching the Markov extension in the paper.
    gamma: float = 0.9
    
    # Regularization
    # Controls curvature: higher tau -> more curvature (mu) -> larger SGN margin (alpha).
    # The default tau=1.0 is chosen so that SGN certifies a nontrivial local margin.
    tau: float = 1.0

    # Reward scaling
    # Multiplicative factor on the base reward tensor. This allows us to
    # stress-test the geometry by increasing cross-player couplings while
    # holding the environment structure fixed.
    reward_scale: float = 1.0
    
    # SGN Certification
    w_init: tuple = (1.0, 1.0)
    
    # Experiment Settings
    seed: int = 42
    n_seeds: int = 20  # For stability stats
    max_steps: int = 2000
    convergence_tol: float = 1e-6
    
    # Timescale Band Sweep
    r_min: float = 0.01
    r_max: float = 100.0
    n_r_grid: int = 20

    # Local SGN certification region around theta*
    # We certify curvature and Lipschitz bounds on a small logit cube
    #   { theta : ||theta - theta*||_inf <= grid_radius }
    # using a tensor-product grid with grid_points_per_dim points per
    # coordinate (typically 2, i.e., the 2^d vertices at ±grid_radius).
    grid_radius: float = 0.1
    grid_points_per_dim: int = 2

    # Output
    results_path: str = "markov/results.npz"

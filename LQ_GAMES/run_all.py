"""
End-to-end driver for the LQ SGN validation experiment.

Running this module as a script will:

  1. Generate a 2-player high-dimensional LQ game and save it under
     `LQ_GAMES/data/game.npz`.
  2. Compute Euclidean and SGN metrics for the balanced weight choice and
     save them under `LQ_GAMES/data/metrics_balanced.npz`.
  3. Compute discrete-time stability grids and thresholds for Euler and RK4
     and save them under `LQ_GAMES/data/stability.npz`.
  4. Run discrete-time simulations for a subset of (λ, h) pairs and save a
     summary of empirical contraction factors under
     `LQ_GAMES/data/simulation_summary.npy`.
  5. Simulate continuous-time flows at a few λ values and save median-norm
     summaries and flow figures.
  6. Run the random LQ robustness suite.
  7. Generate all LQ figures (timescale band, phase diagrams,
     random ensemble, noise robustness) and save them into
     `figs/LQ_GAME/`, ready to be included in the paper.
"""

from __future__ import annotations

from .generate_game import main as generate_game_main
from .compute_sgn_metrics import main as compute_metrics_main
from .compute_stability import main as compute_stability_main
from .simulate_discrete import main as simulate_main
from .simulate_flow import main as simulate_flow_main
from .random_suite import run_random_suite
from .figures import plot_all_figures


def main() -> None:
    """Run the full LQ SGN validation pipeline."""

    print("=== Step 1: Generating LQ game instance ===")
    generate_game_main()

    print("=== Step 2: Computing SGN and Euclidean metrics ===")
    compute_metrics_main()

    print("=== Step 3: Computing discrete-time stability grids ===")
    compute_stability_main()

    print("=== Step 4: Simulating selected discrete-time trajectories ===")
    simulate_main()

    print("=== Step 5: Simulating continuous-time flows ===")
    simulate_flow_main()

    print("=== Step 6: Running random LQ robustness suite ===")
    run_random_suite()

    print("=== Step 7: Plotting all figures (timescale band, phase diagrams, random, noise) ===")
    plot_all_figures()

    print("LQ SGN validation pipeline completed.")


if __name__ == "__main__":
    main()

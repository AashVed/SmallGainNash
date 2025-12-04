# LQ SGN Validation Experiments

This directory contains the code for the high-dimensional Linear–Quadratic (LQ)
game experiments presented in the paper
**“Small-Gain Nash: Certified Contraction to Nash Equilibria in Differentiable Games”**.

It validates the Small-Gain Nash (SGN) theory by:

- generating a canonical 64-dimensional 2-player LQ game,
- computing Euclidean and SGN strong monotonicity margins,
- verifying discrete-time stability bounds for projected Euler and RK4, and
- exploring robustness via structured-noise perturbations and a random LQ ensemble.

## Directory Structure

- `run_all.py`: main entry point; runs the full pipeline (generation, metrics, simulation, plotting).
- `config.py`: centralized configuration for dimensions, coupling grids, and simulation parameters.
- `generate_game.py`: generates the canonical 64-dimensional LQ game instance.
- `compute_sgn_metrics.py`: computes Euclidean margins, SGN margins, Lipschitz constants, and SGN step-size bounds.
- `compute_stability.py`: computes true stability thresholds for Euler and RK4 via spectral radius analysis.
- `simulate_discrete.py`: runs discrete-time trajectories (Euler/RK4) to verify contraction empirically.
- `simulate_flow.py`: simulates continuous-time gradient flows in the SGN metric.
- `random_suite.py`: generates and analyzes an ensemble of random LQ games to test robustness.
- `figures.py`: generates all plots used in the paper (saved to `../figs/LQ_GAME/`).
- `data/`: intermediate metrics and simulation results (created automatically; safe to delete and regenerate).

## Installation

In the recommended monorepo setup, dependencies are installed once at the root
(`SmallGainNash/`) using:

```bash
cd SmallGainNash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r LQ_GAMES/requirements.txt -r markov/requirements.txt
```

If you use this directory standalone, you can instead run:

```bash
pip install -r requirements.txt
```

## Usage

To run the full LQ experiment pipeline and generate all figures (from the
repository root):

```bash
.venv/bin/python -m LQ_GAMES.run_all
```

This will:

1. generate the game data in `LQ_GAMES/data/`,
2. compute all theoretical metrics (Euclidean and SGN margins, Lipschitz constants),
3. simulate discrete-time Euler/RK4 trajectories and continuous-time flows,
4. run the random robustness suite, and
5. produce the following figures in `figs/LQ_GAME/`:
   - `lq_margins.pdf`: comparison of Euclidean vs. SGN margins;
   - `lq_timescale_band.pdf`: certified timescale band for the weight ratio \(w_2/w_1\);
   - `lq_phase_euler.pdf` / `lq_phase_rk4.pdf`: stability phase diagrams in the \((\lambda, h)\)-plane;
   - `lq_flow_timeseries.pdf`: norm decay of continuous flows in the SGN metric;
   - `lq_noise_robustness.pdf`: robustness to structured coupling noise;
   - `lq_random_*.pdf`: summary statistics for the random LQ ensemble (margin ratios, step-size ratios, certification rates).

## Reproducibility and Runtimes

- All randomization is controlled by fixed seeds defined in `config.py`, so reruns
  are deterministic up to linear-algebra library details.
- The `data/` directory is purely intermediate; deleting it and rerunning
  `run_all.py` will regenerate all metrics and figures.
- On a laptop CPU, `run_all.py` typically completes within a minute.

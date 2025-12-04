# Markov Game SGN Experiments

This directory contains the code for the tabular Markov game experiments presented
in the paper **“Small-Gain Nash: Certified Contraction to Nash Equilibria in
Differentiable Games”**.

It demonstrates the application of the mirror-SGN framework to entropy-regularized
Natural Policy Gradient (NPG) dynamics. The experiments:

- certify a local contraction metric (Fisher information) around the Nash equilibrium,
- derive a safe step-size band for NPG in this metric, and
- compare NPG to Euclidean Policy Gradient (EPG) in terms of Lyapunov decay
  and empirical stability.

## Directory Structure

- `run.sh`: main entry point script to run the experiment and plotting.
- `experiment.py`: core logic for the Markov game, NPG/EPG dynamics, and data collection.
- `sgn_theory.py`: implements the grid-based SGN certification pipeline
  (curvature \(\mu\), couplings \(L\), Lipschitz constant \(\beta\)) in the Fisher geometry.
- `env.py`: defines the 2-player tabular Markov game environment.
- `config.py`: configuration for the game, entropy regularization \(\tau\),
  and grid search parameters.
- `plots.py`: generates the Markov figures from the saved results.
- `results.npz`: experimental results produced by `experiment.py`
  (step-size sweep, timescale band, Lyapunov curves).

## Installation

In the monorepo, dependencies are installed once at the root (`SmallGainNash/`):

```bash
cd SmallGainNash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r LQ_GAMES/requirements.txt -r markov/requirements.txt
```

If you only use this directory, you can instead run:

```bash
pip install -r requirements.txt
```

## Usage

From the repository root, run:

```bash
bash markov/run.sh
```

This script will:

1. execute `experiment.py` to
   - find the Nash equilibrium of the regularized game,
   - run the SGN certification pipeline to find the best metric weights \(w\)
     and a safe step-size \(\eta_{\max} = 2\alpha / \beta^2\),
   - simulate NPG and EPG trajectories at a certified step size,
   - sweep over step-size multipliers and timescale ratios, and
   - save all numerical results to `markov/results.npz`;
2. execute `plots.py` to generate the figures in `figs/markov/`:
   - `lyapunov.pdf`: median Lyapunov function decay for NPG vs. EPG;
   - `stability.pdf`: empirical convergence probability vs. step-size multiplier \(\eta / \eta_{\max}\);
   - `timescale.pdf`: the certified SGN timescale band \(r = w_2 / w_1\).

On a laptop CPU, the full run typically completes in a few minutes.

## SGN Certification Pipeline

The `sgn_theory.py` module implements the offline certification procedure described
in the paper:

1. construct a small tensor-product grid in logit space around the Nash equilibrium,
2. at each grid point, compute Hessian and Jacobian blocks in the Fisher metric,
3. aggregate these into conservative bounds \(\mu_i^{\text{lo}}\), \(L_{ij}^{\text{hi}}\),
4. optimize the weight ratio \(r = w_2 / w_1\) to maximize the SGN margin \(\alpha_*(r)\), and
5. compute a Lipschitz constant \(\beta\) for the natural-gradient field and
   output the certified step-size bound \(\eta < 2\alpha_*(r) / \beta^2\).

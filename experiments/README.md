# Mini-Toy Quadratic Game Experiments

This directory contains the code for the 2D quadratic “mini-toy” game used in
**“Small-Gain Nash: Certified Contraction to Nash Equilibria in Differentiable Games”**.
The scripts reproduce:

- the “escape vs. trap” phase portrait for the SGN metric, and
- the Euclidean vs. SGN pseudospectra of the linearized dynamics.

These experiments illustrate how the SGN block metric \(M(w)\) recovers strong
monotonicity and tightens the pseudospectrum for a game that is non-monotone in
Euclidean geometry.

## Directory Structure

- `quadratic_phase.py`  
  Generates the “escape vs. trap” phase portrait. It simulates a single trajectory
  of the gradient flow that escapes a Euclidean ball but remains trapped inside an
  SGN metric ellipsoid, and saves:
  - `figs/mini-toy/quadratic_phase.pdf`
  - `figs/mini-toy/quadratic_phase.png`

- `quadratic_pseudospectrum.py`  
  Computes the resolvent norm \(\|(J-\lambda I)^{-1}\|_2\) in the complex plane
  for:
  - the standard Euclidean geometry, and
  - the SGN-weighted geometry \(M(w)\),
  and saves:
  - `figs/mini-toy/quadratic_pseudospectrum.pdf`
  - `figs/mini-toy/quadratic_pseudospectrum.png`

## Dependencies

These scripts use only NumPy and Matplotlib. If you followed the root-level
instructions and installed

```bash
.venv/bin/pip install -r LQ_GAMES/requirements.txt -r markov/requirements.txt
```

then no additional installation is needed.

## Usage

From the repository root (`SmallGainNash/`):

```bash
.venv/bin/python experiments/quadratic_phase.py
.venv/bin/python experiments/quadratic_pseudospectrum.py
```

The figures will be written to `figs/mini-toy/`. They correspond to the 2D
quadratic example figures in the paper (phase portrait and pseudospectra).

## Parameters

Both scripts use the canonical scalar parameters from the paper:

- curvature: \(\mu_1 = \mu_2 = 1.0\)
- cross-player couplings: \(a = 10.0\), \(b = 0.05\)
- SGN weights: \(w_1 = 1.0\), \(w_2 = 200.0\) (balancing the off-diagonal terms).

You can modify these constants directly in the scripts to explore other
non-normal quadratic games; the defaults are chosen to match the paper.

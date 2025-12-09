# Small-Gain Nash: Experimental Code

[![arXiv](https://img.shields.io/badge/arXiv-2512.06791-b31b1b.svg)](https://arxiv.org/abs/2512.06791)

This directory is the public code companion to the paper
**“Small-Gain Nash: Certified Contraction to Nash Equilibria in Differentiable Games”**.
It contains all scripts used to generate the figures and numerical results.

The canonical GitHub repository is:

- `https://github.com/AashVed/SmallGainNash`

The repository is organized as a small monorepo with three experiment modules:

- `experiments/`: 2D quadratic “mini-toy” game (escape vs. trap and pseudospectra).
- `LQ_GAMES/`: 64-dimensional linear–quadratic (LQ) game experiments.
- `markov/`: tabular Markov game and mirror/Fisher SGN certification for NPG.

All three modules are designed to be self-contained and reproducible on a CPU-only machine.

## 1. Environment and Dependencies

The experiments require Python 3.8+ and a recent NumPy / SciPy / PyTorch stack.
We recommend creating a single shared virtual environment at the repository root:

```bash
cd SmallGainNash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r LQ_GAMES/requirements.txt -r markov/requirements.txt
```

The `experiments/` scripts only depend on NumPy and Matplotlib, which are already
included in the requirements above.

## 2. Reproducing All Figures

After the environment is set up, the full experimental pipeline (excluding the LaTeX build)
can be reproduced with three commands, always run from this directory:

1. **High-dimensional LQ games (all LQ figures)**

   ```bash
   .venv/bin/python -m LQ_GAMES.run_all
   ```

   This:
   - generates the canonical LQ game (`LQ_GAMES/data/`),
   - computes Euclidean and SGN monotonicity margins and Lipschitz constants,
   - computes Euler/RK4 stability regions,
   - runs discrete-time and continuous-time simulations,
   - runs the random LQ robustness suite, and
   - writes all LQ figures to `figs/LQ_GAME/`
     (margins, timescale band, phase diagrams, flow timeseries,
      noise robustness, random ensemble).

2. **Mini-toy quadratic game (2D example)**

   ```bash
   .venv/bin/python experiments/quadratic_phase.py
   .venv/bin/python experiments/quadratic_pseudospectrum.py
   ```

   This produces the 2D phase portraits and pseudospectra in `figs/mini-toy/`,
   matching the quadratic example in the main text.

3. **Tabular Markov game (mirror/Fisher SGN and NPG vs. EPG)**

   ```bash
   bash markov/run.sh
   ```

   This:
   - finds the regularized Nash equilibrium of the Markov game,
   - runs the grid-based SGN certification in the Fisher geometry,
   - simulates Natural Policy Gradient (NPG) and Euclidean Policy Gradient (EPG),
   - sweeps step sizes and timescale ratios, and
   - writes Markov figures to `figs/markov/`
     (Lyapunov decay, stability vs. step size, SGN timescale band).

Typical runtimes on a laptop CPU are:
- mini-toy: a few seconds,
- LQ pipeline: under a minute,
- Markov experiment: a few minutes.

## 3. Module Overview

- `experiments/`  
  2D quadratic game illustrating how the SGN metric traps trajectories that
  escape Euclidean balls and tightens the resolvent pseudospectrum.
  See `experiments/README.md` for precise figure mappings and commands.

- `LQ_GAMES/`  
  High-dimensional LQ validation of SGN, including Euclidean vs. SGN margins,
  certified vs. true stability regions for Euler/RK4, structured-noise robustness,
  and a random LQ ensemble. See `LQ_GAMES/README.md`.

- `markov/`  
  Tabular Markov game and mirror-SGN certification for entropy-regularized NPG,
  including Lyapunov decay and empirical stability bands. See `markov/README.md`.

## 4. Citation

If you use this code or the SGN framework in academic work, please cite the associated paper:

```bibtex
@article{vedansh2025sgn,
  title={Small-Gain Nash: Certified Contraction to Nash Equilibria in Differentiable Games},
  author={Sharma, Vedansh},
  journal={arXiv preprint arXiv:2512.06791},
  year={2025}
}
```

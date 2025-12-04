"""
LQ_GAMES
========

Tools for validating the Small–Gain Nash (SGN) theory on a high-dimensional
linear–quadratic (LQ) game. The main components are:

  - generation of a 2-player high-dimensional quadratic game with asymmetric
    cross-player couplings,
  - exact computation of Euclidean and SGN monotonicity margins,
  - exact Lipschitz constants and discrete-time stability thresholds for
    Euler and RK4,
  - simulation of discrete-time dynamics in the SGN metric, and
  - plotting phase diagrams and margin curves for inclusion in the paper.

See plan_LQ.md in the repository root for a prose description of the
experimental design implemented by these modules.
"""


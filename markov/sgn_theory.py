"""
SGN Certification Pipeline for Markov Games.
"""
import torch
import numpy as np
from markov.env import TabularMarkovGame
from markov.config import MarkovConfig

class SGNCertifier:
    def __init__(self, game: TabularMarkovGame):
        self.game = game
        self.cfg = game.cfg

    def _non_zero_min(self, M: torch.Tensor) -> float:
        """
        Minimum nonzero eigenvalue (in magnitude) of a symmetric matrix.
        Used to ignore the trivial nullspace of softmax Fisher blocks.
        """
        e = torch.linalg.eigvalsh(M)
        e_nz = e[torch.abs(e) > 1e-5]
        if e_nz.numel() == 0:
            return 0.0
        return float(e_nz.min().item())

    def _get_sqrt_and_inv_sqrt(self, M: torch.Tensor):
        """
        Eigen-based square root and inverse square root of an SPD matrix,
        with small eigenvalues clamped for numerical stability.
        """
        e, v = torch.linalg.eigh(M)
        e_clamped = torch.clamp(e, min=1e-6)
        sqrt = v @ torch.diag(torch.sqrt(e_clamped)) @ v.T
        inv_sqrt = v @ torch.diag(1.0 / torch.sqrt(e_clamped)) @ v.T
        return sqrt, inv_sqrt

    def _local_geometry(self, theta: torch.Tensor):
        """
        Compute local curvature, couplings, and Jacobian / Fisher blocks
        at a given parameter vector theta.

        Returns a dict with:
          mu1, mu2   : Fisher-normalized own-player curvatures
          L12, L21   : Fisher-normalized cross-player couplings
          J_mat      : 8x8 Jacobian of the pseudo-gradient F
          F1_sqrt    : Fisher^1/2 for player 1
          F2_sqrt    : Fisher^1/2 for player 2
          F1_inv_sqrt, F2_inv_sqrt : corresponding inverse square roots
        """
        n_p = 4
        theta = theta.detach().clone().requires_grad_(True)

        def get_hessian(loss_idx: int) -> torch.Tensor:
            def func(t):
                l1, l2 = self.game.get_loss(t)
                return l1 if loss_idx == 0 else l2
            return torch.autograd.functional.hessian(func, theta)

        # Hessian blocks of each player's loss
        H1_full = get_hessian(0)
        H2_full = get_hessian(1)

        H11 = H1_full[:n_p, :n_p]
        H12 = H1_full[:n_p, n_p:]
        H21 = H2_full[n_p:, :n_p]
        H22 = H2_full[n_p:, n_p:]

        # Fisher blocks
        _, F1, F2 = self.game.get_fisher(theta)
        F1_sqrt, F1_inv_sqrt = self._get_sqrt_and_inv_sqrt(F1)
        F2_sqrt, F2_inv_sqrt = self._get_sqrt_and_inv_sqrt(F2)

        # Fisher-normalized own-player curvature
        M1 = F1_inv_sqrt @ H11 @ F1_inv_sqrt
        M2 = F2_inv_sqrt @ H22 @ F2_inv_sqrt
        mu1 = self._non_zero_min(M1)
        mu2 = self._non_zero_min(M2)

        # Fisher-normalized cross-player couplings
        K12 = F1_inv_sqrt @ H12 @ F2_inv_sqrt
        K21 = F2_inv_sqrt @ H21 @ F1_inv_sqrt
        L12 = float(torch.linalg.norm(K12, ord=2).item())
        L21 = float(torch.linalg.norm(K21, ord=2).item())

        # Pseudo-gradient Jacobian J_F at theta
        J_mat = torch.zeros((2 * n_p, 2 * n_p), dtype=self.game.dtype)
        J_mat[:n_p, :n_p] = H11
        J_mat[:n_p, n_p:] = H12
        J_mat[n_p:, :n_p] = H21
        J_mat[n_p:, n_p:] = H22

        return {
            "mu1": mu1,
            "mu2": mu2,
            "L12": L12,
            "L21": L21,
            "J_mat": J_mat,
            "F1_sqrt": F1_sqrt,
            "F2_sqrt": F2_sqrt,
            "F1_inv_sqrt": F1_inv_sqrt,
            "F2_inv_sqrt": F2_inv_sqrt,
        }
        
    def find_equilibrium(self, theta_init=None):
        """
        Find Nash Equilibrium via gradient descent on the norm of the gradient.
        """
        if theta_init is None:
            theta = torch.zeros(8, dtype=self.game.dtype, requires_grad=True)
        else:
            theta = theta_init.clone().detach().requires_grad_(True)
            
        optimizer = torch.optim.Adam([theta], lr=0.01)
        
        for i in range(5000):
            optimizer.zero_grad()
            
            # Compute vector field F = [grad_1 f_1, grad_2 f_2]
            l1, l2 = self.game.get_loss(theta)
            
            # Get gradients
            g1 = torch.autograd.grad(l1, theta, create_graph=True)[0]
            g2 = torch.autograd.grad(l2, theta, create_graph=True)[0]
            
            # Masking: theta is flat [theta1, theta2]
            # g1 should only act on theta1 part, g2 on theta2 part
            # The autograd.grad returns full vector deriv, but f1 only depends on theta?
            # No, f1 depends on theta1 AND theta2 (through occupancy).
            # So grad_theta1 f1 is the first half of g1.
            
            n_p = 4 # params per player
            
            # Pseudo-gradient F
            v1 = g1[:n_p]
            v2 = g2[n_p:]
            
            F_norm = torch.sum(v1**2) + torch.sum(v2**2)
            
            if F_norm < 1e-10:
                break
                
            F_norm.backward()
            optimizer.step()
            
        return theta.detach()

    def compute_local_sgn(self, theta_star):
        """
        Compute curvature (mu), couplings (L), margin (alpha), and Lipschitz (beta)
        on a small neighbourhood around theta_star.

        NOTE ON COORDINATE INVARIANCE:
        We work in logit coordinates theta, but all SGN quantities are computed in the
        Fisher metric (Riemannian geometry), which is equivalent to the negative-entropy
        mirror metric on the simplex.

        Definitions at each theta in the neighbourhood:
          mu_i(theta) = min_eig( F_i(theta)^-1/2 H_ii(theta) F_i(theta)^-1/2 )
          L_ij(theta) = max_sing( F_i(theta)^-1/2 H_ij(theta) F_j(theta)^-1/2 )
          beta(theta; w) = max_sing( M(w)^{1/2} J_F(theta) M(w)^{-1/2} ), M(w)=diag(w1 F1, w2 F2).

        We then form conservative bounds
          mu_i^lo  = min_theta mu_i(theta),
          L_ij^hi  = max_theta L_ij(theta),
          beta^hi(w) = max_theta beta(theta; w)
        over a tensor-product grid inside { theta : ||theta - theta*||_inf <= grid_radius }.
        """
        theta_star = theta_star.detach()
        dim = theta_star.numel()

        radius = getattr(self.cfg, "grid_radius", 0.1)
        n_pts_dim = getattr(self.cfg, "grid_points_per_dim", 2)

        # Tensor-product grid in a small cube around theta_star.
        # We always include the center theta_star implicitly.
        if n_pts_dim <= 1:
            levels = torch.zeros(1, dtype=self.game.dtype)
        else:
            levels = torch.linspace(-radius, radius, n_pts_dim, dtype=self.game.dtype)

        grid_axes = [levels for _ in range(dim)]
        offsets = torch.cartesian_prod(*grid_axes)

        geom_points = []
        mu1_vals = []
        mu2_vals = []
        L12_vals = []
        L21_vals = []

        # Evaluate local geometry on each grid point
        for off in offsets:
            theta = theta_star + off
            geom = self._local_geometry(theta)
            geom_points.append(geom)
            mu1_vals.append(geom["mu1"])
            mu2_vals.append(geom["mu2"])
            L12_vals.append(geom["L12"])
            L21_vals.append(geom["L21"])

        # Aggregate conservative block bounds on the neighbourhood
        mu1_lo = float(min(mu1_vals))
        mu2_lo = float(min(mu2_vals))
        L12_hi = float(max(L12_vals))
        L21_hi = float(max(L21_vals))

        print(
            f"DEBUG (grid SGN): mu1_lo={mu1_lo:.4f}, mu2_lo={mu2_lo:.4f}, "
            f"L12_hi={L12_hi:.4f}, L21_hi={L21_hi:.4f}"
        )

        # Optimize over weight ratios using the aggregated bounds.
        best_alpha = -1e9
        best_r = 1.0
        r_grid = np.logspace(-2, 2, 200)

        for r in r_grid:
            w1 = 1.0
            w2 = r
            k12 = L12_hi * np.sqrt(w1 / w2)
            k21 = L21_hi * np.sqrt(w2 / w1)
            off = -0.5 * (k12 + k21)
            H_mx = np.array([[mu1_lo, off], [off, mu2_lo]])
            min_eig = np.linalg.eigvalsh(H_mx).min()
            if min_eig > best_alpha:
                best_alpha = min_eig
                best_r = r

        # Lipschitz constant beta for the *natural-gradient* field on the same
        # neighbourhood for the chosen weights w.
        #
        # In the mirror/NPG setting, the algorithm we actually run is a
        # forward-Euler step on the natural-gradient ODE
        #
        #   dot(theta) = - F(theta)^{-1} grad f(theta),
        #
        # where F is the Fisher information matrix in logit coordinates.
        # The SGN margin alpha has been computed in the Fisher geometry with
        # metric M(w) = diag(w1 F1, w2 F2), so the Lipschitz constant used in
        # the CFL-style bound must correspond to this *same* vector field and
        # metric.  Concretely, for each grid point we:
        #
        #   1. reconstruct the Fisher blocks F1, F2 from their square roots;
        #   2. build the block-diagonal Fisher matrix F = diag(F1, F2);
        #   3. form the Jacobian of the natural-gradient field
        #        J_nat = F^{-1} J_F,
        #      where J_F is the pseudo-gradient Jacobian assembled above; and
        #   4. compute its operator norm in the SGN/Fisher metric
        #        M(w) = diag(w1 F1, w2 F2),
        #      via the similarity transform
        #        M(w)^{1/2} J_nat M(w)^{-1/2}.
        #
        # This replaces the earlier proxy based directly on J_F, and aligns the
        # beta used in the Markov experiment with the natural policy gradient
        # dynamics analysed in the text.
        w1_opt = 1.0
        w2_opt = best_r
        beta_vals = []

        for geom in geom_points:
            J_mat = geom["J_mat"]
            F1_sqrt = geom["F1_sqrt"]
            F2_sqrt = geom["F2_sqrt"]
            F1_inv_sqrt = geom["F1_inv_sqrt"]
            F2_inv_sqrt = geom["F2_inv_sqrt"]

            # Reconstruct Fisher blocks and the joint Fisher matrix.
            F1 = F1_sqrt @ F1_sqrt.T
            F2 = F2_sqrt @ F2_sqrt.T
            F_block = torch.block_diag(F1, F2)

            # Natural-gradient Jacobian: J_nat = F^{-1} J_F.
            # Use a linear solve for numerical stability instead of forming F^{-1}.
            J_nat = torch.linalg.solve(F_block, J_mat)

            # Metric square roots for M(w) = diag(w1 F1, w2 F2).
            M_sqrt_blk1 = (w1_opt ** 0.5) * F1_sqrt
            M_sqrt_blk2 = (w2_opt ** 0.5) * F2_sqrt
            M_sqrt = torch.block_diag(M_sqrt_blk1, M_sqrt_blk2)

            M_inv_sqrt_blk1 = (w1_opt ** -0.5) * F1_inv_sqrt
            M_inv_sqrt_blk2 = (w2_opt ** -0.5) * F2_inv_sqrt
            M_inv_sqrt = torch.block_diag(M_inv_sqrt_blk1, M_inv_sqrt_blk2)

            J_transformed = M_sqrt @ J_nat @ M_inv_sqrt
            beta_theta = torch.linalg.norm(J_transformed, ord=2).item()
            beta_vals.append(beta_theta)

        beta_hi = float(max(beta_vals))

        print(
            f"DEBUG (grid SGN): best_alpha={best_alpha:.4f}, "
            f"best_r={best_r:.4f}, beta_hi={beta_hi:.4f}"
        )

        return {
            "mu": (mu1_lo, mu2_lo),
            "L": (L12_hi, L21_hi),
            "alpha": best_alpha,
            "beta": beta_hi,
            "best_r": best_r,
            "theta_star": theta_star,
        }
    
    def get_timescale_curve(self, params, r_grid):
        """
        Compute alpha_*(r) for a grid of ratios.
        """
        mu1, mu2 = params['mu']
        L12, L21 = params['L']
        alphas = []
        
        for r in r_grid:
            w1, w2 = 1.0, r
            k12 = L12 * np.sqrt(w1/w2)
            k21 = L21 * np.sqrt(w2/w1)
            off = -0.5 * (k12 + k21)
            H_mx = np.array([[mu1, off], [off, mu2]])
            alphas.append(np.linalg.eigvalsh(H_mx).min())
            
        return np.array(alphas)

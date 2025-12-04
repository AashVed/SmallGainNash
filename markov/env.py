"""
Environment and Game Logic for the Tabular Markov Game.
"""
import torch
import numpy as np
from markov.config import MarkovConfig

class TabularMarkovGame:
    def __init__(self, cfg: MarkovConfig):
        self.cfg = cfg
        self.device = torch.device("cpu")
        self.dtype = torch.float64
        
        # Transition Probabilities P(s' | s, a1, a2)
        # Shape: (S, A1, A2, S')
        self.P = torch.zeros(
            (cfg.n_states, cfg.n_actions, cfg.n_actions, cfg.n_states),
            dtype=self.dtype, device=self.device
        )
        self._setup_transitions()
        
        # Rewards r(s, a1, a2) - Cooperative, same for both
        # Shape: (S, A1, A2)
        self.R = torch.zeros(
            (cfg.n_states, cfg.n_actions, cfg.n_actions),
            dtype=self.dtype, device=self.device
        )
        self._setup_rewards()
        
        # Initial state distribution
        self.mu0 = torch.zeros(cfg.n_states, dtype=self.dtype, device=self.device)
        self.mu0[0] = 1.0  # Start at s0
        
    def _setup_transitions(self):
        # s0 transitions
        # 00 -> s0 (0.9), s1 (0.1)
        self.P[0, 0, 0, 0] = 0.9; self.P[0, 0, 0, 1] = 0.1
        # 11 -> s1 (0.9), s0 (0.1)
        self.P[0, 1, 1, 1] = 0.9; self.P[0, 1, 1, 0] = 0.1
        # Mismatch -> 0.5, 0.5
        self.P[0, 0, 1, 0] = 0.5; self.P[0, 0, 1, 1] = 0.5
        self.P[0, 1, 0, 0] = 0.5; self.P[0, 1, 0, 1] = 0.5
        
        # s1 transitions (flipped roles of 00 and 11)
        # 11 -> s1 (0.9), s0 (0.1)
        self.P[1, 1, 1, 1] = 0.9; self.P[1, 1, 1, 0] = 0.1
        # 00 -> s0 (0.9), s1 (0.1)
        self.P[1, 0, 0, 0] = 0.9; self.P[1, 0, 0, 1] = 0.1
        # Mismatch -> 0.5, 0.5
        self.P[1, 0, 1, 0] = 0.5; self.P[1, 0, 1, 1] = 0.5
        self.P[1, 1, 0, 0] = 0.5; self.P[1, 1, 0, 1] = 0.5
        
    def _setup_rewards(self):
        # Cooperative rewards (before scaling).
        # s0: Coord (+1), Mismatch (-1)
        self.R[0, 0, 0] = 1.0; self.R[0, 1, 1] = 1.0
        self.R[0, 0, 1] = -1.0; self.R[0, 1, 0] = -1.0
        
        # s1: Same structure
        self.R[1, 0, 0] = 1.0; self.R[1, 1, 1] = 1.0
        self.R[1, 0, 1] = -1.0; self.R[1, 1, 0] = -1.0

        # Apply global reward scaling to control the strength of the couplings
        # relative to the mirror curvature tau.
        if getattr(self.cfg, "reward_scale", 1.0) != 1.0:
            self.R *= self.cfg.reward_scale

    def get_joint_policy(self, theta):
        """
        Convert logits theta to joint policy probability tensor.
        theta: (2 * S * A,) flat vector containing [theta1, theta2]
        Returns: pi(s, a1, a2) shape (S, A1, A2)
        """
        # Split parameters
        n_params = self.cfg.n_states * self.cfg.n_actions
        theta1 = theta[:n_params].view(self.cfg.n_states, self.cfg.n_actions)
        theta2 = theta[n_params:].view(self.cfg.n_states, self.cfg.n_actions)
        
        # Softmax per state
        pi1 = torch.softmax(theta1, dim=1)
        pi2 = torch.softmax(theta2, dim=1)
        
        # Outer product for joint policy: pi(s, a1, a2) = pi1(s, a1) * pi2(s, a2)
        # Einsum: s i, s j -> s i j
        pi_joint = torch.einsum('si,sj->sij', pi1, pi2)
        return pi_joint, pi1, pi2

    def get_occupancy(self, pi_joint):
        """
        Compute discounted state-action occupancy d(s, a1, a2).
        d_pi = (1-gamma) * (I - gamma * P_pi)^-1 * mu0
        """
        # 1. Compute State Transition Matrix P_pi(s, s')
        # P_pi[s, s'] = sum_{a1, a2} pi(s, a1, a2) * P(s' | s, a1, a2)
        P_pi = torch.einsum('sij,sijk->sk', pi_joint, self.P)
        
        # 2. Compute State Occupancy d_s(s)
        # (I - gamma * P_pi^T) * d_s = (1-gamma) * mu0
        # Note: Usually formulated as d = (1-gamma)(I - gamma P)^-1 mu0
        # Here we solve linear system: (I - gamma * P_pi^T) x = (1-gamma) * mu0
        # Wait, standard formula d^pi(s) = (1-gamma) sum_t gamma^t P(s_t=s)
        # Vector d (state occ): d^T = (1-gamma) mu0^T (I - gamma P_pi)^-1
        # or d = (1-gamma) (I - gamma P_pi^T)^-1 mu0
        
        eye = torch.eye(self.cfg.n_states, dtype=self.dtype, device=self.device)
        mat = eye - self.cfg.gamma * P_pi.T
        rhs = (1 - self.cfg.gamma) * self.mu0
        d_s = torch.linalg.solve(mat, rhs)
        
        # 3. Compute State-Action Occupancy d(s, a1, a2) = d_s(s) * pi(s, a1, a2)
        d_sa = torch.einsum('s,sij->sij', d_s, pi_joint)
        return d_sa

    def get_loss(self, theta):
        """
        Compute the scalar loss for each player: f_i(theta) = -J_i(pi(theta)).
        Returns tuple (loss1, loss2).
        """
        pi_joint, pi1, pi2 = self.get_joint_policy(theta)
        d_sa = self.get_occupancy(pi_joint)
        
        # Expected Return
        expected_return = torch.sum(d_sa * self.R)
        
        # Entropy Regularization
        # H(pi_i(.|s)) = - sum_a pi(a|s) log pi(a|s)
        # Regularizer = tau * sum_s d_s(s) * H(pi(.|s)) 
        # NOTE: The standard formulation in NPG papers (e.g. Cen et al) 
        # weights entropy by state occupancy.
        # The plan says: tau * sum_s H(pi(.|s)) (unweighted sum over states? or expected?)
        # Plan: "sum_{s \in S} H(\pi_i(\cdot|s))" -> This implies unweighted sum.
        # Let's stick to the plan's formula: tau * sum_s H(pi(.|s))
        # This ensures strong convexity in logits even if occupancy vanishes.
        
        eps = 1e-12
        H1 = -torch.sum(pi1 * torch.log(pi1 + eps), dim=1) # (S,)
        H2 = -torch.sum(pi2 * torch.log(pi2 + eps), dim=1) # (S,)
        
        reg1 = self.cfg.tau * torch.sum(H1)
        reg2 = self.cfg.tau * torch.sum(H2)
        
        # Objectives
        J1 = expected_return + reg1
        J2 = expected_return + reg2
        
        return -J1, -J2

    def get_fisher(self, theta):
        """
        Compute block-diagonal Fisher Information Matrix for logits.
        For softmax policy pi(a|s) = softmax(theta(s)), F_s = diag(pi) - pi pi^T.
        We assume unweighted Fisher (standard for NPG updates on parameters).
        Or should it be weighted by occupancy?
        Standard NPG usually weights by d_s(s).
        HOWEVER, if we use the unweighted regularization objective (tau * sum H),
        the natural geometry is the unweighted Fisher sum over states.
        Let's use unweighted Fisher to match the regularization term's geometry exactly.
        
        Returns: Block diag matrix (N_params, N_params)
        """
        _, pi1, pi2 = self.get_joint_policy(theta)
        
        # Build Fisher blocks for Player 1
        # F1: Block diag of (diag(p) - pp^T) for each state
        blocks1 = []
        for s in range(self.cfg.n_states):
            p = pi1[s]
            blk = torch.diag(p) - torch.outer(p, p)
            blocks1.append(blk)
        F1 = torch.block_diag(*blocks1)
        
        # Build Fisher blocks for Player 2
        blocks2 = []
        for s in range(self.cfg.n_states):
            p = pi2[s]
            blk = torch.diag(p) - torch.outer(p, p)
            blocks2.append(blk)
        F2 = torch.block_diag(*blocks2)
        
        # Total Fisher is block diagonal of F1, F2
        F = torch.block_diag(F1, F2)
        
        # Add epsilon for numerical stability inversion
        F += 1e-3 * torch.eye(F.shape[0], dtype=self.dtype, device=self.device)
        
        return F, F1, F2

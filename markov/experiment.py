"""
Experiment Runner for Markov Games.
"""
import torch
import numpy as np
import os
from tqdm import tqdm
from markov.config import MarkovConfig
from markov.env import TabularMarkovGame
from markov.sgn_theory import SGNCertifier

def run_npg(game, theta_init, theta_star, eta, weights, n_steps):
    """
    Weighted Natural Policy Gradient.
    
    Geometric Interpretation:
    In logit coordinates, this update:
      theta_i <- theta_i - eta * F_i^-1 * g_i
    is exactly a projected Euler step for the gradient flow in the Fisher metric.
    Thus, the discrete-time SGN step-size bounds (Theorem 4.3 / mirror analogue) apply directly.
    """
    theta = theta_init.clone().detach()
    
    # Metrics
    # Lyapunov V(x) = sum w_i KL(pi*_i || pi_i)
    # Distance = norm(theta - theta*)
    
    history = {
        'loss': [],
        'dist': [],
        'lyapunov': [],
        'converged': False
    }
    
    w1, w2 = weights
    n_p = 4
    
    # Precompute target policies for KL
    _, pi1_star, pi2_star = game.get_joint_policy(theta_star)
    
    for k in range(n_steps):
        theta.requires_grad_(True)
        l1, l2 = game.get_loss(theta)
        
        # Gradients
        g1 = torch.autograd.grad(l1, theta, create_graph=False, retain_graph=True)[0]
        g2 = torch.autograd.grad(l2, theta, create_graph=False)[0] # Recompute to be safe/clean
        # Actually, get_loss returns scalar. We need gradients wrt respective params.
        # g1 is full vector.
        
        # Extract player gradients
        grad1 = g1[:n_p]
        grad2 = g2[n_p:]
        
        # Fishers
        _, F1, F2 = game.get_fisher(theta)
        
        # Natural Gradients
        # Delta1 = F1^-1 g1
        delta1 = torch.linalg.lstsq(F1, grad1).solution
        delta2 = torch.linalg.lstsq(F2, grad2).solution
        
        # Update
        with torch.no_grad():
            theta[:n_p] -= eta * delta1
            theta[n_p:] -= eta * delta2
            
        # Metrics
        with torch.no_grad():
            # Distance
            dist = torch.norm(theta - theta_star).item()
            
            # Lyapunov
            _, pi1, pi2 = game.get_joint_policy(theta)
            
            # KL(p* || p) = sum p* log(p*/p)
            # Sum over states
            kl1 = torch.sum(pi1_star * (torch.log(pi1_star + 1e-12) - torch.log(pi1 + 1e-12)))
            kl2 = torch.sum(pi2_star * (torch.log(pi2_star + 1e-12) - torch.log(pi2 + 1e-12)))
            
            lyap = (w1 * kl1 + w2 * kl2).item()
            
            # Check gradient norm for convergence
            gnorm = torch.sqrt(torch.sum(grad1**2) + torch.sum(grad2**2)).item()
            
            history['loss'].append(l1.item() + l2.item())
            history['dist'].append(dist)
            history['lyapunov'].append(lyap)
            
            if gnorm < 1e-6:
                history['converged'] = True
                break
            
            if dist > 100 or np.isnan(dist):
                history['converged'] = False
                break
                
    return history

def run_epg(game, theta_init, theta_star, eta, weights, n_steps):
    """
    Euclidean Policy Gradient (Baseline).
    Update: theta_i <- theta_i - eta * g_i  (Ignoring weights usually, or use same scaling?)
    Plan says: "Euclidean PG with the same step size".
    If we want a fair comparison of geometry, we should keep the magnitude similar.
    But Euclidean doesn't use weights. We'll use scalar eta.
    """
    theta = theta_init.clone().detach()
    
    history = {
        'loss': [],
        'dist': [],
        'lyapunov': [],
        'converged': False
    }
    
    w1, w2 = weights # Used only for Lyapunov calculation metric
    n_p = 4
    _, pi1_star, pi2_star = game.get_joint_policy(theta_star)

    for k in range(n_steps):
        theta.requires_grad_(True)
        l1, l2 = game.get_loss(theta)
        
        g1 = torch.autograd.grad(l1, theta, retain_graph=True)[0]
        g2 = torch.autograd.grad(l2, theta)[0]
        
        grad1 = g1[:n_p]
        grad2 = g2[n_p:]
        
        with torch.no_grad():
            theta[:n_p] -= eta * grad1
            theta[n_p:] -= eta * grad2
            
        with torch.no_grad():
            dist = torch.norm(theta - theta_star).item()
            
            _, pi1, pi2 = game.get_joint_policy(theta)
            kl1 = torch.sum(pi1_star * (torch.log(pi1_star + 1e-12) - torch.log(pi1 + 1e-12)))
            kl2 = torch.sum(pi2_star * (torch.log(pi2_star + 1e-12) - torch.log(pi2 + 1e-12)))
            lyap = (w1 * kl1 + w2 * kl2).item()
            
            gnorm = torch.sqrt(torch.sum(grad1**2) + torch.sum(grad2**2)).item()
            
            history['loss'].append(l1.item() + l2.item())
            history['dist'].append(dist)
            history['lyapunov'].append(lyap)
            
            if gnorm < 1e-6:
                history['converged'] = True
                break
            
            if dist > 100 or np.isnan(dist):
                history['converged'] = False
                break
                
    return history

def run_experiment(cfg: MarkovConfig) -> str:
    """
    Run the full Markov SGN experiment for a given configuration.
    Returns the path to the saved results file.
    """
    game = TabularMarkovGame(cfg)
    certifier = SGNCertifier(game)
    
    print("Finding Equilibrium...")
    theta_star = certifier.find_equilibrium()
    print(f"Equilibrium Found. Norm: {torch.norm(theta_star):.4f}")
    
    print("Computing SGN Constants...")
    sgn_data = certifier.compute_local_sgn(theta_star)
    
    alpha = sgn_data['alpha']
    beta = sgn_data['beta']
    best_r = sgn_data['best_r']
    w_opt = (1.0, best_r)
    
    print(f"SGN Results: Alpha={alpha:.4f}, Beta={beta:.4f}, Best Ratio={best_r:.4f}")
    
    if alpha <= 0:
        print("WARNING: Alpha <= 0. SGN failed to certify contraction.")
    
    # Derived Safe Step
    eta_max = 2 * alpha / (beta**2)
    eta_safe = 0.5 * eta_max
    
    print(f"Safe Step Size (eta_max): {eta_max:.4f}")
    print(f"Using eta = {eta_safe:.4f}")
    
    # 1. Main Comparison: NPG vs EPG at safe step
    print("Running Main Comparison...")
    # Init close to uniform but random
    # Uniform logits = 0.
    # Perturb
    
    seeds = range(cfg.seed, cfg.seed + cfg.n_seeds)
    
    npg_curves = []
    epg_curves = []
    
    for s in tqdm(seeds, desc="Comparison Seeds"):
        torch.manual_seed(s)
        theta_init = torch.randn_like(theta_star) * 0.1 
        # Small perturbation from 0 is better?
        # Plan says: N(0, 0.01^2).
        theta_init = torch.randn_like(theta_star) * 0.01
        
        h_npg = run_npg(game, theta_init, theta_star, eta_safe, w_opt, cfg.max_steps)
        h_epg = run_epg(game, theta_init, theta_star, eta_safe, w_opt, cfg.max_steps)
        
        npg_curves.append(h_npg)
        epg_curves.append(h_epg)
        
    # 2. Stability Sweep
    print("Running Step Size Sweep...")
    # Sweep a broad range of step-size multipliers so that the stability
    # curves are informative (showing both clearly stable and unstable
    # regimes) while still resolving behaviour near the certified SGN
    # bound at eta / eta_max = 1.
    eta_mults = np.logspace(-1, 1.0, 20)  # 0.1 to 10 x eta_max
    probs_npg = []
    probs_epg = []
    
    for m in tqdm(eta_mults, desc="Sweep"):
        eta = m * eta_max
        n_conv_npg = 0
        n_conv_epg = 0
        
        for s in seeds:
            torch.manual_seed(s)
            theta_init = torch.randn_like(theta_star) * 0.01
            
            h_npg = run_npg(game, theta_init, theta_star, eta, w_opt, 500) # Shorter run for check
            h_epg = run_epg(game, theta_init, theta_star, eta, w_opt, 500)
            
            # Check stability (did not diverge and final dist is small)
            # Converged flag is strictly gradient norm.
            # Let's use "didn't blow up" as stability proxy?
            # Or strict convergence.
            # NPG should converge.
            if h_npg['converged'] or h_npg['dist'][-1] < 1.0:
                 n_conv_npg += 1
            if h_epg['converged'] or h_epg['dist'][-1] < 1.0:
                 n_conv_epg += 1
                 
        probs_npg.append(n_conv_npg / len(seeds))
        probs_epg.append(n_conv_epg / len(seeds))
        
    # 3. Timescale Band Verification
    print("Verifying Timescale Band...")
    r_grid = np.logspace(-2, 2, cfg.n_r_grid)
    # Compute Theoretical Alpha Curve
    band_alphas = certifier.get_timescale_curve(sgn_data, r_grid)
    
    # Empirical Convergence Rate for Weighted NPG
    # Fix eta to something aggressive but safe for optimal?
    # If we fix eta, and alpha drops, eta might exceed 2 alpha / beta^2.
    # So we expect failure when alpha is small.
    # Use eta = eta_safe (calculated at optimal). 
    # When r is bad, alpha is small, so eta_safe might be > 2 alpha(r) / beta(r)^2.
    
    band_success = []
    for r in tqdm(r_grid, desc="Timescale"):
        w_r = (1.0, r)
        n_conv = 0
        for s in seeds:
            torch.manual_seed(s)
            theta_init = torch.randn_like(theta_star) * 0.01
            h = run_npg(game, theta_init, theta_star, eta_safe, w_r, 500)
            if h['converged'] or h['dist'][-1] < 1.0:
                n_conv += 1
        band_success.append(n_conv / len(seeds))
        
    # Save Results
    results = {
        'sgn_alpha': alpha,
        'sgn_beta': beta,
        'best_r': best_r,
        'eta_safe': eta_safe,
        'eta_max': eta_max,
        # Curves (save median or all? All is safer)
        'npg_curves_lyap': np.array([h['lyapunov'] for h in npg_curves], dtype=object),
        'epg_curves_lyap': np.array([h['lyapunov'] for h in epg_curves], dtype=object),
        'npg_curves_dist': np.array([h['dist'] for h in npg_curves], dtype=object),
        'epg_curves_dist': np.array([h['dist'] for h in epg_curves], dtype=object),
        # Sweep
        'sweep_mults': eta_mults,
        'sweep_probs_npg': probs_npg,
        'sweep_probs_epg': probs_epg,
        # Band
        'band_r_grid': r_grid,
        'band_alphas': band_alphas,
        'band_success': band_success
    }
    
    np.savez(cfg.results_path, **results)
    print(f"Results saved to {cfg.results_path}")
    return cfg.results_path

def main():
    cfg = MarkovConfig()
    run_experiment(cfg)

if __name__ == "__main__":
    main()

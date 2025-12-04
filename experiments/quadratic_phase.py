"""
Visualizes the "Escape vs. Trap" phase portraits for the 2D quadratic game (Section 5.1).

This script simulates the gradient flow of the game:
    f_1(x) = 0.5*mu1*x1^2 + a*x1*x2
    f_2(x) = 0.5*mu2*x2^2 + b*x1*x2
    
It produces Figure 1 of the paper, showing:
(a) The trajectory escaping the Euclidean unit ball (monotonicity failure).
(b) The same trajectory strictly entering the SGN metric ellipsoid (certified contraction).

The output is saved to `figs/mini-toy/quadratic_phase.pdf`.
"""
import os
import matplotlib.pyplot as plt
import numpy as np

# Paper-style defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 300,
    "axes.unicode_minus": False,
})

def make_phase_plot():
    # Quadratic game parameters
    mu1 = 1.0
    mu2 = 1.0
    a = 10.0
    b = 0.05

    # SGN-balanced weights: w1/w2 = L21/L12 = 0.05/10 = 1/200 => r = 200.
    w1 = 1.0
    w2 = 200.0

    # Standard vector field: -F(x)
    # Note: This is the SAME dynamics for both panels.
    def G_standard(x, y):
        return -(mu1 * x + a * y), -(mu2 * y + b * x)

    # Grid for streamplot
    limit = 2.0
    grid = np.linspace(-limit, limit, 80)
    X, Y = np.meshgrid(grid, grid)
    U, V = G_standard(X, Y)

    # Trajectory simulator (Euler)
    def simulate(x0, steps=2000, h=0.001):
        traj = np.zeros((steps + 1, 2))
        traj[0] = x0
        for k in range(steps):
            dx, dy = G_standard(traj[k, 0], traj[k, 1])
            traj[k + 1] = traj[k] + h * np.array([dx, dy])
        return traj

    # Start point that exhibits transient growth
    # The eigenvector of the symmetric part corresponding to negative eigenvalue
    # is roughly aligned with x2 = -x1.
    x0 = np.array([0.15, -0.15])
    traj = simulate(x0, steps=3000, h=0.0005)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharex=False, sharey=False, constrained_layout=True)

    # Colors
    stream_color = "#dddddd" # faint grey
    traj_color = "#d62728"   # red
    geom_color = "#1f77b4"   # blue

    # --- Panel (a): Euclidean View ---
    ax = axes[0]
    # Streamlines
    ax.streamplot(X, Y, U, V, color=stream_color, density=0.8, linewidth=0.6, arrowsize=0.6)
    
    # Euclidean Unit Circle (Radius = norm of x0 for reference)
    r0 = np.linalg.norm(x0)
    circle = plt.Circle((0, 0), r0, color=geom_color, fill=False, linestyle='--', linewidth=1.5, label='Euclidean Ball')
    ax.add_patch(circle)

    # Trajectory
    ax.plot(traj[:, 0], traj[:, 1], color=traj_color, linewidth=1.5, label='Trajectory')
    
    # Mark start
    ax.scatter(x0[0], x0[1], color='k', s=20, zorder=5, label='Start')

    ax.set_title(r"(a) Euclidean View: Escapes the Ball")
    ax.legend(loc='upper right', fontsize=7)
    
    # Limits for Euclidean view (Zoomed in)
    ax.set_xlim([-0.6, 0.6])
    ax.set_ylim([-0.6, 0.6])

    # --- Panel (b): SGN View ---
    ax = axes[1]
    # Streamlines (Same flow!)
    ax.streamplot(X, Y, U, V, color=stream_color, density=0.8, linewidth=0.6, arrowsize=0.6)

    # SGN Ellipse
    # Energy level E0 = w1*x0^2 + w2*y0^2
    E0 = w1*x0[0]**2 + w2*x0[1]**2
    # Plot contour w1*x^2 + w2*y^2 = E0
    # Parametric plot: x = sqrt(E0/w1) cos t, y = sqrt(E0/w2) sin t
    t = np.linspace(0, 2*np.pi, 200)
    ex = np.sqrt(E0/w1) * np.cos(t)
    ey = np.sqrt(E0/w2) * np.sin(t)
    ax.plot(ex, ey, color=geom_color, linestyle='--', linewidth=1.5, label='SGN Metric Ball')

    # Trajectory (Same trajectory!)
    ax.plot(traj[:, 0], traj[:, 1], color=traj_color, linewidth=1.5, label='Trajectory')
    
    # Mark start
    ax.scatter(x0[0], x0[1], color='k', s=20, zorder=5, label='Start')

    ax.set_title(r"(b) SGN View: Trapped in the Ellipse")
    ax.legend(loc='upper right', fontsize=7)
    
    # Limits for SGN view (Zoomed out to see full ellipse)
    # Intercept is sqrt(4.5) approx 2.12.
    ax.set_xlim([-2.4, 2.4])
    ax.set_ylim([-0.6, 0.6])

    # Formatting
    for ax in axes:
        ax.set_xlabel(r"$x_1$")
        ax.axhline(0.0, color="0.5", linewidth=0.6, alpha=0.8)
        ax.axvline(0.0, color="0.5", linewidth=0.6, alpha=0.8)
        # Aspect ratio auto to allow different ranges to fill the square panels
        # ax.set_aspect("equal", adjustable="box") 
        ax.tick_params(width=0.6, length=2.6, labelsize=7)
    axes[0].set_ylabel(r"$x_2$")

    # Save
    base_dir = os.path.dirname(os.path.abspath(__file__))
    figs_dir = os.path.join(base_dir, "..", "figs", "mini-toy")
    os.makedirs(figs_dir, exist_ok=True)
    
    pdf_path = os.path.join(figs_dir, "quadratic_phase.pdf")
    png_path = os.path.join(figs_dir, "quadratic_phase.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    print(f"Saved phase plots to {pdf_path}")

if __name__ == "__main__":
    make_phase_plot()

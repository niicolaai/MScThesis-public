import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Joint limits (4 joints × 2 sides = 8 values)
joint_angle_min_constraint = [
    -0.3,   # HipAA_L: Adduction limit
    -0.3,   # HipAA_R
    -0.7,   # HipFE_L: Extension limit
    -0.7,   # HipFE_R
    0.0,    # KneeFE_L: Full extension
    0.0,    # KneeFE_R
    -0.65,  # AnkleFE_L: Plantarflexion limit
    -0.65,  # AnkleFE_R
]

joint_angle_max_constraint = [
    0.35,   # HipAA_L: Abduction limit
    0.35,   # HipAA_R
    2.0,    # HipFE_L: Flexion limit
    2.0,    # HipFE_R
    2.1,    # KneeFE_L: Flexion limit
    2.1,    # KneeFE_R
    0.65,   # AnkleFE_L: Dorsiflexion limit
    0.65,   # AnkleFE_R
]

# Joint names for labeling
joint_names = ["HipAA", "HipFE", "KneeFE", "AnkleFE"]


def h_exp(s, s_min, s_max, delta_a, delta_b):
    """
    Compute the exponential control barrier function.

    h_exp(s, δ) = δ_a * [1 - (e^(δ_b(s - s_max)) + e^(δ_b(s_min - s)))]
    """
    term1 = np.exp(delta_b * (s - s_max))
    term2 = np.exp(delta_b * (s_min - s))
    h = delta_a * (1.0 - (term1 + term2))
    return h


def plot_cbf_single_joint(ax, s_min, s_max, delta_a_values, delta_b):
    """
    Plot CBF for a single joint with different delta_a values.
    """
    # Create state range: extend beyond bounds for visualization
    range_margin = 0.3
    s = np.linspace(s_min - range_margin, s_max + range_margin, 300)

    # Plot for different delta_a values
    colors = plt.cm.viridis(np.linspace(0, 1, len(delta_a_values)))

    for delta_a, color in zip(delta_a_values, colors):
        h_vals = h_exp(s, s_min, s_max, delta_a, delta_b)
        ax.plot(s, h_vals, linewidth=2.5, label=f'δ_a={delta_a}', color=color)

    # Mark the safe region
    ax.axvspan(s_min, s_max, alpha=0.1, color='green', label='Safe region')

    # Mark boundaries
    ax.axvline(s_min, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Bounds')
    ax.axvline(s_max, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    # Mark zero crossing (barrier boundary)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

    ax.set_xlabel('Joint angle (rad)', fontsize=10)
    ax.set_ylabel('h(s)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')


def plot_cbf_parameter_effects(ax, s_min, s_max, delta_a, delta_b_values):
    """
    Plot CBF for a single joint with different delta_b values.
    """
    # Create state range
    range_margin = 0.3
    s = np.linspace(s_min - range_margin, s_max + range_margin, 300)

    # Plot for different delta_b values
    colors = plt.cm.plasma(np.linspace(0, 1, len(delta_b_values)))

    for delta_b, color in zip(delta_b_values, colors):
        h_vals = h_exp(s, s_min, s_max, delta_a, delta_b)
        ax.plot(s, h_vals, linewidth=2.5, label=f'δ_b={delta_b}', color=color)

    # Mark the safe region
    ax.axvspan(s_min, s_max, alpha=0.1, color='green', label='Safe region')

    # Mark boundaries
    ax.axvline(s_min, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Bounds')
    ax.axvline(s_max, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    # Mark zero crossing
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

    ax.set_xlabel('Joint angle (rad)', fontsize=10)
    ax.set_ylabel('h(s)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')


# Create 4 figures (one for each joint type)
for joint_idx in range(4):
    left_idx = joint_idx * 2      # Left joint (same limits as right)

    s_min = joint_angle_min_constraint[left_idx]
    s_max = joint_angle_max_constraint[left_idx]

    joint_name = joint_names[joint_idx]

    # Create figure with subplots
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 2, figure=fig, hspace=0.35, wspace=0.3)

    # === Left: Varying delta_a with fixed delta_b ===
    ax1 = fig.add_subplot(gs[0, 0])
    delta_a_values = [0.5, 1.0, 2.0, 5.0, 10.0]
    delta_b_fixed = 5.0
    plot_cbf_single_joint(ax1, s_min, s_max, delta_a_values, delta_b_fixed)
    ax1.set_title(f'Effect of δ_a (δ_b={delta_b_fixed})', fontsize=11, fontweight='bold')
    ax1.set_ylim(-50, 10)

    # === Right: Varying delta_b with fixed delta_a ===
    ax2 = fig.add_subplot(gs[0, 1])
    delta_b_values = [5.0, 10.0, 20.0, 50.0, 100.0]
    delta_a_fixed = 1.0
    plot_cbf_parameter_effects(ax2, s_min, s_max, delta_a_fixed, delta_b_values)
    ax2.set_title(f'Effect of δ_b (δ_a={delta_a_fixed})', fontsize=11, fontweight='bold')
    ax2.set_ylim(-50, 10)
    ax2.legend(loc='best')

    # Overall figure title
    fig.suptitle(f'Control Barrier Function for {joint_name} Joint',
                 fontsize=13, fontweight='bold', y=1.02)

    # Save figure
    filename = f'cbf_{joint_name.lower()}_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


print("\nAll CBF plots generated successfully!")

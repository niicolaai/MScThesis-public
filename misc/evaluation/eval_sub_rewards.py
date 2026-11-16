"""
Evaluation script for analyzing sub-rewards logged during training or playing.

This script loads the sub-rewards.npz file and provides comprehensive analysis
including statistics, visualizations, and insights into reward component behavior.
"""
# flake8: noqa: W293

import numpy as np
import os
import glob
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================
NPZ_PATTERN = "*sub_rewards.npz"
DT = 0.01  # Time delta between timesteps in seconds
NUM_ENVS_TO_LOAD = 20  # Number of environments to load (None = load all)
DOWNSAMPLE_FACTOR = 50  # Downsample data by this factor (1 = no downsampling)

# Find matching sub_rewards files
matching_files = glob.glob(NPZ_PATTERN)
if not matching_files:
    raise FileNotFoundError(f"Could not find any sub-rewards files matching pattern: {NPZ_PATTERN}")

# Use the most recent file (by modification time)
NPZ_PATH = max(matching_files, key=os.path.getmtime)
print(f"Loading sub-rewards from: {NPZ_PATH}")

# ============================================================================
# DATA STRUCTURE
# ============================================================================
# Each reward component is stored as shape (num_steps, num_envs)
REWARD_COMPONENTS = [
    "orientation_reward",
    "height_deviation_penalty",
    "torque_limit_violation_penalty",
    "action_change_penalty",
    "joint_angle_constraint_penalty",
    "cbf_h_value",
    "cbf_h_dot",
    "cbf_h_torque_value",
    "cbf_h_torque_dot",
    "joint_symmetry_penalty",
    "spawn_position_penalty",
    "walk_velocity_reward",
    "imitation_q_reward",
    "imitation_dq_reward",
    "imitation_pelvis_reward",
    "alive_reward",
    "early_termination_penalty",
    "total_reward",
]

PLOT_TITLE = [
    "Orientation Reward",
    "Height Deviation Reward",
    "Torque Limit Violation Penalty",
    "Action Change Reward",
    "Joint Angle Constraint Reward",
    "CBF H Value (Joint Angle)",
    "CBF H Dot (Joint Angle)",
    "CBF H Value (Torque)",
    "CBF H Dot (Torque)",
    "Joint Symmetry Penalty",
    "Spawn Position Penalty",
    "Walking Velocity Reward",
    "Imitation Q (Joint Pos) Reward",
    "Imitation Dq (Joint Vel) Reward",
    "Imitation Pelvis Reward",
    "Alive Reward",
    "Early Termination Penalty",
    "Total Reward",
]

# Load and process data efficiently - only load what we need
rewards = {}
with np.load(NPZ_PATH) as data:
    print("Successfully loaded sub-rewards data")
    for component in REWARD_COMPONENTS:
        if component in data.files:
            env_slice = slice(None, NUM_ENVS_TO_LOAD)
            time_slice = slice(None, None, DOWNSAMPLE_FACTOR)
            rewards[component] = data[component][time_slice, env_slice]
            print(f"  {component}: shape {rewards[component].shape}")

num_steps, num_envs = rewards["total_reward"].shape
time_array = np.arange(num_steps) * DT * DOWNSAMPLE_FACTOR

# ============================================================================
# VISUALIZATION
# ============================================================================
output_dir = "/home/nicolai/Documents/exoWE2/exoRL/evaluation/sub_rewards_plots"
os.makedirs(output_dir, exist_ok=True)

# Also save individual component plots
fig2, axes = plt.subplots(4, 5, figsize=(16, 16))
axes = axes.flatten()

for idx, component in enumerate(REWARD_COMPONENTS):
    if component in rewards:
        ax = axes[idx]

        # Calculate mean and 95% confidence interval (percentile-based)
        mean_values = np.mean(rewards[component], axis=1)
        ci_lower = np.percentile(rewards[component], 2.5, axis=1)
        ci_upper = np.percentile(rewards[component], 97.5, axis=1)

        # Apply smoothing to mean
        window = min(50, max(5, num_steps // 20))
        mean_values = np.convolve(mean_values, np.ones(window) / window, mode="valid")
        time_array_smooth = time_array[:len(mean_values)]
        ci_lower = np.convolve(ci_lower, np.ones(window) / window, mode="valid")
        ci_upper = np.convolve(ci_upper, np.ones(window) / window, mode="valid")

        # Plot mean with confidence interval band
        ax.plot(time_array_smooth, mean_values, color="black", linewidth=1.5, label="Mean")
        ax.fill_between(time_array_smooth, ci_lower, ci_upper, 
                        alpha=0.2, color="gray", label="95% CI")

        ax.set_title(PLOT_TITLE[idx], fontweight="bold", fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Reward", fontsize=9)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
        ax.tick_params(labelsize=8)

# Hide any empty subplots in the 6x2 grid (if any)
for idx in range(len(REWARD_COMPONENTS), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
output_file2_png = os.path.join(output_dir, "sub_rewards_components.png")
fig2.savefig(output_file2_png, dpi=300, bbox_inches="tight")
print(f"Component plots saved to: {output_file2_png}")
output_file2_svg = os.path.join(output_dir, "sub_rewards_components.svg")
fig2.savefig(output_file2_svg, dpi=300, bbox_inches="tight", format="svg")
print(f"Component plots saved to: {output_file2_svg}")

plt.show()

print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)

"""
Bulk evaluation script for analyzing multiple sub-rewards files at once.

This script loads all sub-rewards.npz files from a rewards/ subfolder,
concatenates them, and provides comprehensive analysis with transition markers
showing where one file ends and the next begins.
"""
# flake8: noqa: W293

import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================
REWARDS_DIR = "rewards/"
DT = 0.01  # Time delta between timesteps in seconds
NUM_ENVS_TO_LOAD = 20  # Number of environments to load (None = load all)
DOWNSAMPLE_FACTOR = 50  # Downsample data by this factor (1 = no downsampling)

print(f"Loading sub-rewards from: {REWARDS_DIR}")
if not os.path.exists(REWARDS_DIR):
    raise FileNotFoundError(f"Could not find rewards directory at {REWARDS_DIR}")

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
    "CBF H Value",
    "CBF H Dot",
    "CBF H Torque Value",
    "CBF H Torque Dot",
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

# ============================================================================
# FIND AND LOAD NPZ FILES
# ============================================================================
npz_files = sorted(glob.glob(os.path.join(REWARDS_DIR, "*.npz")))

if not npz_files:
    raise FileNotFoundError(f"No .npz files found in {REWARDS_DIR}")

print(f"Found {len(npz_files)} NPZ files")
print(f"\nProcessing files in order...")
for i, npz_file in enumerate(npz_files):
    print(f"  {i+1}. {os.path.basename(npz_file)}")

# ============================================================================
# LOAD AND CONCATENATE DATA
# ============================================================================
rewards_concat = {component: [] for component in REWARD_COMPONENTS}
transition_indices = [0]  # Track where each file ends

total_steps = 0

for file_idx, npz_file in enumerate(npz_files):
    print(f"\nLoading {os.path.basename(npz_file)}...")
    
    with np.load(npz_file) as data:
        # Get first environment's reward to determine number of steps
        if REWARD_COMPONENTS[0] in data.files:
            first_component = data[REWARD_COMPONENTS[0]]
            num_steps_in_file = first_component.shape[0]
            print(f"  Shape: {first_component.shape}")
            
            # Load each component
            for component in REWARD_COMPONENTS:
                if component in data.files:
                    env_slice = slice(None, NUM_ENVS_TO_LOAD)
                    time_slice = slice(None, None, DOWNSAMPLE_FACTOR)
                    component_data = data[component][time_slice, env_slice]
                    rewards_concat[component].append(component_data)

# Concatenate all components along the time axis
final_rewards = {}
for component in REWARD_COMPONENTS:
    if rewards_concat[component]:
        final_rewards[component] = np.concatenate(rewards_concat[component], axis=0)

# Calculate transition indices (in downsampled steps)
current_step = 0
transition_indices = [0]
for i, npz_file in enumerate(npz_files[:-1]):
    with np.load(npz_file) as data:
        if REWARD_COMPONENTS[0] in data.files:
            num_steps = data[REWARD_COMPONENTS[0]].shape[0]
            downsampled_steps = (num_steps + DOWNSAMPLE_FACTOR - 1) // DOWNSAMPLE_FACTOR
            current_step += downsampled_steps
            transition_indices.append(current_step)

print(f"\nTransition indices (in downsampled time steps): {transition_indices}")

num_steps, num_envs = final_rewards["total_reward"].shape
time_array = np.arange(num_steps) * DT * DOWNSAMPLE_FACTOR

print(f"Total concatenated data shape: {final_rewards['total_reward'].shape}")

# ============================================================================
# VISUALIZATION
# ============================================================================
output_dir = "/home/nicolai/Documents/exoWE2/exoRL/evaluation/sub_rewards_plots"
os.makedirs(output_dir, exist_ok=True)

# Create individual component plots with transition markers
fig2, axes = plt.subplots(5, 4, figsize=(16, 16))
axes = axes.flatten()

for idx, component in enumerate(REWARD_COMPONENTS):
    if component in final_rewards:
        ax = axes[idx]

        # Calculate mean and 95% confidence interval (percentile-based)
        mean_values = np.mean(final_rewards[component], axis=1)
        ci_lower = np.percentile(final_rewards[component], 2.5, axis=1)
        ci_upper = np.percentile(final_rewards[component], 97.5, axis=1)

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

        # Add transition markers (skip the first one at 0)
        for trans_idx_step in transition_indices[1:]:
            trans_time = trans_idx_step * DT * DOWNSAMPLE_FACTOR
            ax.axvline(trans_time, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

        ax.set_title(PLOT_TITLE[idx], fontweight="bold", fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Reward", fontsize=9)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
        ax.tick_params(labelsize=8)

# Hide any empty subplots in the grid
for idx in range(len(REWARD_COMPONENTS), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
output_file2_png = os.path.join(output_dir, "bulk_sub_rewards_components.png")
fig2.savefig(output_file2_png, dpi=300, bbox_inches="tight")
print(f"\nComponent plots saved to: {output_file2_png}")
output_file2_svg = os.path.join(output_dir, "bulk_sub_rewards_components.svg")
fig2.savefig(output_file2_svg, dpi=300, bbox_inches="tight", format="svg")
print(f"Component plots saved to: {output_file2_svg}")

plt.show()

print("\n" + "=" * 80)
print("BULK EVALUATION COMPLETE")
print("=" * 80)

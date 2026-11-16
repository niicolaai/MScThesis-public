"""
Export selected sub-rewards to CSV for pgfplots visualization.

This script loads all sub-rewards.npz files, calculates mean and 95% CI
across all environments, computes cumulated reward, downsamples to ~500 rows,
and exports to CSV with proper column structure for pgfplots.

Output files:
  - export_subrewards_pgfplots.csv: Main data for plotting
  - transition_indices_downsampled.txt: Curriculum transition indices
"""

import numpy as np
import os
import glob
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================
REWARDS_DIR = "rewards/"
DT = 0.01  # Time delta between timesteps in seconds
NUM_ENVS = 50  # Number of environments
TARGET_CSV_ROWS = 500  # Target number of rows in output CSV

print(f"Loading sub-rewards from: {REWARDS_DIR}")
if not os.path.exists(REWARDS_DIR):
    raise FileNotFoundError(f"Could not find rewards directory at {REWARDS_DIR}")

# ============================================================================
# SELECTED SUBREWARDS FOR PGFPLOTS
# ============================================================================
# These are the subrewards we want to include in the output CSV
SELECTED_SUBREWARDS = [
    "orientation_reward",
    "height_deviation_penalty",
    "alive_reward",
    "early_termination_penalty",
    "torque_limit_violation_penalty",
    "action_change_penalty",
    "joint_angle_constraint_penalty",
    "joint_symmetry_penalty",
]

SUBREWARD_LABELS = {
    "orientation_reward": "R_orient",
    "height_deviation_penalty": "R_height",
    "alive_reward": "R_alive",
    "early_termination_penalty": "R_early_term",
    "torque_limit_violation_penalty": "R_torque_limit",
    "action_change_penalty": "R_action_change",
    "joint_angle_constraint_penalty": "R_joint_angle",
    "joint_symmetry_penalty": "R_symmetry",
}

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
rewards_concat = {component: [] for component in SELECTED_SUBREWARDS}
transition_indices = [0]  # Track where each file ends (in original steps)

total_steps_original = 0

for file_idx, npz_file in enumerate(npz_files):
    print(f"\nLoading {os.path.basename(npz_file)}...")
    
    with np.load(npz_file) as data:
        # Get first subreward to determine number of steps
        if SELECTED_SUBREWARDS[0] in data.files:
            first_component = data[SELECTED_SUBREWARDS[0]]
            num_steps_in_file = first_component.shape[0]
            num_envs_in_file = first_component.shape[1]
            print(f"  Shape: {first_component.shape}")
            
            # Validate and load each subreward
            for component in SELECTED_SUBREWARDS:
                if component in data.files:
                    # Load all environments for this component
                    component_data = data[component][:, :NUM_ENVS]
                    rewards_concat[component].append(component_data)
                else:
                    print(f"  WARNING: {component} not found in {os.path.basename(npz_file)}")
            
            total_steps_original += num_steps_in_file
            transition_indices.append(total_steps_original)

# Concatenate all components along the time axis
print("\nConcatenating data...")
final_rewards = {}
for component in SELECTED_SUBREWARDS:
    if rewards_concat[component]:
        final_rewards[component] = np.concatenate(rewards_concat[component], axis=0)

num_steps_original, num_envs = final_rewards[SELECTED_SUBREWARDS[0]].shape
print(f"Original data shape: ({num_steps_original}, {num_envs})")
print(f"Total steps across all files: {total_steps_original}")

# Calculate appropriate downsampling factor
downsample_factor = max(1, num_steps_original // TARGET_CSV_ROWS)
num_steps_downsampled = (num_steps_original + downsample_factor - 1) // downsample_factor
print(f"\nDownsampling by factor of {downsample_factor}")
print(f"Downsampled data will have {num_steps_downsampled} rows")

# ============================================================================
# DOWNSAMPLE AND CALCULATE STATISTICS
# ============================================================================
print("\nCalculating statistics (mean and 95% CI)...")

# Downsample and calculate stats for each subreward
time_array_downsampled = np.arange(num_steps_downsampled) * DT * downsample_factor
csv_data = {"time": time_array_downsampled}

# Calculate cumulated reward
cumulated_reward_full = np.zeros((num_steps_original, num_envs))
for component in SELECTED_SUBREWARDS:
    cumulated_reward_full += final_rewards[component]

# Downsample cumulated reward
cumulated_downsampled = cumulated_reward_full[::downsample_factor, :]
# Handle edge case where last step might be incomplete
if num_steps_original % downsample_factor != 0:
    last_idx = (num_steps_downsampled - 1) * downsample_factor
    if last_idx < num_steps_original:
        cumulated_downsampled[-1] = cumulated_reward_full[last_idx:]

# Calculate cumulated stats
cumulated_mean = np.mean(cumulated_downsampled, axis=1)
cumulated_ci_lower = np.percentile(cumulated_downsampled, 2.5, axis=1)
cumulated_ci_upper = np.percentile(cumulated_downsampled, 97.5, axis=1)

csv_data["cumulated_reward"] = cumulated_mean
csv_data["cumulated_ci_lower"] = cumulated_ci_lower
csv_data["cumulated_ci_upper"] = cumulated_ci_upper

# Calculate stats for each selected subreward
for component in SELECTED_SUBREWARDS:
    label = SUBREWARD_LABELS[component]
    
    # Downsample
    component_downsampled = final_rewards[component][::downsample_factor, :]
    # Handle edge case
    if num_steps_original % downsample_factor != 0:
        last_idx = (num_steps_downsampled - 1) * downsample_factor
        if last_idx < num_steps_original:
            component_downsampled[-1] = final_rewards[component][last_idx:]
    
    # Calculate statistics
    mean_values = np.mean(component_downsampled, axis=1)
    ci_lower = np.percentile(component_downsampled, 2.5, axis=1)
    ci_upper = np.percentile(component_downsampled, 97.5, axis=1)
    
    csv_data[label] = mean_values
    csv_data[f"{label}_ci_lower"] = ci_lower
    csv_data[f"{label}_ci_upper"] = ci_upper

# ============================================================================
# CREATE DATAFRAME AND EXPORT TO CSV
# ============================================================================
df = pd.DataFrame(csv_data)

output_file = "export_subrewards_pgfplots.csv"
df.to_csv(output_file, index=False)
print(f"\nCSV exported to: {output_file}")
print(f"CSV shape: {df.shape}")
print(f"\nColumn names:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

# ============================================================================
# CALCULATE AND EXPORT TRANSITION INDICES (DOWNSAMPLED)
# ============================================================================
print("\n" + "=" * 80)
print("CURRICULUM TRAINING TRANSITION INDICES (downsampled)")
print("=" * 80)

# Convert original transition indices to downsampled indices
transition_indices_downsampled = []
for trans_idx in transition_indices[1:]:  # Skip the first 0
    downsampled_idx = (trans_idx + downsample_factor - 1) // downsample_factor
    downsampled_idx = min(downsampled_idx, num_steps_downsampled - 1)  # Clamp to valid range
    transition_indices_downsampled.append(downsampled_idx)

print(f"\nFound {len(transition_indices_downsampled)} transition(s)")
print(f"Transition indices (row numbers in downsampled CSV):")
for i, idx in enumerate(transition_indices_downsampled, 1):
    print(f"  {i}. Row {idx} (time: {time_array_downsampled[idx]:.2f}s)")

# Save to file
transition_file = "transition_indices_downsampled.txt"
with open(transition_file, "w") as f:
    f.write("Curriculum Training Transition Indices (Downsampled Data)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Downsampling factor: {downsample_factor}\n")
    f.write(f"Time delta: {DT}s\n\n")
    f.write("Row indices in downsampled CSV:\n")
    for i, idx in enumerate(transition_indices_downsampled, 1):
        f.write(f"{i}. Row {idx} (time: {time_array_downsampled[idx]:.2f}s)\n")

print(f"\nTransition indices saved to: {transition_file}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("EXPORT COMPLETE")
print("=" * 80)
print(f"CSV file: {output_file}")
print(f"Transitions file: {transition_file}")
print(f"Final data shape: {num_steps_downsampled} rows × {len(df.columns)} columns")

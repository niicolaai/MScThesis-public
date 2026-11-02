"""
Export play data to CSV for pgfplots visualization.

This script loads play_data.npz, downsamples to ~500 rows, and exports
CSV with data needed for plotting:
  - Figure 2: Base Euler Angles (Roll, Pitch)
  - Figure 4: Base Angular Velocities (all 3 dimensions)
  - Figure 8: Actual vs Desired Joint Positions

All angles are converted to degrees, velocities to deg/s, and actions scaled to Nm.
"""

import numpy as np
import os
import pandas as pd
from scipy.spatial.transform import Rotation

# ============================================================================
# CONFIGURATION
# ============================================================================
NPZ_PATH = "play_data.npz"
DT = 0.01  # Time delta between timesteps in seconds
TARGET_CSV_ROWS = 500  # Target number of rows in output CSV
ACTION_SCALE = 100  # Scale actions to Nm

print(f"Loading NPZ file from: {NPZ_PATH}")
if not os.path.exists(NPZ_PATH):
    raise FileNotFoundError(f"Could not find {NPZ_PATH}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def quaternion_to_euler(quat_data):
    """
    Convert quaternion data (w, x, y, z) to Euler angles (roll, pitch, yaw).
    
    Args:
        quat_data: numpy array of shape (N, 4) with quaternions [w, x, y, z]
    
    Returns:
        numpy array of shape (N, 3) with Euler angles in radians [roll, pitch, yaw]
    """
    # scipy expects (x, y, z, w) format; our data is (w, x, y, z)
    quat_reordered = quat_data[:, [1, 2, 3, 0]]  # [x, y, z, w]
    rotations = Rotation.from_quat(quat_reordered)
    euler = rotations.as_euler("xyz", degrees=False)  # Roll, Pitch, Yaw in radians
    return euler

# ============================================================================
# LOAD DATA
# ============================================================================
data = np.load(NPZ_PATH, allow_pickle=True)

observations = data["observations"]
actions = data["actions"]

# Extract observation components
# Observation structure: [joint_pos_act_norm(8), joint_vel_act_norm(8), roll_pitch_norm(2), 
#                        base_ang_vel_norm(3), speed_x_norm(1), speed_y_norm(1)]
obs_data = np.array([obs["policy"].flatten() for obs in observations], dtype=np.float32)

# Extract individual components based on indices
joint_pos = obs_data[:, 0:8]      # 8 dimensions
joint_vel = obs_data[:, 8:16]     # 8 dimensions
roll_pitch = obs_data[:, 16:18]   # 2 dimensions
base_ang_vel = obs_data[:, 18:21]  # 3 dimensions (roll_rate, pitch_rate, yaw_rate)
speed_x = obs_data[:, 21:22]      # 1 dimension
speed_y = obs_data[:, 22:23]      # 1 dimension

# Extract and scale actions
actions_data = np.array([a.flatten() for a in actions], dtype=np.float32)
actions_data = actions_data * ACTION_SCALE  # Scale to Nm

num_steps_original = len(observations)
print(f"Original data: {num_steps_original} timesteps")

# ============================================================================
# CALCULATE DOWNSAMPLING FACTOR
# ============================================================================
# downsample_factor = max(1, num_steps_original // TARGET_CSV_ROWS)
downsample_factor = 1
num_steps_downsampled = (num_steps_original + downsample_factor - 1) // downsample_factor

print(f"Downsampling by factor: {downsample_factor}")
print(f"Downsampled data: {num_steps_downsampled} timesteps")

# ============================================================================
# DOWNSAMPLE DATA
# ============================================================================
time_array = np.arange(num_steps_downsampled) * DT * downsample_factor

joint_pos_ds = joint_pos[::downsample_factor]
joint_vel_ds = joint_vel[::downsample_factor]
roll_pitch_ds = roll_pitch[::downsample_factor]
base_ang_vel_ds = base_ang_vel[::downsample_factor]
speed_x_ds = speed_x[::downsample_factor]
speed_y_ds = speed_y[::downsample_factor]
actions_ds = actions_data[::downsample_factor]

# Handle edge case where last step might be incomplete
if num_steps_original % downsample_factor != 0:
    last_idx = (num_steps_downsampled - 1) * downsample_factor
    joint_pos_ds = np.vstack([joint_pos_ds, joint_pos[last_idx:]])
    joint_vel_ds = np.vstack([joint_vel_ds, joint_vel[last_idx:]])
    roll_pitch_ds = np.vstack([roll_pitch_ds, roll_pitch[last_idx:]])
    base_ang_vel_ds = np.vstack([base_ang_vel_ds, base_ang_vel[last_idx:]])
    speed_x_ds = np.vstack([speed_x_ds, speed_x[last_idx:]])
    speed_y_ds = np.vstack([speed_y_ds, speed_y[last_idx:]])
    actions_ds = np.vstack([actions_ds, actions_data[last_idx:]])

# Ensure we have exactly num_steps_downsampled rows
joint_pos_ds = joint_pos_ds[:num_steps_downsampled]
joint_vel_ds = joint_vel_ds[:num_steps_downsampled]
roll_pitch_ds = roll_pitch_ds[:num_steps_downsampled]
base_ang_vel_ds = base_ang_vel_ds[:num_steps_downsampled]
speed_x_ds = speed_x_ds[:num_steps_downsampled]
speed_y_ds = speed_y_ds[:num_steps_downsampled]
actions_ds = actions_ds[:num_steps_downsampled]

# ============================================================================
# CONVERT TO DEGREES AND DEG/S
# ============================================================================
# Convert angles from radians to degrees
roll_deg = np.degrees(roll_pitch_ds[:, 0])
pitch_deg = np.degrees(roll_pitch_ds[:, 1])

# Convert angular velocities from rad/s to deg/s (all 3 dimensions)
roll_rate_degps = np.degrees(base_ang_vel_ds[:, 0])
pitch_rate_degps = np.degrees(base_ang_vel_ds[:, 1])
yaw_rate_degps = np.degrees(base_ang_vel_ds[:, 2])

# Convert joint positions and velocities to degrees/deg/s
joint_pos_deg = np.degrees(joint_pos_ds)
joint_vel_degps = np.degrees(joint_vel_ds)

# Flatten speed arrays to 1D
speed_x_flat = speed_x_ds.flatten()
speed_y_flat = speed_y_ds.flatten()

# ============================================================================
# BUILD CSV DATA
# ============================================================================
csv_data = {
    "time": time_array,
    "roll": roll_deg,
    "pitch": pitch_deg,
    "roll_rate": roll_rate_degps,
    "pitch_rate": pitch_rate_degps,
    "yaw_rate": yaw_rate_degps,
    "speed_x": speed_x_flat,
    "speed_y": speed_y_flat,
}

# Joint names in order: Hip AA, Hip FE, Knee FE, Ankle FE (left-right pairs)
joint_names = ["hipAA", "hipFE", "kneeFE", "ankleFE"]

# Add joint positions (degrees)
for joint_idx, joint_name in enumerate(joint_names):
    left_col_idx = joint_idx * 2
    right_col_idx = joint_idx * 2 + 1
    csv_data[f"joint_pos_{joint_name}_L"] = joint_pos_deg[:, left_col_idx]
    csv_data[f"joint_pos_{joint_name}_R"] = joint_pos_deg[:, right_col_idx]

# Add joint velocities (deg/s)
for joint_idx, joint_name in enumerate(joint_names):
    left_col_idx = joint_idx * 2
    right_col_idx = joint_idx * 2 + 1
    csv_data[f"joint_vel_{joint_name}_L"] = joint_vel_degps[:, left_col_idx]
    csv_data[f"joint_vel_{joint_name}_R"] = joint_vel_degps[:, right_col_idx]

# Add actions (Nm) - these are desired torques
for joint_idx, joint_name in enumerate(joint_names):
    left_col_idx = joint_idx * 2
    right_col_idx = joint_idx * 2 + 1
    csv_data[f"action_{joint_name}_L"] = actions_ds[:, left_col_idx]
    csv_data[f"action_{joint_name}_R"] = actions_ds[:, right_col_idx]

# ============================================================================
# CREATE DATAFRAME AND EXPORT
# ============================================================================
df = pd.DataFrame(csv_data)

output_file = "play_data_downsampled.csv"
df.to_csv(output_file, index=False)
print(f"\nCSV exported to: {output_file}")
print(f"CSV shape: {df.shape}")
print(f"\nColumn names ({len(df.columns)} total):")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

print("\n" + "=" * 80)
print("EXPORT COMPLETE")
print("=" * 80)
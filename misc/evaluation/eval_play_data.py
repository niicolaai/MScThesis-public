# flake8: noqa: W293

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# ============================================================================
# CONFIGURATION
# ============================================================================
NPZ_PATH = "play_data.npz"
DT = 0.01  # Time delta between timesteps in seconds
START_TIMESTEP = 0  # Index of the starting timestep to process
END_TIMESTEP = None  # Set to None to process all timesteps
TORQUE_MODE_ACTIVE = True  # If True, actions are in Nm and scaled by 100; if False, actions are in rad

# Configure observation space (match the obs composition in your RL environment)
OBSERVATION_COMPONENTS = {
    "joint_pos_act": True,          # 8 dimensions
    "joint_vel_act": True,          # 8 dimensions
    "roll_pitch": True,             # 2 dimensions
    "base_lin_acc": False,          # 3 dimensions
    "base_ang_vel": True,           # 3 dimensions
}

# Select which figures to generate (set True/False for each)
PLOT_FIGURES = {
    "joint_angles_velocities": True,     # Figure 1
    "base_euler_angles": True,           # Figure 2
    "base_linear_accelerations": False,  # Figure 3
    "base_angular_velocities": True,     # Figure 4
    "actions": True,                     # Figure 6
    "rewards_dones": True,               # Figure 7
    "actual_vs_desired": True,           # Figure 8
}

# Set action scale and units based on torque mode
if TORQUE_MODE_ACTIVE:
    ACTION_SCALE = 100
    ACTION_UNITS = "Torque (Nm)"
else:
    ACTION_SCALE = 1
    ACTION_UNITS = "Angle (rad)"


# ============================================================================
# DYNAMIC OBSERVATION INDEX CALCULATOR
# ============================================================================
def compute_observation_indices(components_config):
    """
    Compute start and end indices for each observation component based on enabled components.
    Returns a dict with indices and component sizes.
    """
    component_sizes = {
        "joint_pos_act": 8,
        "joint_vel_act": 8,
        "roll_pitch": 2,
        "base_lin_acc": 3,
        "base_ang_vel": 3,
    }

    indices = {}
    current_idx = 0

    for component_name, enabled in components_config.items():
        if enabled:
            size = component_sizes[component_name]
            indices[component_name] = (current_idx, current_idx + size)
            current_idx += size

    indices["total_obs_size"] = current_idx
    return indices


# Calculate observation indices based on configuration
OBS_INDICES = compute_observation_indices(OBSERVATION_COMPONENTS)

# load the npz file
print(f"Loading NPZ file from: {NPZ_PATH}")
print(f"File exists: {os.path.exists(NPZ_PATH)}")
print()


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
    # scipy.spatial.transform.Rotation expects (x, y, z, w) format
    # Our data is (w, x, y, z), so we need to reorder
    quat_reordered = quat_data[:, [1, 2, 3, 0]]  # [x, y, z, w]
    rotations = Rotation.from_quat(quat_reordered)
    euler = rotations.as_euler("xyz", degrees=False)  # Roll, Pitch, Yaw in radians
    return euler


# Load the data
data = np.load(NPZ_PATH, allow_pickle=True)

# Extract and process data
observations = data["observations"]
actions = data["actions"]
rewards = data["rewards"]
dones = data["dones"]

# Process observations (extract policy data from dict)
obs_data = np.array([obs["policy"].flatten() for obs in observations], dtype=np.float32)
print(f"Observation data shape: {obs_data.shape}")

# Process actions (already in shape (200, 1, 8))
actions_data = np.array([a.flatten() for a in actions], dtype=np.float32)
actions_data = actions_data * ACTION_SCALE

# Process rewards - convert to float
rewards_data = np.array([r.flatten() for r in rewards], dtype=np.float32).flatten()

# Process dones - convert to float
dones_data = np.array([d.flatten() for d in dones], dtype=np.float32).flatten()

# Extract observation components based on enabled configuration
joint_pos = None
joint_vel = None
roll_pitch = None
base_lin_acc = None
base_ang_vel = None

if OBSERVATION_COMPONENTS["joint_pos_act"]:
    start, end = OBS_INDICES["joint_pos_act"]
    joint_pos = obs_data[:, start:end]

if OBSERVATION_COMPONENTS["joint_vel_act"]:
    start, end = OBS_INDICES["joint_vel_act"]
    joint_vel = obs_data[:, start:end]

if OBSERVATION_COMPONENTS["roll_pitch"]:
    start, end = OBS_INDICES["roll_pitch"]
    roll_pitch = obs_data[:, start:end]

if OBSERVATION_COMPONENTS["base_lin_acc"]:
    start, end = OBS_INDICES["base_lin_acc"]
    base_lin_acc = obs_data[:, start:end]

if OBSERVATION_COMPONENTS["base_ang_vel"]:
    start, end = OBS_INDICES["base_ang_vel"]
    base_ang_vel = obs_data[:, start:end]

print("Configuration in last timesteps")
if joint_pos is not None:
    print(f"Joint positions (last timestep): {joint_pos[-1]}")

# Apply MAX_TIMESTEPS limit if specified
if END_TIMESTEP is not None:
    obs_data = obs_data[START_TIMESTEP:END_TIMESTEP]
    actions_data = actions_data[START_TIMESTEP:END_TIMESTEP]
    rewards_data = rewards_data[START_TIMESTEP:END_TIMESTEP]
    dones_data = dones_data[START_TIMESTEP:END_TIMESTEP]
    if joint_pos is not None:
        joint_pos = joint_pos[START_TIMESTEP:END_TIMESTEP]
    if joint_vel is not None:
        joint_vel = joint_vel[START_TIMESTEP:END_TIMESTEP]
    if roll_pitch is not None:
        roll_pitch = roll_pitch[START_TIMESTEP:END_TIMESTEP]
    if base_lin_acc is not None:
        base_lin_acc = base_lin_acc[START_TIMESTEP:END_TIMESTEP]
    if base_ang_vel is not None:
        base_ang_vel = base_ang_vel[START_TIMESTEP:END_TIMESTEP]

timesteps = np.linspace(start=0, stop=len(obs_data) * DT, num=len(obs_data))

# Set up output directory
output_dir = os.path.join(os.getcwd(), "play_data_plots")
os.makedirs(output_dir, exist_ok=True)

print(f"Total timesteps: {len(observations)}")
print(f"Observations shape: {obs_data.shape}")
print(f"Expected observation size (from config): {OBS_INDICES['total_obs_size']}")
print(f"Actions shape: {actions_data.shape}")
print(f"Rewards shape: {rewards_data.shape}")
print(f"Output directory: {output_dir}")
print("Observation components enabled:")
for component_name, enabled in OBSERVATION_COMPONENTS.items():
    status = "✓" if enabled else "✗"
    print(f"  {status} {component_name}")
print("Selected figures to plot:")
for figure_name, enabled in PLOT_FIGURES.items():
    status = "ON" if enabled else "OFF"
    print(f"  - {figure_name}: {status}")
print()

# ============================================================================
# Figure 1: Joint Angles and Velocities (4x2 Grid)
# ============================================================================
if PLOT_FIGURES["joint_angles_velocities"]:
    try:
        fig_joints = plt.figure(figsize=(16, 10))

        # Joint names for subplots (in order)
        joint_names = ["Hip AA", "Hip FE", "Knee FE", "Ankle FE"]

        # Upper row: Joint positions (angles)
        for idx, joint_name in enumerate(joint_names):
            ax = plt.subplot(2, 4, idx + 1)
            left_idx = idx * 2
            right_idx = idx * 2 + 1

            ax.plot(timesteps, np.degrees(joint_pos[:, left_idx]), linewidth=2, label="Left", color="blue")
            ax.plot(timesteps, np.degrees(joint_pos[:, right_idx]), linewidth=2, label="Right", color="red")
            ax.set_title(f"{joint_name} Angle", fontweight="bold")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Angle (deg)")
            ax.set_xlim(left=timesteps[0], right=timesteps[-1])
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

        # Lower row: Joint velocities
        for idx, joint_name in enumerate(joint_names):
            ax = plt.subplot(2, 4, idx + 5)
            left_idx = idx * 2
            right_idx = idx * 2 + 1

            ax.plot(timesteps, joint_vel[:, left_idx], linewidth=2, label="Left", color="blue")
            ax.plot(timesteps, joint_vel[:, right_idx], linewidth=2, label="Right", color="red")
            ax.set_title(f"{joint_name} Velocity", fontweight="bold")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Velocity (rad/s)")
            ax.set_xlim(left=timesteps[0], right=timesteps[-1])
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

        plt.tight_layout(pad=0.0)
        joints_path = os.path.join(output_dir, "01_joint_angles_velocities.png")
        plt.savefig(joints_path, dpi=100, bbox_inches="tight")
        print(f"✓ Saved joint angles/velocities plot to {joints_path}")
        plt.close(fig_joints)
    except Exception as e:
        print(f"✗ Error generating joint angles/velocities plot: {e}")
        plt.close("all")

# ============================================================================
# Figure 2: Base Euler Angles (Roll and Pitch)
# ============================================================================
if PLOT_FIGURES["base_euler_angles"]:
    try:
        fig_euler = plt.figure(figsize=(12, 5))

        ax = plt.subplot(1, 1, 1)
        ax.plot(
            timesteps, np.degrees(roll_pitch[:, 0]), linewidth=2, label="Roll", color="red"
        )
        ax.plot(
            timesteps, np.degrees(roll_pitch[:, 1]), linewidth=2, label="Pitch", color="green"
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (degrees)")
        ax.set_title("Base Euler Angles (Roll, Pitch)")
        ax.set_xlim(left=timesteps[0], right=timesteps[-1])
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout(pad=0.0)
        euler_path = os.path.join(output_dir, "02_base_euler_angles.png")
        plt.savefig(euler_path, dpi=100, bbox_inches="tight")
        print(f"✓ Saved euler angles plot to {euler_path}")
        plt.close(fig_euler)
    except Exception as e:
        print(f"✗ Error generating euler angles plot: {e}")
        plt.close("all")

# ============================================================================
# Figure 3: Base Linear Accelerations
# ============================================================================
if PLOT_FIGURES["base_linear_accelerations"]:
    try:
        fig_lin_acc = plt.figure(figsize=(12, 5))

        ax = plt.subplot(1, 1, 1)
        ax.plot(
            timesteps, base_lin_acc[:, 0], linewidth=2, label="X (forward/back)", color="red"
        )
        ax.plot(
            timesteps, base_lin_acc[:, 1], linewidth=2, label="Y (left/right)", color="green"
        )
        ax.plot(timesteps, base_lin_acc[:, 2], linewidth=2, label="Z (up/down)", color="blue")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Linear Acceleration (m/s²)")
        ax.set_title("Base Linear Accelerations")
        ax.set_xlim(left=timesteps[0], right=timesteps[-1])
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout(pad=0.0)
        lin_acc_path = os.path.join(output_dir, "03_base_linear_accelerations.png")
        plt.savefig(lin_acc_path, dpi=100, bbox_inches="tight")
        print(f"✓ Saved linear accelerations plot to {lin_acc_path}")
        plt.close(fig_lin_acc)
    except Exception as e:
        print(f"✗ Error generating linear accelerations plot: {e}")
        plt.close("all")

# ============================================================================
# Figure 4: Base Angular Velocities
# ============================================================================
if PLOT_FIGURES["base_angular_velocities"]:
    try:
        fig_ang_vel = plt.figure(figsize=(12, 5))

        ax = plt.subplot(1, 1, 1)
        ax.plot(timesteps, base_ang_vel[:, 0], linewidth=2, label="Roll rate (X)", color="red")
        ax.plot(
            timesteps, base_ang_vel[:, 1], linewidth=2, label="Pitch rate (Y)", color="green"
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angular Velocity (rad/s)")
        ax.set_title("Base Angular Velocities")
        ax.set_xlim(left=timesteps[0], right=timesteps[-1])
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout(pad=0.0)
        ang_vel_path = os.path.join(output_dir, "04_base_angular_velocities.png")
        plt.savefig(ang_vel_path, dpi=100, bbox_inches="tight")
        print(f"✓ Saved angular velocities plot to {ang_vel_path}")
        plt.close(fig_ang_vel)
    except Exception as e:
        print(f"✗ Error generating angular velocities plot: {e}")
        plt.close("all")

# ============================================================================
# Figure 6: Actions
# ============================================================================
if PLOT_FIGURES["actions"]:
    try:
        fig2 = plt.figure(figsize=(14, 6))

        joint_names = ["Hip AA", "Hip FE", "Knee FE", "Ankle FE"]

        for idx, joint_name in enumerate(joint_names):
            plt.subplot(1, 4, idx + 1)
            left_idx = idx * 2
            right_idx = idx * 2 + 1

            plt.plot(timesteps, actions_data[:, left_idx], linewidth=1.5, label="Left", color="blue")
            plt.plot(timesteps, actions_data[:, right_idx], linewidth=1.5, label="Right", color="red")

            plt.title(f"{joint_name}", fontweight="bold")
            plt.xlabel("Time (s)")
            plt.ylabel(ACTION_UNITS)
            ax_action = plt.gca()
            ax_action.set_xlim(left=timesteps[0], right=timesteps[-1])
            plt.grid(True, alpha=0.3)
            plt.legend(loc="best")

        plt.tight_layout(pad=0.0)
        actions_path = os.path.join(output_dir, "06_actions.png")
        plt.savefig(actions_path, dpi=100, bbox_inches="tight")
        print(f"✓ Saved actions plot to {actions_path}")
        plt.close(fig2)
    except Exception as e:
        print(f"✗ Error generating actions plot: {e}")
        plt.close("all")

# ============================================================================
# Figure 7: Rewards and Episode Termination
# ============================================================================
if PLOT_FIGURES["rewards_dones"]:
    try:
        fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))

        # Rewards
        ax1.plot(timesteps, rewards_data, linewidth=2, color="green", label="Reward")
        ax1.fill_between(timesteps, 0, rewards_data, alpha=0.3, color="green")
        ax1.set_ylabel("Reward Value")
        ax1.set_title("Rewards")
        ax1.set_xlim(left=timesteps[0], right=timesteps[-1])
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        stats_text = f"Min: {rewards_data.min():.4f} | Max: {rewards_data.max():.4f} | Mean: {rewards_data.mean():.4f}"
        ax1.text(
            0.02,
            0.95,
            stats_text,
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Episode termination
        episode_ends = np.where(dones_data > 0)[0]
        ax2.scatter(
            episode_ends,
            np.ones_like(episode_ends),
            s=100,
            color="red",
            marker="x",
            linewidths=3,
            label="Episode Ended",
        )
        ax2.set_ylim([0.5, 1.5])
        ax2.set_xlim([0, timesteps[-1]])
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Status")
        ax2.set_yticks([])
        ax2.set_title(f"Episode Termination Events (Total: {len(episode_ends)} terminations)")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="x")

        plt.tight_layout(pad=0.0)
        rewards_path = os.path.join(output_dir, "07_rewards_dones.png")
        plt.savefig(rewards_path, dpi=100, bbox_inches="tight")
        print(f"✓ Saved rewards/dones plot to {rewards_path}")
        plt.close(fig3)
    except Exception as e:
        print(f"✗ Error generating rewards/dones plot: {e}")
        plt.close("all")

# ============================================================================
# Figure 8: Actual vs Desired Joint Positions
# ============================================================================
if PLOT_FIGURES["actual_vs_desired"]:
    try:
        fig8 = plt.figure(figsize=(16, 6))

        joint_names = ["Hip AA", "Hip FE", "Knee FE", "Ankle FE"]

        for idx, joint_name in enumerate(joint_names):
            ax1 = plt.subplot(1, 4, idx + 1)
            left_idx = idx * 2
            right_idx = idx * 2 + 1

            # Actual joint positions (always on left axis)
            line1 = ax1.plot(timesteps, np.degrees(joint_pos[:, left_idx]), linewidth=2, label="Actual Left", color="blue", linestyle="-")
            line2 = ax1.plot(timesteps, np.degrees(joint_pos[:, right_idx]), linewidth=2, label="Actual Right", color="red", linestyle="-")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Actual Angle (deg)", color="black")
            ax1.tick_params(axis="y", labelcolor="black")
            ax1.set_xlim(left=timesteps[0], right=timesteps[-1])
            ax1.grid(True, alpha=0.3)

            # Desired joint positions/torques
            if TORQUE_MODE_ACTIVE:
                # Use secondary y-axis for torques
                ax2 = ax1.twinx()
                line3 = ax2.plot(timesteps, actions_data[:, left_idx], linewidth=1.5, label="Desired Left (Torque)", color="blue", linestyle="--", alpha=0.7)
                line4 = ax2.plot(timesteps, actions_data[:, right_idx], linewidth=1.5, label="Desired Right (Torque)", color="red", linestyle="--", alpha=0.7)
                ax2.set_ylabel("Desired Torque (Nm)", color="black")
                ax2.tick_params(axis="y", labelcolor="black")
                
                min_angle = np.degrees(joint_pos[:, [left_idx, right_idx]]).min()
                max_angle = np.degrees(joint_pos[:, [left_idx, right_idx]]).max()
                angle_end = max(abs(min_angle), abs(max_angle))
                ax1.set_ylim(-angle_end * 1.2, angle_end * 1.2)
                
                min_torque = actions_data[:, [left_idx, right_idx]].min()
                max_torque = actions_data[:, [left_idx, right_idx]].max()
                torque_end = max(abs(min_torque), abs(max_torque))
                ax2.set_ylim(-torque_end * 1.2, torque_end * 1.2)

                # Combine legends from both axes
                lines = line1 + line2 + line3 + line4
                labels = [line_obj.get_label() for line_obj in lines]
                ax1.legend(lines, labels, loc="best", fontsize=7)
            else:
                # Single axis for angles
                line3 = ax1.plot(timesteps, np.degrees(actions_data[:, left_idx]), linewidth=1.5, label="Desired Left", color="blue", linestyle="--", alpha=0.7)
                line4 = ax1.plot(timesteps, np.degrees(actions_data[:, right_idx]), linewidth=1.5, label="Desired Right", color="red", linestyle="--", alpha=0.7)
                lines = line1 + line2 + line3 + line4
                labels = [line_obj.get_label() for line_obj in lines]
                ax1.legend(lines, labels, loc="best", fontsize=7)

            ax1.set_title(f"{joint_name}", fontweight="bold")

        plt.tight_layout(pad=0.0)
        overlay_path = os.path.join(output_dir, "08_actual_vs_desired_joint_positions.png")
        plt.savefig(overlay_path, dpi=100, bbox_inches="tight")
        print(f"✓ Saved actual vs desired joint positions plot to {overlay_path}")
        plt.close(fig8)
    except Exception as e:
        print(f"✗ Error generating actual vs desired joint positions plot: {e}")
        plt.close("all")

print("All plots generated and saved successfully!")

"""
Plot hardware system dynamics from recorded CSV data.

This script reads hardware recordings and creates visualizations of:
- Joint angles vs desired angles (tracking performance)
- Joint torques vs desired torques (control effort)
- Joint velocities over time
- IMU data (gyroscope, accelerometer)
- Euler angles (body orientation)
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
START_TIME_SEC = 11.0    # Start time in seconds
END_TIME_SEC = 15.0     # End time in seconds


def load_data(csv_path):
    """Load CSV data with clean column names directly."""
    # Define clean column names in the expected order
    column_names = [
        'pitch', 'roll', 'yaw',
        'gyr_x', 'gyr_y', 'gyr_z',
        'acc_x', 'acc_y', 'acc_z',
        # Hip AA (0=L, 4=R)
        'hip_aa_l_torque', 'hip_aa_r_torque',
        'hip_aa_l_torque_des', 'hip_aa_r_torque_des',
        'hip_aa_l_angle', 'hip_aa_r_angle',
        'hip_aa_l_angle_des', 'hip_aa_r_angle_des',
        'hip_aa_l_vel', 'hip_aa_r_vel',
        # Hip FE (1=L, 5=R)
        'hip_fe_l_torque', 'hip_fe_r_torque',
        'hip_fe_l_torque_des', 'hip_fe_r_torque_des',
        'hip_fe_l_angle', 'hip_fe_r_angle',
        'hip_fe_l_angle_des', 'hip_fe_r_angle_des',
        'hip_fe_l_vel', 'hip_fe_r_vel',
        # Knee FE (2=L, 6=R)
        'knee_fe_l_torque', 'knee_fe_r_torque',
        'knee_fe_l_torque_des', 'knee_fe_r_torque_des',
        'knee_fe_l_angle', 'knee_fe_r_angle',
        'knee_fe_l_angle_des', 'knee_fe_r_angle_des',
        'knee_fe_l_vel', 'knee_fe_r_vel',
        # Ankle FE (3=L, 7=R)
        'ankle_fe_l_torque', 'ankle_fe_r_torque',
        'ankle_fe_l_torque_des', 'ankle_fe_r_torque_des',
        'ankle_fe_l_angle', 'ankle_fe_r_angle',
        'ankle_fe_l_angle_des', 'ankle_fe_r_angle_des',
        'ankle_fe_l_vel', 'ankle_fe_r_vel',
    ]

    # Add NN input columns (0 to 42)
    for i in range(43):
        column_names.append(f'nn_input_{i}')

    # Load CSV without header, applying clean names directly
    df = pd.read_csv(csv_path, header=None, names=column_names)

    # Create time column assuming 1kHz sampling rate (0.001s per sample)
    df['time_s'] = df.index * 0.001
    df['roll'] = -df['roll']

    print(f"Data loaded: {df.shape[0]} samples, {len(df.columns)} columns")
    print(f"Time range: {df['time_s'].min():.2f}s to {df['time_s'].max():.2f}s\n")

    return df


def plot_joint_angles(df):
    """Plot joint angles grouped by joint type."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Joint Angles', fontsize=16, fontweight='bold')
    axes = axes.flatten()

    # Define which data to plot for each subplot
    joints_to_plot = [
        ('Hip AA', [('hip_aa_l_angle', 'Left'),
                    ('hip_aa_r_angle', 'Right')]),
        ('Hip FE', [('hip_fe_l_angle', 'Left'),
                    ('hip_fe_r_angle', 'Right')]),
        ('Knee FE', [('knee_fe_l_angle', 'Left'),
                     ('knee_fe_r_angle', 'Right')]),
        ('Ankle FE', [('ankle_fe_l_angle', 'Left'),
                      ('ankle_fe_r_angle', 'Right')]),
    ]

    for idx, (joint_type, joints) in enumerate(joints_to_plot):
        ax = axes[idx]

        for angle_col, side in joints:
            # Convert from radians to degrees
            angle_deg = df[angle_col] * 180 / 3.14159
            ax.plot(df['time_s'], angle_deg, label=f'{side}', linewidth=2, alpha=0.8)

        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Angle (deg)', fontsize=11)
        ax.set_title(f'{joint_type}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_joint_torques(df):
    """Plot joint torques vs desired torques grouped by joint type."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Joint Torques vs Desired Torques', fontsize=16, fontweight='bold')
    axes = axes.flatten()

    # Define which data to plot for each subplot
    joints_to_plot = [
        ('Hip AA', [('hip_aa_l_torque', 'hip_aa_l_torque_des', 'Left'),
                    ('hip_aa_r_torque', 'hip_aa_r_torque_des', 'Right')]),
        ('Hip FE', [('hip_fe_l_torque', 'hip_fe_l_torque_des', 'Left'),
                    ('hip_fe_r_torque', 'hip_fe_r_torque_des', 'Right')]),
        ('Knee FE', [('knee_fe_l_torque', 'knee_fe_l_torque_des', 'Left'),
                     ('knee_fe_r_torque', 'knee_fe_r_torque_des', 'Right')]),
        ('Ankle FE', [('ankle_fe_l_torque', 'ankle_fe_l_torque_des', 'Left'),
                      ('ankle_fe_r_torque', 'ankle_fe_r_torque_des', 'Right')]),
    ]

    for idx, (joint_type, joints) in enumerate(joints_to_plot):
        ax = axes[idx]

        for torque_col, torque_des_col, side in joints:
            ax.plot(df['time_s'], df[torque_col], label=f'{side} actual', linewidth=2, alpha=0.8)
            ax.plot(df['time_s'], df[torque_des_col], '--', label=f'{side} desired', linewidth=2, alpha=0.8)

        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Torque (Nm)', fontsize=11)
        ax.set_title(f'{joint_type}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_joint_velocities(df):
    """Plot joint velocities grouped by joint type."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Joint Velocities', fontsize=16, fontweight='bold')
    axes = axes.flatten()

    # Define which data to plot for each subplot
    joints_to_plot = [
        ('Hip AA', [('hip_aa_l_vel', 'Left'),
                    ('hip_aa_r_vel', 'Right')]),
        ('Hip FE', [('hip_fe_l_vel', 'Left'),
                    ('hip_fe_r_vel', 'Right')]),
        ('Knee FE', [('knee_fe_l_vel', 'Left'),
                     ('knee_fe_r_vel', 'Right')]),
        ('Ankle FE', [('ankle_fe_l_vel', 'Left'),
                      ('ankle_fe_r_vel', 'Right')]),
    ]

    for idx, (joint_type, joints) in enumerate(joints_to_plot):
        ax = axes[idx]

        for vel_col, side in joints:
            ax.plot(df['time_s'], df[vel_col], label=f'{side}', linewidth=2, alpha=0.8)

        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Velocity (rad/s)', fontsize=11)
        ax.set_title(f'{joint_type}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_imu_data(df):
    """Plot IMU data (gyroscope and accelerometer)."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('IMU Data', fontsize=16, fontweight='bold')

    # Gyroscope data
    ax = axes[0]
    ax.plot(df['time_s'], df['gyr_x'], label='X', linewidth=2, alpha=0.8)
    ax.plot(df['time_s'], df['gyr_y'], label='Y', linewidth=2, alpha=0.8)
    ax.plot(df['time_s'], df['gyr_z'], label='Z', linewidth=2, alpha=0.8)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Angular Velocity (rad/s)', fontsize=11)
    ax.set_title('Gyroscope', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Accelerometer data
    ax = axes[1]
    ax.plot(df['time_s'], df['acc_x'], label='X', linewidth=2, alpha=0.8)
    ax.plot(df['time_s'], df['acc_y'], label='Y', linewidth=2, alpha=0.8)
    ax.plot(df['time_s'], df['acc_z'], label='Z', linewidth=2, alpha=0.8)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Acceleration (m/s²)', fontsize=11)
    ax.set_title('Accelerometer', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_euler_angles(df):
    """Plot Euler angles (body orientation)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Body Orientation (Euler Angles)', fontsize=16, fontweight='bold')

    ax.plot(df['time_s'], df['roll'], label='Roll (X)', linewidth=2, alpha=0.8)
    ax.plot(df['time_s'], df['pitch'], label='Pitch (Y)', linewidth=2, alpha=0.8)
    ax.plot(df['time_s'], df['yaw'], label='Yaw (Z)', linewidth=2, alpha=0.8)

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Angle (deg)', fontsize=11)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """Main function to load data and create plots."""
    csv_file = Path(__file__).parent / 'ID06_HardwareTest_HigherSaturation.csv'

    if not csv_file.exists():
        print(f"Error: CSV file not found at {csv_file}")
        return

    print(f"Loading data from {csv_file}...")
    df = load_data(csv_file)

    # Filter data to the specified time range
    print(f"Filtering data: {START_TIME_SEC}s to {END_TIME_SEC}s...")
    df_filtered = df[(df['time_s'] >= START_TIME_SEC) & (df['time_s'] <= END_TIME_SEC)].copy()

    # Reset time axis to start from 0
    df_filtered['time_s'] = df_filtered['time_s'] - START_TIME_SEC

    print(f"Filtered data: {df_filtered.shape[0]} samples")
    print(f"Time range: {df_filtered['time_s'].min():.2f}s to {df_filtered['time_s'].max():.2f}s\n")

    # Create output directory for plots
    output_dir = Path(__file__).parent / 'hardware_data_plots'
    output_dir.mkdir(exist_ok=True)
    print(f"Saving plots to {output_dir}\n")

    # Create all plots
    print("Creating plots...")
    fig1 = plot_joint_angles(df_filtered)
    fig2 = plot_joint_torques(df_filtered)
    fig3 = plot_joint_velocities(df_filtered)
    fig4 = plot_imu_data(df_filtered)
    fig5 = plot_euler_angles(df_filtered)

    # Save all plots in PNG and SVG formats
    plots = [
        (fig1, 'joint_angles'),
        (fig2, 'joint_torques'),
        (fig3, 'joint_velocities'),
        (fig4, 'imu_data'),
        (fig5, 'euler_angles'),
    ]

    for fig, name in plots:
        png_path = output_dir / f'{name}.png'
        svg_path = output_dir / f'{name}.svg'
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        fig.savefig(svg_path, bbox_inches='tight')
        print(f"Saved {name}")

    print("\nDone! Opening plots...")
    plt.show()


if __name__ == '__main__':
    main()

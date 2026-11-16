import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load CSV file
# csv_path = "StandstillMeasurement_forNoise.csv"
csv_path = "2026-02-12-Standstill.csv"
df = pd.read_csv(csv_path)
df = df.apply(pd.to_numeric, errors='coerce')
df = df.head(5000)

# Convert time from ms to s
df['Time (ms)'] = df['Time (ms)'] / 1000
df = df.rename(columns={'Time (ms)': 'Time (s)'})

# Exclude channels that are not relevant for noise analysis
time_data = df['Time (s)'].values
print(time_data)
channels = df.columns.tolist()
channels.remove('Time (s)')

# Filter out unwanted channels by prefix
excluded_prefixes = ['eul_', 'quat_', 'GyrHR', 'AccHR', 'A_JointTorqueDesired_Nm', 'A_desiredJointAngles', 'A_jointAngleErrors', 'Mag']
channels = [ch for ch in channels if not any(ch.startswith(prefix) for prefix in excluded_prefixes)]

# Sort channels alphabetically
channels.sort()

# Calculate mean and noise amplitude (standard deviation) for each channel
print("=" * 80)
print("NOISE ANALYSIS RESULTS")
print("=" * 80)
print(f"Number of samples: {len(df)}")
print(f"Duration: {time_data[0]:.3f} s to {time_data[-1]:.3f} s")
print("=" * 80)
print()

stats = {}
for channel in channels:
    data = df[channel].values
    mean = np.mean(data)
    noise_amplitude = np.std(data)
    stats[channel] = {'mean': mean, 'noise_amplitude': noise_amplitude}
    print(f"{channel:30s} | Mean: {mean:+14.6e} | Noise Amplitude (sigma): {noise_amplitude:+14.6e}")

print()
print("=" * 80)

# Create subplots - one for each channel
num_channels = len(channels)
fig, axes = plt.subplots(num_channels, 1, figsize=(12, 3 * num_channels))

# Handle case where there's only one subplot
if num_channels == 1:
    axes = [axes]

for idx, channel in enumerate(channels):
    ax = axes[idx]
    data = df[channel].values
    mean = stats[channel]['mean']
    noise_amplitude = stats[channel]['noise_amplitude']

    # Plot data
    ax.plot(time_data, data, linewidth=0.7, label='Data')

    # Plot mean line
    ax.axhline(y=mean, color='r', linestyle='--', linewidth=1.5, label=f'Mean: {mean:.6e}')

    # Plot noise amplitude band (±1σ)
    ax.axhline(y=mean + noise_amplitude, color='g', linestyle=':', linewidth=1, alpha=0.7, label=f'±σ: ±{noise_amplitude:.6e}')
    ax.axhline(y=mean - noise_amplitude, color='g', linestyle=':', linewidth=1, alpha=0.7)

    # Fill region between ±σ
    ax.fill_between(time_data, mean - noise_amplitude, mean + noise_amplitude, alpha=0.1, color='green')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(channel)
    ax.set_title(f'{channel} (Mean: {mean:.6e}, Noise Amplitude (sigma): {noise_amplitude:.6e})')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('noise_analysis_plots.png', dpi=150, bbox_inches='tight')
plt.close()

print("Analysis complete!")

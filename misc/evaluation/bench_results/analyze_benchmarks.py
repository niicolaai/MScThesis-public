import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Define CSV files to analyze
csv_files = [
    'envs_1.csv',
    'envs_10.csv',
    'envs_100.csv',
    'envs_1000.csv',
    'envs_5000.csv',
    'envs_10000.csv',
    'envs_20000.csv',
    'envs_30000.csv',
]


# Extract number of environments from filename
def extract_env_count(filename):
    return int(filename.replace('envs_', '').replace('.csv', ''))


# Load and process data
data = []
for csv_file in csv_files:
    filepath = Path(csv_file)
    if filepath.exists():
        df = pd.read_csv(filepath)
        env_count = extract_env_count(csv_file)

        # Get maximum values (skip the initial ramp-up, use max)
        max_gpu_util = df['gpu_util'].max()
        max_gpu_mem = df['gpu_mem_used_mb'].max()
        max_cpu = df['cpu_percent'].max()
        max_ram = df['ram_used_mb'].max()

        # Get static info (these shouldn't change during run)
        gpu_mem_total = df['gpu_mem_total_mb'].iloc[0]
        ram_total = df['ram_total_mb'].iloc[0]

        data.append({
            'env_count': env_count,
            'gpu_util': max_gpu_util,
            'gpu_mem_used': max_gpu_mem,
            'gpu_mem_total': gpu_mem_total,
            'cpu_percent': max_cpu,
            'ram_used': max_ram,
            'ram_total': ram_total,
        })

        print(f"✓ Loaded {csv_file}: {env_count} envs")
    else:
        print(f"✗ File not found: {csv_file}")

results_df = pd.DataFrame(data)
results_df = results_df.sort_values('env_count')

print("\n" + "=" * 80)
print("BENCHMARK RESULTS - MAXIMUM VALUES")
print("=" * 80)
print(results_df.to_string(index=False))
print("=" * 80)

# Calculate GPU memory percentage for reference
results_df['gpu_mem_percent'] = (results_df['gpu_mem_used'] / results_df['gpu_mem_total']) * 100
results_df['ram_percent'] = (results_df['ram_used'] / results_df['ram_total']) * 100


env_counts = results_df['env_count'].values
limits = {}

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('GPU/CPU/RAM Benchmark Analysis - Maximum Values per Environment Count',
             fontsize=14, fontweight='bold')

# Plot 1: GPU Utilization
ax = axes[0, 0]
ax.plot(env_counts, results_df['gpu_util'].values, 'o-', linewidth=2.5, markersize=8, label='GPU Util', color='#FF6B6B')
ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='100% limit')
ax.set_xlabel('Number of Environments', fontsize=11, fontweight='bold')
ax.set_ylabel('GPU Utilization (%)', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.set_xscale('log')

# Plot 2: GPU Memory
ax = axes[0, 1]
ax.plot(env_counts, results_df['gpu_mem_used'].values, 'o-', linewidth=2.5, markersize=8, label='GPU Memory Used', color='#4ECDC4')
ax.axhline(y=results_df['gpu_mem_total'].max(), color='red', linestyle='--', alpha=0.5, label=f'Max: {results_df["gpu_mem_total"].max():.0f} MB')
ax.set_xlabel('Number of Environments', fontsize=11, fontweight='bold')
ax.set_ylabel('GPU Memory Used (MB)', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.set_xscale('log')

# Plot 3: CPU Usage
ax = axes[1, 0]
ax.plot(env_counts, results_df['cpu_percent'].values, 'o-', linewidth=2.5, markersize=8, label='CPU Usage', color='#FFD93D')
ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='100% limit')
ax.set_xlabel('Number of Environments', fontsize=11, fontweight='bold')
ax.set_ylabel('CPU Usage (%)', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.set_xscale('log')

# Plot 4: RAM Usage
ax = axes[1, 1]
ax.plot(env_counts, results_df['ram_used'].values, 'o-', linewidth=2.5, markersize=8, label='RAM Used', color='#95E1D3')
ax.axhline(y=results_df['ram_total'].max(), color='red', linestyle='--', alpha=0.5, label=f'Max: {results_df["ram_total"].max():.0f} MB')
ax.set_xlabel('Number of Environments', fontsize=11, fontweight='bold')
ax.set_ylabel('RAM Used (MB)', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.set_xscale('log')

plt.tight_layout()
plt.savefig('benchmark_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: benchmark_analysis.png")

# Create a summary comparison plot
fig, ax = plt.subplots(figsize=(12, 6))

metrics = ['GPU Util (%)', 'GPU Mem (%)', 'CPU (%)', 'RAM (%)']
x = np.arange(len(results_df))
width = 0.2

bars1 = ax.bar(x - 1.5 * width, results_df['gpu_util'], width, label='GPU Util (%)', color='#FF6B6B')
bars2 = ax.bar(x - 0.5 * width, results_df['gpu_mem_percent'], width, label='GPU Mem (%)', color='#4ECDC4')
bars3 = ax.bar(x + 0.5 * width, results_df['cpu_percent'], width, label='CPU (%)', color='#FFD93D')
bars4 = ax.bar(x + 1.5 * width, results_df['ram_percent'], width, label='RAM (%)', color='#95E1D3')

ax.set_ylabel('Usage (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Number of Environments', fontsize=12, fontweight='bold')
ax.set_title('Resource Usage Comparison - Maximum Values', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{int(e):,}' for e in results_df['env_count']])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=100, color='red', linestyle='--', alpha=0.3, linewidth=1.5, label='Saturation (100%)')
ax.set_ylim(0, 110)

plt.tight_layout()
plt.savefig('benchmark_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: benchmark_comparison.png")

print("\nDone! Check the generated PNG files.")

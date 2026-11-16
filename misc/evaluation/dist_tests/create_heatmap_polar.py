#!/usr/bin/env python3
"""
Create a heatmap polar plot showing success rate across force magnitudes and directions.

For each direction and force magnitude bin (5N bins):
- Calculate the success rate (% of True values)
- Create a PNG polar heatmap visualization
- Export CSV data for pgfplots
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.cm as cm
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = Path(os.getcwd())
print(f"Base directory: {BASE_DIR}")
MEASUREMENT_TYPE = "Torque"
BIN_SIZE = 5  # N
MAX_FORCE = 500  # N
# ============================================================================

INPUT_DIR = BASE_DIR / MEASUREMENT_TYPE
OUTPUT_PNG = BASE_DIR / f"{MEASUREMENT_TYPE}_heatmap_polar.png"

def extract_direction(filename: str) -> float:
    """Extract direction from filename"""
    import re
    match = re.search(r"direction_([\d.]+)deg\.csv", filename)
    if match:
        return float(match.group(1))
    raise ValueError(f"Could not extract direction from filename: {filename}")

def process_csv_file(csv_file: Path) -> tuple:
    """
    Read CSV file and return (direction, data_dict)
    where data_dict maps force_bin to (total_count, true_count)
    """
    direction = extract_direction(csv_file.name)
    data = {}
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            force_mag = float(row['force_magnitude_N'])
            survived = row['survived'].strip().lower() == 'true'
            
            # Determine bin
            bin_idx = int(force_mag // BIN_SIZE)
            if bin_idx not in data:
                data[bin_idx] = [0, 0]  # [total, true_count]
            
            data[bin_idx][0] += 1
            if survived:
                data[bin_idx][1] += 1
    
    return direction, data

def main():
    if not INPUT_DIR.exists():
        print(f"Error: Input directory does not exist: {INPUT_DIR}")
        return
    
    # Find all matching CSV files
    csv_files = sorted(INPUT_DIR.glob("disturbance_tests_direction_*.csv"))
    
    if not csv_files:
        print(f"Warning: No CSV files found in {INPUT_DIR}")
        return
    
    print(f"Found {len(csv_files)} CSV file(s)")
    
    # Process all files
    all_data = {}
    directions = []
    
    for csv_file in csv_files:
        try:
            direction, data = process_csv_file(csv_file)
            directions.append(direction)
            all_data[direction] = data
            print(f"  Processed {csv_file.name}")
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
    
    directions.sort()
    
    # Add 360 degrees as duplicate of 0 degrees to complete the circle
    if 0.0 in all_data:
        all_data[360.0] = all_data[0.0]
        directions.append(360.0)
        print(f"  Added 360.0° as duplicate of 0.0° to complete the circle")
    
    # Create polar heatmap PNG
    create_polar_heatmap(directions, all_data)

def create_polar_heatmap(directions, all_data):
    """Create and save polar heatmap as PNG"""
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.family'] = 'serif'
    
    # Create bins array
    num_bins = int(np.ceil(MAX_FORCE / BIN_SIZE))
    force_bins = np.arange(0, MAX_FORCE + BIN_SIZE, BIN_SIZE)
    
    # Prepare data grid
    theta = np.array(directions) * np.pi / 180  # Convert to radians
    r = force_bins[:-1] + BIN_SIZE / 2  # Bin centers
    THETA, R = np.meshgrid(theta, r)
    
    # Create success rate matrix
    Z = np.zeros_like(THETA, dtype=float)
    for i, direction in enumerate(directions):
        data = all_data[direction]
        for bin_idx in range(num_bins):
            if bin_idx in data:
                total, true_count = data[bin_idx]
                success_rate = true_count / total if total > 0 else 0
            else:
                success_rate = np.nan  # No data for this bin
            Z[bin_idx, i] = success_rate
    
    # Create polar plot
    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw=dict(projection='polar'))
    
    # Rotate so 0 degrees is at the bottom and angles increase clockwise
    ax.set_theta_zero_location('S')  # S = South (bottom)
    ax.set_theta_direction(1)       # 1 = counterclockwise
    ax.set_rlabel_position(250)
    
    # Plot with color mesh
    contourf = ax.contourf(THETA, R, Z, levels=20, cmap='RdYlGn', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = fig.colorbar(contourf, ax=ax, pad=0.1, label='Rescue Success Rate')
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # Labels and title
    ax.set_xlabel(r'$\angle [F_{\mathrm{pert}} ]$ ($\text{deg}$)')
    ax.set_ylabel(r'$|F_{\mathrm{pert}}|$ (N)', labelpad=-30)  
      
    # Set radial limit
    ax.set_ylim(0, MAX_FORCE)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_PNG.with_suffix('.svg'), format='svg', bbox_inches='tight')
    print(f"Saved polar heatmap to {OUTPUT_PNG} and {OUTPUT_PNG.with_suffix('.svg')}")
    plt.close()


if __name__ == "__main__":
    main()

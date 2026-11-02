#!/usr/bin/env python3
"""
Merge disturbance test CSV files into a single polar plot CSV.

Scans a subdirectory for disturbance test files with format:
    disturbance_tests_direction_X.Xdeg.csv

For each file, extracts:
    - Direction (from filename)
    - First force magnitude where survived == False
    - Last force magnitude where survived == True

If no False values exist, first_false is 0.
If no True values exist, last_true is 0.

Output format:
    direction_deg,first_false_force_magnitude_N,last_true_force_magnitude_N
"""

import csv
import re
from pathlib import Path

# ============================================================================
# CONFIGURATION - Modify these variables as needed
# ============================================================================
BASE_DIR = Path("/home/nicolai/Documents/exoWE2/exoRL/evaluation/dist_tests")
MEASUREMENT_TYPE = "Torque"  # Change to "PD" or other types as needed
# ============================================================================

INPUT_DIR = BASE_DIR / MEASUREMENT_TYPE
OUTPUT_FILE = BASE_DIR / f"4_{MEASUREMENT_TYPE}_Disturbance_Test.csv"

def extract_direction(filename: str) -> float:
    """Extract direction from filename like 'disturbance_tests_direction_10.5deg.csv'"""
    match = re.search(r"direction_([\d.]+)deg\.csv", filename)
    if match:
        return float(match.group(1))
    raise ValueError(f"Could not extract direction from filename: {filename}")

def find_first_false_and_last_true(csv_file: Path) -> tuple:
    """
    Read CSV file and find:
    - First force_magnitude where survived == False
    - Last force_magnitude where survived == True
    
    Returns (first_false_force, last_true_force).
    If not found, returns 0 for that value.
    """
    first_false = 0
    last_true = 0
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            force_mag = float(row['force_magnitude_N'])
            survived = row['survived'].strip().lower() == 'true'
            
            if not survived and first_false == 0:
                first_false = force_mag
            
            if survived:
                last_true = force_mag
    
    return first_false, last_true

def main():
    if not INPUT_DIR.exists():
        print(f"Error: Input directory does not exist: {INPUT_DIR}")
        return
    
    # Find all matching CSV files
    csv_files = sorted(INPUT_DIR.glob("disturbance_tests_direction_*.csv"))
    
    if not csv_files:
        print(f"Warning: No CSV files found in {INPUT_DIR}")
        return
    
    print(f"Found {len(csv_files)} CSV file(s) in {INPUT_DIR}")
    
    # Process each file
    results = []
    for csv_file in csv_files:
        try:
            direction = extract_direction(csv_file.name)
            first_false, last_true = find_first_false_and_last_true(csv_file)
            results.append((direction, first_false, last_true))
            print(f"  {csv_file.name}: direction={direction}°, first_false={first_false}N, last_true={last_true}N")
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
            
    # Add a row which is the duplicate of the first row but with direction 360 to close the polar plot
    if results:
        first_row = results[0]
        results.append((360.0, first_row[1], first_row[2]))
    
    # Sort by direction
    results.sort(key=lambda x: x[0])
    
    # Write output CSV
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['direction_deg', 'first_false_force_magnitude_N', 'last_true_force_magnitude_N'])
        for direction, first_false, last_true in results:
            writer.writerow([direction, first_false, last_true])
    
    print(f"\nSuccessfully wrote {len(results)} rows to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

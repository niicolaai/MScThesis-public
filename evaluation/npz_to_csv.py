import numpy as np
import pandas as pd
import sys

def npz_to_csv(npz_file, output_csv, max_datapoints=1000):
    """
    Convert NPZ file to CSV, keeping only the first max_datapoints.
    
    Args:
        npz_file: Path to the .npz file
        output_csv: Path to the output .csv file
        max_datapoints: Number of datapoints to keep (default 1000)
    """
    # Load the NPZ file
    data = np.load(npz_file)
    
    print(f"NPZ file contains the following arrays:")
    for key in data.files:
        print(f"  - {key}: shape {data[key].shape}")
    
    # If multiple arrays, combine them
    if len(data.files) == 1:
        # Single array case
        array = data[data.files[0]]
    else:
        # Multiple arrays - stack them horizontally
        arrays = [data[key] for key in sorted(data.files)]
        array = np.column_stack(arrays)
    
    # Take only first max_datapoints
    if len(array) > max_datapoints:
        array = array[:max_datapoints]
        print(f"\nTrimmed to first {max_datapoints} datapoints")
    else:
        print(f"\nFile has {len(array)} datapoints (less than {max_datapoints})")
    
    # Convert to DataFrame and save to CSV
    if len(array.shape) == 1:
        df = pd.DataFrame(array)
    else:
        df = pd.DataFrame(array)
    
    df.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")
    print(f"CSV shape: {df.shape}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python npz_to_csv.py <input.npz> [output.csv] [max_datapoints]")
        print("Example: python npz_to_csv.py play_data.npz play_data.csv 1000")
        sys.exit(1)
    
    npz_file = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else npz_file.replace('.npz', '.csv')
    max_datapoints = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    
    npz_to_csv(npz_file, output_csv, max_datapoints)

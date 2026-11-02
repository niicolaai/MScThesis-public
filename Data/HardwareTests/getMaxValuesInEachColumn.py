import pandas as pd
from pathlib import Path

# Get all CSV files in the current directory
current_dir = Path.cwd()
csv_files = sorted(current_dir.glob('*.csv'))

if not csv_files:
    print("No CSV files found in the current directory.")
else:
    for csv_file in csv_files:
        print(f"\n{'='*60}")
        print(f"File: {csv_file.name}")
        print(f"{'='*60}")
        
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Get maximum values for each column
        max_values = df.max()
        
        # Print results
        for column, max_val in max_values.items():
            print(f"{column}: {max_val}")

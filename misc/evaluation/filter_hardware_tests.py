import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('28-04-HardwareTests.csv')

# Filter rows where A_state is either 40 or 50
filtered_df = df[df['A_state'].isin([40, 50])]

# Select only columns starting with 'A_JointTorqueDesired_Nm' or 'Name' or 'A_state'
torque_columns = [col for col in filtered_df.columns if col.startswith('A_JointTorqueDesired_Nm') or col == 'Name' or col == 'A_state']
result_df = filtered_df[torque_columns].copy()

# Rename 'Name' to 'time' and process time values
result_df.rename(columns={'Name': 'time'}, inplace=True)
first_time_value = result_df['time'].iloc[0]
result_df['time'] = (result_df['time'] - first_time_value) / 1000

# Limit to first 1200 points
result_df = result_df.iloc[:1200]

# Export to a new CSV file
output_file = '28-04-HardwareTests_filtered.csv'
result_df.to_csv(output_file, index=False)

print(f"Filtered data exported to {output_file}")
print(f"Number of rows: {len(result_df)}")
print(f"Columns: {list(result_df.columns)}")

# Plot every column
num_cols = len(result_df.columns)
fig, axes = plt.subplots(num_cols, 1, figsize=(12, 4 * num_cols))

# Handle case with single subplot
if num_cols == 1:
    axes = [axes]

for idx, col in enumerate(result_df.columns):
    axes[idx].plot(result_df[col])
    axes[idx].set_title(col)
    axes[idx].set_xlabel('Index')
    axes[idx].set_ylabel('Value')
    axes[idx].grid(True)

plt.tight_layout()
plt.show()

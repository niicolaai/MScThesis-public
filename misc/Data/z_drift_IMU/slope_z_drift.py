import numpy as np
import matplotlib.pyplot as plt

# ==== Load data ====
filename = "GyroZEulDrift.csv"

raw_data = np.genfromtxt(
    filename, delimiter=",", skip_header=1, dtype=float
)

time_ms = raw_data[:, 0]
euler_z = raw_data[:, 1] % 360  # Wrap angles to [0, 360]

# Convert time to seconds
time_s = time_ms / 1000.0

# ==== Linear fit ====
# Fit first-order polynomial (linear fit)
slope, intercept = np.polyfit(time_s, euler_z, 1)

print(f"Slope: {slope}")
print(f"Intercept: {intercept}")

# Optional: plot fit line as well
plt.figure()
plt.plot(time_s, euler_z, label="Data")
plt.plot(time_s, slope * time_s + intercept, label="Linear Fit")
plt.xlabel("Time (s)")
plt.ylabel("Euler Z Angle (deg)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("euler_z_linear_fit.png", dpi=150, bbox_inches="tight")
plt.savefig("euler_z_linear_fit.svg", format="svg", bbox_inches="tight")
plt.show()

# Slope: 0.3543224677331786
# Intercept: 135.71014503020203

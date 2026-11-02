import numpy as np
import matplotlib.pyplot as plt

# Knee joint limits
s_min = 0.0    # Full extension
s_max = 2.1    # Flexion limit

# Fixed parameters
delta_a = 1.0
delta_b = 10.0


def h_exp(s, s_min, s_max, delta_a, delta_b):
    """
    Compute the exponential control barrier function.

    h_exp(s, δ) = δ_a * [1 - (e^(δ_b(s - s_max)) + e^(δ_b(s_min - s)))]
    """
    term1 = np.exp(delta_b * (s - s_max))
    term2 = np.exp(delta_b * (s_min - s))
    h = delta_a * (1.0 - (term1 + term2))
    return h


# Create state range: extend beyond bounds for visualization
range_margin = 0.3
s = np.linspace(s_min - range_margin, s_max + range_margin, 300)

# Compute CBF values
h_vals = h_exp(s, s_min, s_max, delta_a, delta_b)

# Convert s to degrees for plotting
s_deg = np.degrees(s)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

plt.rcParams['text.usetex'] = False
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.family'] = 'serif'

# Plot CBF
ax.plot(s_deg, h_vals, linewidth=2.5, color='blue', label=r'$h(s)$')

# Mark the safe region
ax.axvspan(np.degrees(s_min), np.degrees(s_max), alpha=0.1, color='green', label='Safe region')

# Mark boundaries
ax.axvline(np.degrees(s_min), color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Bounds')
ax.axvline(np.degrees(s_max), color='red', linestyle='--', linewidth=1.5, alpha=0.7)

# Mark zero crossing (barrier boundary)
ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

ax.set_xlabel(r'$q_\text{knee}\,(\text{deg})$', fontsize=12)
ax.set_ylabel(r'$h(s)$', fontsize=12)
ax.set_title(f'Exponential Control Barrier Function for Knee Joint',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='best')

# Save figure
filename = 'cbf_knee_simple.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.savefig(filename.replace('.png', '.svg'), format='svg', bbox_inches='tight')  # Save SVG version
print(f"Saved: {filename}")
plt.show()

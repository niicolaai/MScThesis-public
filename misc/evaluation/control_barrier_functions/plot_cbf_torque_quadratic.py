import numpy as np
import matplotlib.pyplot as plt

# Torque limits
s_min = -100.0  # Nm
s_max = 100.0   # Nm


def h_quadratic(s, s_min, s_max):
    """
    Compute the quadratic control barrier function.

    h_quad(s) = 1 - ((s - center) / half_width)^2
    
    This is positive inside [s_min, s_max] and negative outside.
    """
    center = (s_min + s_max) / 2.0
    half_width = (s_max - s_min) / 2.0
    h = 1.0 - ((s - center) / half_width) ** 2
    return h


# Create state range: extend beyond bounds for visualization
range_margin = 30.0
s = np.linspace(s_min - range_margin, s_max + range_margin, 300)

# Compute CBF values
h_vals = h_quadratic(s, s_min, s_max)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

plt.rcParams['text.usetex'] = False
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.family'] = 'serif'

# Plot CBF
ax.plot(s, h_vals, linewidth=2.5, color='purple', label=r'$h(s)$')

# Mark the safe region
ax.axvspan(s_min, s_max, alpha=0.1, color='green', label='Safe region')

# Mark boundaries
ax.axvline(s_min, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Bounds')
ax.axvline(s_max, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

# Mark zero crossing (barrier boundary)
ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

ax.set_xlabel(r'$\tau\,(\text{Nm})$', fontsize=12)
ax.set_ylabel(r'$h(s)$', fontsize=12)
ax.set_title(f'Quadratic Control Barrier Function for Torque',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='best')

# Save figure
filename = 'cbf_torque_quadratic.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.savefig(filename.replace('.png', '.svg'), format='svg', bbox_inches='tight')  # Save SVG version
print(f"Saved: {filename}")
plt.show()

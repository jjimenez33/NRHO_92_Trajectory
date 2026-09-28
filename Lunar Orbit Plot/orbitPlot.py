import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import spiceypy as spice

# Load SPICE kernels
spice.furnsh("SPICE_Data/de440s.bsp")
spice.furnsh("SPICE_Data/naif0012.tls")
print("SPICE Kernels successfully loaded.")

# Define time array (1 year of daily steps)
start_utc = "2026-01-01T00:00:00"
et_start = spice.utc2et(start_utc)
end_utc = "2027-01-01T00:00:00"
et_end = spice.utc2et(end_utc)
et_times = np.linspace(et_start, et_end, 365)

# Retrieve positions for Moon relative to the Earth in J2000
moon_positions = []
for et in et_times:
  pos, _ = spice.spkpos("Moon", et, "J2000", "NONE", "EARTH")
  moon_positions.append(pos)
moon_positions = np.array(moon_positions)

# Set up 3D figure
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

# Show the entire path permanently using alpha for transparency
ax.plot(
    moon_positions[:, 0],
    moon_positions[:, 1],
    moon_positions[:, 2],
    color="gray",
    linestyle="-",
    alpha=0.5,
    label="Full 1-Year Path",
)

# Plot Earth at origin
ax.scatter(0, 0, 0, color="blue", s=100, label="Earth")

# Drawn on top of the static plot
(trail,) = ax.plot([], [], [], label="Current Progress", color="grey", linewidth=2)
(moon_point,) = ax.plot([], [], [], "ro", label="Moon", markersize=8)

# Set axis limits based on data range (+ a small buffer)
max_limit = np.max(np.abs(moon_positions)) * 1.1
ax.set_xlim([-max_limit, max_limit])
ax.set_ylim([-max_limit, max_limit])
ax.set_zlim([-max_limit, max_limit])

ax.set_xlabel("X (KM)")
ax.set_ylabel("Y (KM)")
ax.set_zlabel("Z (KM)")
ax.set_title("Lunar Orbit: Static Path & Live Animation")
ax.legend()


# Initialization function
def init():
  trail.set_data([], [])
  trail.set_3d_properties([])
  moon_point.set_data([], [])
  moon_point.set_3d_properties([])
  return (
      trail,
      moon_point,
  )  

# Animation update function
def update(frame):
  # Update moving trail up to current frame
  trail.set_data(moon_positions[:frame, 0], moon_positions[:frame, 1])
  trail.set_3d_properties(moon_positions[:frame, 2])

  # Update current position of the Moon
  moon_point.set_data([moon_positions[frame, 0]], [moon_positions[frame, 1]])
  moon_point.set_3d_properties([moon_positions[frame, 2]])
  return trail, moon_point


# Create the animation
ani = animation.FuncAnimation(
    fig, update, frames=len(et_times), init_func=init, interval=50, blit=False
)

plt.show()

# Clean up kernels
spice.kclear()

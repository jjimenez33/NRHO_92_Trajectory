import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import spiceypy as spice

def kernel2state(start_utc, end_utc, step_size):
    """
    Input: 
        string start_utc : start time in ISO
            ex) "2026-01-01T00:00:00"
        string end_utc   : end time in ISO
        double step_size  : deltaT
    Returns:
        nx5 array lunarState : [double x (km), double y (km), double z (km), double MET (s), string UTC]
    """

    # Load SPICE kernels
    spice.furnsh("SPICE_Data/de440s.bsp")
    spice.furnsh("SPICE_Data/naif0012.tls")
    print("SPICE Kernels successfully loaded.")

    # Define time array 
    et_start = spice.utc2et(start_utc)
    et_end = spice.utc2et(end_utc)
    et_array = np.arange(et_start, et_end, step_size)

    # Initialize a lsit to hold state rows
    state_data = []

    for et in et_array:
        # 1. Calculate Mission Elapsed Time (MET) in seconds
        met = et - et_start

        # 2. Get position vector (x, y, z) in (km)
        position, lt = spice.spkpos("MOON", et, "J2000", "NONE", "EARTH")

        #3. Convert current ET back to an ISO UTC string for output
        utc_str = spice.et2utc(et, "ISOC", 3)

        # Combine data
        # [x, y, z, MET, UTC]
        row = [position[0], position[1], position[2], met, utc_str]
        state_data.append(row)

    # Unload kernels to clean up memory
    spice.kclear()

    # Return as a numpy object array to support mixed types (floats and strings)
    return np.array(state_data, dtype=object)

# Test
start_utc = "2026-01-01T00:00:00"
end_utc = "2027-01-01T00:00:00"
step_size = 43200 # step size[s] = 0.5 day
state = kernel2state(start_utc, end_utc, step_size)

# Extract from state
pos = state[:, 0:3].astype(float)
met = state[:, 3].astype(float) # Seconds
utc = state[:, 4] # ISO strings

# Set up 3D figure
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

# Plot Earth at origin
ax.scatter(0, 0, 0, color="blue", s=100, label="Earth")

# Draw on top of static plot
(trail,) = ax.plot([], [], [], label="Current Progress", color="grey", linewidth=1)
(moon_point,) = ax.plot([], [], [], color='grey', marker='o', label="Moon", markersize=8)

# Set axis limits based on data range (+ a small buffer)
max_limit = np.max(np.abs(pos[:,0:2])) * 1.1
ax.set_xlim([-max_limit, max_limit])
ax.set_ylim([-max_limit, max_limit])
ax.set_zlim([-max_limit, max_limit])

ax.set_xlabel("X (KM)")
ax.set_ylabel("Y (KM)")
ax.set_zlabel("Z (KM)")
ax.set_title("Lunar Orbit")
ax.legend()

# Display MET and UTC
met_text = fig.text(0.02, 0.95, "", fontsize=11, family="monospace",
                    va="top", ha="left")

# Initialization function
def init():
  trail.set_data([], [])
  trail.set_3d_properties([])
  moon_point.set_data([], [])
  moon_point.set_3d_properties([])
  met_text.set_text("")
  return (
      trail,
      moon_point,
      met_text
  )  

# Animation update function
def update(frame):
  # Update moving trail up to current frame
  trail.set_data(pos[:frame + 1, 0], pos[:frame + 1, 1])
  trail.set_3d_properties(pos[:frame + 1, 2])

  # Update current position of the Moon
  moon_point.set_data([pos[frame, 0]], [pos[frame, 1]])
  moon_point.set_3d_properties([pos[frame, 2]])

  days = met[frame] / 86400
  met_text.set_text(f"MET: {days:6.1f} d\n"
                      f"     {met[frame]:,.0f} s\n"
                      f"UTC: {utc[frame][:10]}")

  return trail, moon_point, met_text


# Create the animation
ani = animation.FuncAnimation(
    fig, update, frames=len(pos), init_func=init, interval=50, blit=False
)

plt.show()


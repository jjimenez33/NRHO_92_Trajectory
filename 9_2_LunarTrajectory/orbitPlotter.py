import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt


def animate_trajectory(state, interval=50, title="Lunar Orbit"):
    """
    Build the 3D animated trajectory plot.

    Input:
        state    : n x 5 array [x (km), y (km), z (km), MET (s), UTC string]
        interval : delay between frames in ms (bigger = slower)
        title    : plot title
    Returns:
        ani : the FuncAnimation object (must be kept alive by the caller)
    """
    # Extract from state
    pos = state[:, 0:3].astype(float)
    met = state[:, 3].astype(float)  # seconds
    utc = state[:, 4]                # ISO strings

    # Set up 3D figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Earth at origin
    ax.scatter(0, 0, 0, color="blue", s=100, label="Earth")

    # Artists updated by the animation
    (trail,) = ax.plot([], [], [], label="Current Progress", color="grey", linewidth=1)
    (moon_point,) = ax.plot([], [], [], color="grey", marker="o", label="Moon", markersize=8)

    # Axis limits
    max_limit = np.max(np.abs(pos)) * 1.1
    ax.set_xlim([-max_limit, max_limit])
    ax.set_ylim([-max_limit, max_limit])
    ax.set_zlim([-max_limit, max_limit])

    ax.set_xlabel("X (KM)")
    ax.set_ylabel("Y (KM)")
    ax.set_zlabel("Z (KM)")
    ax.set_title(title)
    ax.legend()

    # MET / UTC readout
    met_text = fig.text(0.02, 0.95, "", fontsize=11, family="monospace",
                        va="top", ha="left")

    def init():
        trail.set_data([], [])
        trail.set_3d_properties([])
        moon_point.set_data([], [])
        moon_point.set_3d_properties([])
        met_text.set_text("")
        return trail, moon_point, met_text

    def update(frame):
        trail.set_data(pos[:frame + 1, 0], pos[:frame + 1, 1])
        trail.set_3d_properties(pos[:frame + 1, 2])

        moon_point.set_data([pos[frame, 0]], [pos[frame, 1]])
        moon_point.set_3d_properties([pos[frame, 2]])

        days = met[frame] / 86400
        met_text.set_text(f"MET: {days:6.1f} d\n"
                          f"     {met[frame]:,.0f} s\n"
                          f"UTC: {utc[frame][:10]}")
        return trail, moon_point, met_text

    ani = animation.FuncAnimation(
        fig, update, frames=len(pos), init_func=init, interval=interval, blit=False
    )
    return ani
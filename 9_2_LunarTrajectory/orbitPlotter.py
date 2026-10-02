import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt


def animate_trajectory(moon_state, sc_state=None, interval=50, title="Earth-Moon System"):
    """
    Build the 3D animated trajectory plot for the Moon and (optionally) a spacecraft.

    Input:
        moon_state : n x 5 array [x, y, z, MET (s), UTC string]
        sc_state   : n x 8 array [x, y, z, MET, UTC, vx, vy, vz], or None
                     (must share epochs with moon_state)
        interval   : delay between frames in ms (bigger = slower)
        title      : plot title
    Returns:
        ani : the FuncAnimation object (must be kept alive by the caller)
    """
    moon_pos = moon_state[:, 0:3].astype(float)
    met = moon_state[:, 3].astype(float)
    utc = moon_state[:, 4]

    has_sc = sc_state is not None
    if has_sc:
        sc_pos = sc_state[:, 0:3].astype(float)
        n = min(len(moon_pos), len(sc_pos))   # in case the integration stopped early
    else:
        n = len(moon_pos)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Earth at origin
    ax.scatter(0, 0, 0, color="blue", s=100, label="Earth")

    # Moon artists
    (moon_trail,) = ax.plot([], [], [], color="grey", linewidth=1, label="Moon path")
    (moon_point,) = ax.plot([], [], [], color="grey", marker="o", markersize=8, label="Moon")

    # Spacecraft artists
    if has_sc:
        (sc_trail,) = ax.plot([], [], [], color="red", linewidth=1, label="Spacecraft path")
        (sc_point,) = ax.plot([], [], [], color="red", marker="o", markersize=4, label="Spacecraft")

    # Axis limits from everything plotted
    all_pos = np.vstack((moon_pos, sc_pos)) if has_sc else moon_pos
    max_limit = np.max(np.abs(all_pos)) * 1.1
    ax.set_xlim([-max_limit, max_limit])
    ax.set_ylim([-max_limit, max_limit])
    ax.set_zlim([-max_limit, max_limit])

    ax.set_xlabel("X (KM)")
    ax.set_ylabel("Y (KM)")
    ax.set_zlabel("Z (KM)")
    ax.set_title(title)
    ax.legend()

    met_text = fig.text(0.02, 0.95, "", fontsize=11, family="monospace",
                        va="top", ha="left")

    # Helper so each body is updated the same way
    def set_body(trail, point, pos, frame):
        trail.set_data(pos[:frame + 1, 0], pos[:frame + 1, 1])
        trail.set_3d_properties(pos[:frame + 1, 2])
        point.set_data([pos[frame, 0]], [pos[frame, 1]])
        point.set_3d_properties([pos[frame, 2]])

    def artists():
        out = [moon_trail, moon_point, met_text]
        if has_sc:
            out += [sc_trail, sc_point]
        return tuple(out)

    def init():
        for line in (moon_trail, moon_point) + ((sc_trail, sc_point) if has_sc else ()):
            line.set_data([], [])
            line.set_3d_properties([])
        met_text.set_text("")
        return artists()

    def update(frame):
        set_body(moon_trail, moon_point, moon_pos, frame)
        if has_sc:
            set_body(sc_trail, sc_point, sc_pos, frame)

        days = met[frame] / 86400
        met_text.set_text(f"MET: {days:6.1f} d\n"
                          f"     {met[frame]:,.0f} s\n"
                          f"UTC: {utc[frame][:10]}")
        return artists()

    ani = animation.FuncAnimation(
        fig, update, frames=n, init_func=init, interval=interval, blit=False
    )
    return ani
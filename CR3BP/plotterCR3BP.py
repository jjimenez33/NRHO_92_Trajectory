import matplotlib.pyplot as plt
import constants as c
import numpy as np

def plot_cr3bp_orbit(solRot, solNonrot, solInert, solLLO):
    """Plots the 3D rotating and non-rotating frame orbit for CR3BP.

    Parameters:
    - solRot: array-like, trajectory solution where columns 0, 1, 2 are X, Y, Z
    positions (Non-Dimensional Units)
    - solNonrot: array-like, trajectory solution where columns 0, 1, 2 are X, Y, Z
        positions (Dimensional Units)
    """
    # Rotational Frame Position Time History
    X_rot = solRot[:, 0]
    Y_rot = solRot[:, 1]
    Z_rot = solRot[:, 2]

    # Non-rotational Frame Position Time History
    X_nonrot = solNonrot[:, 0]
    Y_nonrot = solNonrot[:, 1]
    Z_nonrot = solNonrot[:, 2]

    # 9:2 NRHO Position Time History (Inertial Moon Centered Frame)
    X_inert = solInert[:, 0]
    Y_inert = solInert[:, 1]
    Z_inert = solInert[:, 2]

    # LLO Position Time History (Inertial Moon Centered Frame)
    X_llo = solLLO[:, 0]
    Y_llo = solLLO[:, 1]
    Z_llo = solLLO[:, 2]

    # Constant m1 and m2 Rotational Frame Location
    m1_loc = [-c.mu, 0, 0]
    m2_loc = [1 - c.mu, 0, 0]

    def set_axes_equal(ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_center = np.mean(x_limits)
        y_center = np.mean(y_limits)
        z_center = np.mean(z_limits)

        radius = 0.5 * max(
            x_limits[1] - x_limits[0],
            y_limits[1] - y_limits[0],
            z_limits[1] - z_limits[0]
        )

        ax.set_xlim3d([x_center - radius, x_center + radius])
        ax.set_ylim3d([y_center - radius, y_center + radius])
        ax.set_zlim3d([z_center - radius, z_center + radius])
        ax.set_box_aspect([1, 1, 1])

    # --------------------------------------------------
    # Figure 1: Rotating Frame
    # --------------------------------------------------
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(projection="3d")

    # Adding Figure Title and Labels
    ax1.set_title("Rotating Frame CR3BP Orbit (μ = " + str(round(c.mu, 6)) + ")")
    ax1.set_xlabel("x [ND]")
    ax1.set_ylabel("y [ND]")
    ax1.set_zlabel("z [ND]")

    # Plotting Rotating Frame Positions
    ax1.plot3D(X_rot, Y_rot, Z_rot, c="green", label="Spacecraft")
    ax1.plot3D(m1_loc[0], m1_loc[1], m1_loc[2], c="blue", marker="o", label="Earth")
    ax1.plot3D(m2_loc[0], m2_loc[1], m2_loc[2], c="black", marker="o", label="Moon")

    set_axes_equal(ax1)

    ax1.legend()

    # --------------------------------------------------
    # Figure 2: non-Rotating Frame
    # --------------------------------------------------
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(projection="3d")

    # Adding Figure Title and Labels
    ax2.set_title("Non-Rotating Frame CR3BP Orbit (μ = " + str(round(c.mu, 6)) + ")")
    ax2.set_xlabel("x [km]")
    ax2.set_ylabel("y [km]")
    ax2.set_zlabel("z [km]")

    # Plotting non-rotating Frame Positions
    ax2.plot3D(X_nonrot, Y_nonrot, Z_nonrot, color="green", label="Spacecraft")
    ax2.plot3D(-c.rB1, 0, 0, c="blue", marker="o", label="Earth (initial)")
    ax2.plot3D(c.rB2, 0, 0, c="black", marker="o", label="Moon (initial)")

    set_axes_equal(ax2)

    ax2.legend()

    # --------------------------------------------------
    # Figure 3: Inertial Frame with origin at the Moon's center
    # --------------------------------------------------
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(projection="3d")

    # Adding Figure Title and Labels
    ax3.set_title("9:2 NRHO in Moon Centered Inertial Frame")
    ax3.set_xlabel("x [ND]")
    ax3.set_ylabel("y [ND]")
    ax3.set_zlabel("z [ND]")

    # Plotting inertial frame positions
    ax3.plot3D(X_inert, Y_inert, Z_inert, color="green", label="Spacecraft")
    ax3.plot3D(0, 0, 0, c="black", marker="o", label="Moon (initial)")

    set_axes_equal(ax3)
    
    ax3.legend()

    # --------------------------------------------------
    # Figure 4: LLO Orbit in Moon Centered Inertial Frame
    # --------------------------------------------------
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(projection="3d")

    # Adding Figure Title and Labels
    ax4.set_title("LLO in Moon Centered Inertial Frame")
    ax4.set_xlabel("x [ND]")
    ax4.set_ylabel("y [ND]")
    ax4.set_zlabel("z [ND]")

    # Plotting inertial frame positions
    ax4.plot3D(X_llo, Y_llo, Z_llo, color="green", label="Spacecraft")
    ax4.plot3D(0, 0, 0, c="black", marker="o", label="Moon (initial)")

    set_axes_equal(ax4)
    
    ax4.legend()

    plt.show()
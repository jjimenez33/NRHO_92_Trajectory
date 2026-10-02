import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice
from kernel2state import kernel2state
from orbitPlotter import animate_trajectory
from spacecraftOrbit import orbitDynamics, propagate_spacecraft 

# Test
start_utc = "2026-01-01T00:00:00"
end_utc = "2027-01-01T00:00:00"
step_size = 43200 # step size[s] = 0.5 day
lunar_state = kernel2state(start_utc, end_utc, step_size)

# Moon state [x y z vx vy vz] at the start epoch, Earth-centered J2000
spice.furnsh("SPICE_Data/de440s.bsp")
spice.furnsh("SPICE_Data/naif0012.tls")
et0 = spice.utc2et(start_utc)
moon0, _ = spice.spkezr("MOON", et0, "J2000", "NONE", "EARTH")
spice.kclear()

# Circular orbit 100 km above the lunar surface (R_moon = 1737.4 km)
r_rel = np.array([1837.4, 0.0, 0.0])
v_rel = np.array([0.0, 1.633, 0.0])   # sqrt(muM / r)

state0 = np.concatenate((moon0[0:3] + r_rel, moon0[3:6] + v_rel))
scState = propagate_spacecraft(state0, lunar_state)

ani = animate_trajectory(lunar_state, scState, interval=50)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from kernel2state import kernel2state
from orbitPlotter import animate_trajectory
from spacecraftOrbit import orbitDynamics, propagate_spacecraft 
import constants as c

# Test
start_utc = "2026-01-01T00:00:00"
end_utc = "2026-02-01T00:00:00"
step_size = 60 # step size[s] = 1 minute
lunar_state = kernel2state(start_utc, end_utc, step_size)

# Circular orbit 400 km above Earth
r_rel = np.array([400+c.Re, 0.0, 0.0])
v_rel = np.array([0.0, np.sqrt(c.muE/(r_rel[0])), 0.0])   # sqrt(muM / r)

state0 = np.concatenate((r_rel, v_rel))
scState = propagate_spacecraft(state0, lunar_state)

ani = animate_trajectory(lunar_state, scState, interval=50)
plt.show()
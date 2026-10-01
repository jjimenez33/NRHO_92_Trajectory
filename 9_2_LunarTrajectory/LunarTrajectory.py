import numpy as np
import matplotlib.pyplot as plt
from kernel2state import kernel2state
from orbitPlotter import animate_trajectory

# Test
start_utc = "2026-01-01T00:00:00"
end_utc = "2027-01-01T00:00:00"
step_size = 43200 # step size[s] = 0.5 day
state = kernel2state(start_utc, end_utc, step_size)

# Plot
ani = animate_trajectory(state, interval=50)
plt.show()
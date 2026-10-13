import numpy as np
import constants as c
from scipy.integrate import odeint
from modelCR3BP import modelCR3BP_rot, modelCR3BP_nonrot
from plotterCR3BP import plot_cr3bp_orbit

# Initial Conditions [non-dimensional (ND)]
# Initial conditions for a 9:2 NRHO are provided by the following paper:
# "Transfers between Near Rectilinear Halo Orbits and Low Lunar Orbits"
#                   By Chelsea Thangavelu, 20218
x0_ND = 1.0221
y0_ND = 0.0 
z0_ND = -0.1821
vx0_ND = 0.0
vy0_ND = -0.1033
vz0_ND = 0.0

state0_NU = [x0_ND, y0_ND, z0_ND, vx0_ND, vy0_ND, vz0_ND]

# Initial Conditions [Dimensional Units (DU)]
x0_DU = x0_ND * c.DU
y0_DU = y0_ND * c.DU
z0_DU = z0_ND * c.DU
vx0_DU = vx0_ND * c.VU - c.ws*y0_DU
vy0_DU = vy0_ND * c.VU + c.ws*x0_DU
vz0_DU = vz0_ND * c.VU

state0_DU = [x0_DU, y0_DU, z0_DU, vx0_DU, vy0_DU, vz0_DU]

# Number of periods
numPeriods = 5
tND = np.linspace(0, 1.5112*numPeriods, 10000) # [ND Time]
tDU = np.linspace(0, 1.5112*numPeriods/c.ws, 10000) # [ND Time]

# Numerical Integration
solRot    = odeint(modelCR3BP_rot, state0_NU, tND) # Position and velocity in a rotating frame
solNonrot = odeint(modelCR3BP_nonrot, state0_DU, tDU) # Position and velocity in a non-rotating frame

# Plotting
plot_cr3bp_orbit(solRot, solNonrot)
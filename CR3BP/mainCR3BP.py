import numpy as np
import constants as c
from scipy.integrate import odeint
from modelCR3BP import *
from plotterCR3BP import plot_cr3bp_orbit
from lambertSolution import *

print("Simulation Initialized")

# 1) NRHO: Set initial conditions
# Initial conditions for a 9:2 NRHO are provided by the following paper:
# "Transfers between Near Rectilinear Halo Orbits and Low Lunar Orbits"
#                   By Chelsea Thangavelu, 20218
# Initial Conditions [Non-dimensional Units (NU)]
state0_NU = [1.0221, 0.0 , -0.1821, 0.0, -0.1033, 0.0]

# Initial Conditions [Dimensional Units (DU)]
x0_DU = state0_NU[0] * c.DU
y0_DU = state0_NU[1] * c.DU
z0_DU = state0_NU[2] * c.DU
vx0_DU = state0_NU[3] * c.VU - c.ws*y0_DU
vy0_DU = state0_NU[4] * c.VU + c.ws*x0_DU
vz0_DU = state0_NU[5] * c.VU

state0_DU = [x0_DU, y0_DU, z0_DU, vx0_DU, vy0_DU, vz0_DU]

# LLO Parameters
a_llo = 0.00478 # Semi major axis (ND) equivalent to 100 km
e_llo = 0       # Eccentricty 
i_llo = 87      # Inclination [deg]
o_llo = 5.58    # Obliquity of the Moon [deg]
L_llo = np.linspace(0, 2*np.pi, 10000) # Mean longitude [deg]

lloParams = [a_llo, e_llo, i_llo, o_llo]

# Number of periods
numPeriods = 5
tND = np.linspace(0, 1.5112*numPeriods, 10000)      # [ND Time]
tDU = np.linspace(0, 1.5112*numPeriods/c.ws, 10000) # [ND Time]
C = jacobi_constant(state0_NU)

# Numerical Integration
solRot    = odeint(modelCR3BP_rot, state0_NU, tND)    # Position and velocity in a rotating frame
solNonrot = odeint(modelCR3BP_nonrot, state0_DU, tDU) # Position and velocity in a non-rotating frame
solInert = rot2inert(solRot, tND)
solInert_llo = lloInert(lloParams, L_llo)

# Lambert (Moon-Centered Inertial, ND)
t0 = 0.0
deltaT = 3.1e5 / c.TU
t_arr = t0 + deltaT

r0 = solInert[0, 0:3] 
print(f"r0 = {r0} N.D \n")
LLO_arr = lloInert(lloParams, [np.pi/2])[0]
r1 = LLO_arr[0:3]
print(f"r1 = {r1} N.D \n")
v0, v1 = lambertSol(r0, r1, deltaT, tm=1, tol=1e-8)
print("Solution to lambert problem \n")
print(f"V0 = {v0} N.D")
print(f"v1 = {v1} N.D")

# Transfer initial state -> rotating frame at t0
state_transfer0_rot = inert2rot(np.hstack((r0, v0)), t0)

# LLO target -> rotating frame at arrival time
LLO_arr_rot = inert2rot(LLO_arr, t_arr)
r_target_rot = LLO_arr_rot[0:3] # used in F = r(tt) - r_target
v_LLO_rot = LLO_arr_rot[3:6]    # used for dv2 = v_LLO_rot - v(tt)

print("Simulation Complete")
print(f"Jacobi Constant {C}")

# Plotting
plot_cr3bp_orbit(solRot, solNonrot, solInert, solInert_llo)

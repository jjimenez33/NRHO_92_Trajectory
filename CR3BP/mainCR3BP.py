import numpy as np
import constants as c
from scipy.integrate import odeint, solve_ivp
from modelCR3BP import *
from plotterCR3BP import plot_cr3bp_orbit, plot_transfer_frames
from lambertSolution import *
import matplotlib.pyplot as plt

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
deltaT = 75 * 3600 / c.TU # 75 hours in seconds
t_arr = t0 + deltaT

r0 = solInert[0, 0:3] 
print(f"r0 = {r0} N.D \n")
LLO_arr = lloInert(lloParams, [np.pi/2])[0]
r1 = LLO_arr[0:3]
print(f"r1 = {r1} N.D \n")
v0_ND, v1_ND = lambertSol(r0, r1, deltaT, tm=1, tol=1e-8)
v0_DU, v1_DU = v0_ND*c.VU, v1_ND*c.VU
print("Solution to lambert problem \n")
print(f"V0 = {v0_ND} N.D")
print(f"v1 = {v1_ND} N.D")
print(f"V0 = {v0_DU} km/s")
print(f"v1 = {v1_DU} km/s")

# Transfer initial state -> rotating frame at t0
state_transfer0_rot = inert2rot(np.hstack((r0, v0_ND)), t0)

# LLO target -> rotating frame at arrival time
LLO_arr_rot = inert2rot(LLO_arr, t_arr)
r_target_rot = LLO_arr_rot[0:3] # used in F = r(tt) - r_target
v_LLO_rot = LLO_arr_rot[3:6]    # used for dv2 = v_LLO_rot - v(tt)

print("Simulation Complete")
print(f"Jacobi Constant {C}")

# Check how accurate Lambert's Solution is: should be ~1e-2
sol = solve_ivp(lambda t, y: modelCR3BP_rot(y, t), [0, deltaT], state_transfer0_rot,
                method='DOP853', rtol=1e-12, atol=1e-12)
missLambert = np.linalg.norm(sol.y[0:3, -1] - r_target_rot)
print(f"Open-loop miss at arrival: {missLambert:.3e} ND  ({missLambert*c.DU:.0f} km)")

# Single Shooting Correction of Lambert's Solution
V, Xf, hist, ok = correct_transfer(state_transfer0_rot[0:3], state_transfer0_rot[3:6], deltaT, r_target_rot, tol=1e-12, max_iter=100)
tf = V[3]
tau = np.linspace(0, tf, 2000)
X0_corr = np.hstack((state_transfer0_rot[0:3], V[0:3]))
solCorr = solve_ivp(lambda t, y: modelCR3BP_rot(y, t), [0, tf], X0_corr,
                    t_eval=tau, method='DOP853', rtol=1e-12, atol=1e-12)
transfer_rot = solCorr.y.T # (N, 6) Barycentric rotating, ND
transfer_inert = rot2inert(transfer_rot, t0 + tau) # (N, 6) Moon-centered inertial, ND
missCorrected = np.linalg.norm(transfer_rot[-1, 0:3] - r_target_rot)

print("converged:", ok, "| iterations: ", len(hist))
print(f"Corrected Trajectory miss at arrival: {missCorrected:.3e} ND  ({1e6*missCorrected*c.DU:.5f} mm)")

# DeltaV 1: Transfer initial velocity - NRHO velocity at departure 
v_NRHO_rot = solRot[0, 3:6]
dv1_vec = V[0:3] - v_NRHO_rot # (4.10)

# DeltaV 2: LLO velocity at arrival - transfer velocity at arrival
LL_arr_rot_corr = inert2rot(LLO_arr, t0 + V[3]) # LLO state at corrected arrival
v_LLO_rot_corr = LL_arr_rot_corr[3:6]
dv2_vec = v_LLO_rot_corr - Xf[3:6]

dv1 = np.linalg.norm(dv1_vec)
dv2 = np.linalg.norm(dv2_vec)

print("Delta-v 1 [km/s] (x, y, z, |.|):", dv1_vec*c.VU, dv1*c.VU)
print("Delta-v 2 [km/s] (x, y, z, |.|):", dv2_vec*c.VU, dv2*c.VU)
print(f"Total Delta-v = {(dv1 + dv2)*c.VU:.4f} km/s")
print(f"TOF = {V[3]*c.TU/3600:.1f} hr")

# Plotting
plot_cr3bp_orbit(solRot, solNonrot, solInert, solInert_llo)
one_period = tND <= 1.5112     
plot_transfer_frames(transfer_rot, transfer_inert, r_target_rot, LLO_arr[0:3],
                         nrho_rot=solRot[one_period], nrho_inert=solInert[one_period],
                         llo_inert=solInert_llo)


plt.show()
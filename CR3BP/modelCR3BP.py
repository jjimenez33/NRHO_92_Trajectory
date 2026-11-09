import numpy as np
import constants as c
from scipy.integrate import solve_ivp

# CR3BP Model (Rotating Frame)
def modelCR3BP_rot(state, t):
    """
    Description: CR3BP Equations of Motion in the rotating frame
                 taken from Vallado pg. 971 Eq.(12-16)
    Inputs:
        state  : state vector [6x1]
    Returns:
        dstate : derivative of state vector [6x1]
    """
    x = state[0] 
    y = state[1] 
    z = state[2]
    vx = state[3] 
    vy = state[4] 
    vz = state[5]

    r1 = np.sqrt((x + c.mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x + c.mu - 1)**2 + y**2 + z**2)

    ax = 2*vy + x - (1-c.mu)*(x + c.mu) / r1**3 - c.mu*(x-1 + c.mu)/r2**3
    ay = -2*vx + y - (1-c.mu)*y / r1**3 - c.mu*y / r2**3
    az = -(1-c.mu)*z / r1**3 - c.mu*z / r2**3

    dstate = [vx, vy, vz, ax, ay, az]

    return dstate

# CR3BP Model (non-rotating Frame)
def modelCR3BP_nonrot(state, t):
    """
    Description: CR3BP dimensional equations of motion in the non-rotating 
                 frame from Vallado pg. 971, Eq. (12-17)
    Inputs:
        state  : state vector [6x1]
    Returns:
        dstate : derivative of state vector [6x1]
    """
    x = state[0] 
    y = state[1] 
    z = state[2] 
    vx = state[3] 
    vy = state[4] 
    vz = state[5] 

    r1 = np.sqrt((x+c.rB1*np.cos(c.ws*t))**2 + (y+c.rB1*np.sin(c.ws*t))**2 + z**2)
    r2 = np.sqrt((x-c.rB2*np.cos(c.ws*t))**2 + (y-c.rB2*np.sin(c.ws*t))**2 + z**2)

    ax = -c.mu1*(x+c.rB1*np.cos(c.ws*t))/r1**3 - c.mu2*(x-c.rB2*np.cos(c.ws*t))/r2**3
    ay = -c.mu1*(y+c.rB1*np.sin(c.ws*t))/r1**3 - c.mu2*(y-c.rB2*np.sin(c.ws*t))/r2**3
    az = -c.mu1*z/r1**3 - c.mu2*z/r2**3

    dstate = [vx, vy, vz, ax, ay, az]

    return dstate

# CR3BP Model Transformation from Rotating to inertial Frame(Moon Centered Inertial Frame)
def rot2inert(state_rot, t):
    """
    Description: Transformation matrix from rotating frame to inertial frame
                 based on equations from "Transfers between Near Rectilinear 
                 Halo Orbits and Low Lunar Orbits" by Thangavelu (2018). 
                 Specifically Section 2
    Inputs:
        state_rot: state vector in rotating frame
        t: non-dimensional time
    Returns:
        state_I: state vector in moon-centered rotating frame
    """
    t = np.asarray(t)
    N = state_rot.shape[0]

    # Shift origin to P2 
    r = state_rot[:, :3].copy()
    v = state_rot[:, 3:6].copy()
    r[:, 0] -= (1 - c.mu)

    # Rotation angle at each time (ws = 1 in nondimensional CR3BP)
    alpha = t
    ca, sa = np.cos(alpha), np.sin(alpha)

    # Rotation matrices C_RN, shape (N, 3, 3)
    C = np.zeros((N, 3, 3))
    C[:, 0, 0] = ca; C[:, 0, 1] = -sa
    C[:, 1, 0] = sa; C[:, 1, 1] = ca
    C[:, 2, 2] = 1

    # Rotation matrices Cdot_RN, shape (N, 3, 3)
    Cdot = np.zeros((N, 3, 3))
    Cdot[:, 0, 0] = -sa; Cdot[:, 0, 1] = -ca
    Cdot[:, 1, 0] = ca; Cdot[:, 1, 1] = -sa

    # r_I = C * r_R; v_I = Cdot r_R + C v_R
    r_inert = np.einsum('nij, nj->ni', C, r)
    v_inert = np.einsum('nij, nj->ni', Cdot, r) + np.einsum('nij, nj->ni', C, v)

    return np.hstack((r_inert, v_inert))

def inert2rot(state_I, t):
    r_I, v_I = state_I[:3], state_I[3:]
    ca, sa = np.cos(t), np.sin(t)
    C    = np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]])
    Cdot = np.array([[-sa,-ca,0],[ca,-sa,0],[0,0,0]])
    r_R = C.T @ r_I
    v_R = C.T @ (v_I - Cdot @ r_R)
    r_R[0] += (1 - c.mu)        # shift back to barycentric rotating
    return np.hstack((r_R, v_R))

def lloInert(lloParams, L):
    """
    Description: Transformation matrix from rotating frame to inertial frame
                 based on equations from "Transfers between Near Rectilinear 
                 Halo Orbits and Low Lunar Orbits" by Thangavelu (2018). 
                 Specifically Section 4
    Inputs:
        lloParams: parameters of the llo orbit which include
            a: semi-major axis
            e: eccentricity
            i: inclination
            o: obliquity of the Moon
        L: Mean longitude
    Returns:
        position and velocity vector (N.D)
    """
    a, e, i, o = lloParams[0], lloParams[1], lloParams[2], lloParams[3]
    d2r = np.pi/180
    L = np.asarray(L)
    N = L.shape[0]
    
    co, so = np.cos(o*d2r), np.sin(o*d2r)
    ci, si = np.cos(i*d2r), np.sin(i*d2r)
    sl, cl = np.sin(L), np.cos(L)

    # Rotation about x by o ("tilt")
    Ctilt = np.array([[1.0, 0.0, 0.0],
                      [0.0, co, -so],
                      [0.0, so,  co]])

    # Rotation about x by i (inclination)
    Cinc = np.array([[1.0, 0.0, 0.0],
                     [0.0, ci, -si],
                     [0.0, si,  ci]])

    C = Ctilt @ Cinc

    zeros = np.zeros(N)
    r_llom3 = np.array([a*cl, a*sl, zeros]) # Original paper didn't have a*sl only sl, a*sl should be correct
    vc = np.sqrt(c.mu / a) # ND
    v_llom3 = np.array([-vc*sl, vc*cl, zeros])

    r_llom2 = C @ r_llom3
    v_llom2 = C @ v_llom3

    return np.hstack((r_llom2.T, v_llom2.T))

# Jacobi constant (rotating frame, ND)
def jacobi_constant(state):
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x + c.mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x + c.mu - 1)**2 + y**2 + z**2)
    v = np.sqrt(vx**2+vy**2+vz**2)
    C = x**2 + y**2 + 2*(1-c.mu)/r1 + 2*c.mu/r2 - v**2

    return C

# State Transition Matrix (rotating frame, ND)
def modelCR3BP_STM(t, y):
    """
    Description: State Transition Matrix in the rotating frame with 
                 ND units based on the Thungavelu paper Section 3 and 4
    Inputs: 
        t = time [ND]
        y = State vector [x, y, z, vx, vy, vz, Phi(36)]

    Returns:
        dy = derivative of State vector [vx, vy, vz, ax, ay, az, dPhi(36)]
    """
    x, yy, z, vx, vy, vz = y[:6]
    Phi = y[6:].reshape(6, 6)
    mu = c.mu

    # Equations of motion
    r1 = np.sqrt((x+mu)**2 + yy**2 + z**2)
    r2 = np.sqrt((x+mu-1)**2 + yy**2 + z**2)

    ax = 2*vy + x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
    ay = -2*vx + yy - (1-mu)*yy/r1**3 - mu*yy/r2**3
    az = -(1-mu)*z/r1**3 - mu*z/r2**3
 
    a, b = (1-mu)/r1**3, mu/r2**3
    d1, d2 = 3*(1-mu)/r1**5, 3*mu/r2**5
    xm, xp = x + mu, x + mu - 1
 
    Uxx = 1 - a - b + d1*xm**2 + d2*xp**2
    Uyy = 1 - a - b + d1*yy**2 + d2*yy**2
    Uzz =   - a - b + d1*z**2  + d2*z**2
    Uxy = d1*xm*yy + d2*xp*yy
    Uxz = d1*xm*z  + d2*xp*z
    Uyz = d1*yy*z  + d2*yy*z

    # State Matrix from Eq (3.24, 3.25-34)
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 0:3] = [[Uxx, Uxy, Uxz],
                   [Uxy, Uyy, Uyz],
                   [Uxz, Uyz, Uzz]]
    A[3:6, 3:6] = [[0, 2, 0],
                   [-2, 0, 0],
                   [0, 0, 0]]

    dPhi = A @ Phi
    return np.hstack(([vx, vy, vz, ax, ay, az], dPhi.ravel()))

# State with STM Propagator
def propagate_with_STM(X0, tt):
    """
    Description: Propagates state with STM from 0 to tt. 
    Inputs:
        X0: Initial state vector 
        tt: Transfer time of flight?
    Returns:
        X(tt) (6,) : State 
        Phi(tt) : State Transition Matrix
    """
    y0 = np.hstack((X0, np.eye(6).ravel()))
    sol = solve_ivp(modelCR3BP_STM, [0, tt], y0, method='DOP853',
                    rtol=1e-12, atol=1e-12)
    yf = sol.y[:, -1]
    Xtt = yf[:6]
    Phitt = yf[6:].reshape(6,6)
    return Xtt, Phitt


# Single Shooting Corrector (Thesis Section 4.1.4, eqs 4.7-4.9)
# Free variables = [vx0, vy0, vz0, tt] (rotating frame)
# Constraint F = r(tt) - r_target
def correct_transfer(r0_rot, v_guess, tt_guess, r_target, tol, max_iter):
    """
    Description: Single Shooting Corrector (Thungavelu Section 4.1.4, eqs 4.7-4.9)
    Inputs: 
        r0_rot: Initial state vector in non dimenionsal rotating frame
        v_guess: initial velocity vector from Lambert's Problem
        tt_guess: Transfer time of flight guess
        r_target: Desired position
        tol: Tolerance
        max_iter: Maximum iteration #
    Return:
        V
        Xf
        history
        False
    """
    V = np.hstack((v_guess, tt_guess)).astype(float) # (4.7)
    history = []

    for j in range(max_iter):
        X0 = np.hstack((r0_rot, V[0:3]))
        Xf, Phi = propagate_with_STM(X0, V[3])

        F = Xf[0:3] - r_target # (4.8)
        history.append(np.linalg.norm(F))
        if history[-1] < tol:
            return V, Xf, history, True

        DF = np.zeros((3, 4)) # (4.9)
        DF[:, 0:3] = Phi[0:3, 3:6]
        DF[:, 3] = Xf[3:6]

        V = V - DF.T @ np.linalg.solve(DF @ DF.T, F) # (3.48)

    return V, Xf, history, False
# Helper function to obtain deltafV2
def dv2_vector(V, Xf, LLO_arr, t0):
    """deltaV2 = v_LLO(rotating frame at t0+tt) - v_transfer(tt)"""
    v_LLO_rot = inert2rot(LLO_arr, t0 + V[3])[3:6]
    return v_LLO_rot - Xf[3:6] # Equation 4.11

# Corrections for Transfers Including deltaV Constraint (Thesis Section 4.2)
# Free Variables [vx0, vy0, vz0, tt] (rotating frame)
# Constraint F = [r(tt)-r_target, deltaV2-deltaV2_corrected]
def correct_transfer_deltaV_constraint(r0_rot, v_guess, r_target, LLO_arr, t0, dv2_target, tol=1e-12, max_iter=50):
    """
    Description: Single Shooting Corrector with deltaV constraint (Thungavelu Section 4.2, eq 4.12)
    Inputs:
        r0_rot: Initial state vector in non dimenionsal rotating frame
        v_guess: initial velocity vector from Lambert's Problem
        tt_guess: Transfer time of flight guess
        r_target: Desired position
        deltaV: ????
        tol: Tolerance
        max_iter: Maximum iteration #
    Return:
        V
        Xf
        history 
        False
    """
    
    V = np.array(v_guess, dtype=float)

    for j in range(max_iter):
        X0 = np.hstack((r0_rot, V[0:3]))
        Xf, Phi = propagate_with_STM(X0, V[3])

        dv2_vec = dv2_vector(V, Xf, LLO_arr, t0)
        dv2_mag = np.linalg.norm(dv2_vec)

        F = np.hstack((Xf[0:3] - r_target, dv2_mag - dv2_target)) # (4.12)
        if np.linalg.norm(F) < tol:
            return V, Xf, True

        # Derivative of the deltaV2 vector with respect to tt
        acc_final = np.array(modelCR3BP_rot(Xf, 0.0)[3:6])
        h = 1e-6
        vL_p = inert2rot(LLO_arr, t0 + V[3] + h)[3:6]
        vL_m = inert2rot(LLO_arr, t0 + V[3] - h)[3:6]
        dvLLO_dtt = (vL_p - vL_m) / (2*h)
        d_dv2vec_dtt = dvLLO_dtt - acc_final

        # Jacobian (4x4)
        u = dv2_vec / dv2_mag # unit vector of deltaV2
        DF = np.zeros((4,4)) # (4.9) with added deltaV2
        DF[0:3, 0:3] = Phi[0:3, 3:6] # Position vs initial velocity
        DF[0:3, 3] = Xf[3:6] # POsition vs tt = final velocity
        DF[3, 0:3] = -u @ Phi[3:6, 3:6] # DeltaV2 vs initial velocity
        DF[3, 3] = u @ d_dv2vec_dtt # deltaV2 vs tt

        V = V - np.linalg.solve(DF, F) # Ordinary Newton Step

    return V, Xf, False
        
# Continuation loop
def dv2_continuation(r0_rot, V0, Xf0, r_target, LLO_arr, t0, v_NRHO_rot, step=1e-3, n_steps=50, min_steps=1e-6):
    V, Xf = np.array(V0, float), Xf0
    results = []

    dv2_now = np.linalg.norm(dv2_vector(V, Xf, LLO_arr, t0))
    dv1_now = np.linalg.norm(V[0:3] - v_NRHO_rot)
    results.append((dv1_now, dv2_now, dv1_now + dv2_now, V[3]))

    for k in range(n_steps):
        dv2_target = dv2_now - step
        V_new, Xf_new, ok = correct_transfer_deltaV_constraint(r0_rot, V, r_target, LLO_arr, t0, dv2_target)

        if not ok:
            step /= 2
            if step < min_steps:
                print("Stopping: could not converge with a small step.")
                break
            continue

    V, Xf = V_new, Xf_new
    dv2_now = np.linalg.norm(dv2_vector(V, Xf, LLO_arr, t0))
    dv1_now = np.linalg.norm(V[0:3] - v_NRHO_rot)
    results.append((dv1_now, dv2_now, dv1_now + dv2_now, V[3]))

    return np.array(results) # columns: dv1, dv2, total, tt



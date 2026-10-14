import numpy as np
import constants as c

# CR3BP Model (Rotating Frame)
def modelCR3BP_rot(state, t):
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
    # Convert initial conditions to dimensionalized units
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
    t = np.asarray(t)
    N = state_rot.shape[0]

    # Shift origin to P2 
    r = state_rot[:, :3].copy()
    v = state_rot[:, :3].copy()
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
    C[:, 0, 0] = -sa; C[:, 0, 1] = -ca
    C[:, 1, 0] = ca; C[:, 1, 1] = -sa

    # r_I = C * r_R; v_I = Cdot r_R + C v_R
    r_inert = np.einsum('nij, nj->ni', C, r)
    v_inert = np.einsum('nij, nj->ni', Cdot, r) + np.einsum('nij, nj->ni', C, v)

    return np.hstack((r_inert, v_inert))

def lloInert(lloParams, L):
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
    r_llom2 = np.array([a*cl, a*sl, zeros]) # Original paper didn't have a*sl only sl, a*sl should be correct
    v_llom2 = np.array([-np.sqrt(c.mu2ND/a)*sl, np.sqrt(c.mu2/a)*cl, zeros])

    r_llom3 = C @ r_llom2
    v_llom3 = C @ v_llom2

    return np.hstack((r_llom3.T, v_llom3.T))

# Jacobi constant (rotating frame, ND)
def jacobi_constant(state):
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x + c.mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x + c.mu - 1)**2 + y**2 + z**2)
    v = np.sqrt(vx**2+vy**2+vz**2)
    C = x**2 + y**2 + 2*(1-c.mu)/r1 + 2*c.mu/r2 - v**2

    return C
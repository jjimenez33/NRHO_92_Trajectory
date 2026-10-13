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
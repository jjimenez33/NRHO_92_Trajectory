import numpy as np
import constants as c

# Lambert Solution - Universal Variables 
# Algorithm taken from paper titled:
#   LAMBERT UNIVERSAL VARIABLE ALGORITHM (2003)
#       M.A. Sharaf, A.N. Saad and M.I. Nouh

def lambertSol(r0, r1, deltaT, tm, tol):
    """ Inputs: 
            r0: initial position vector
            r1 : final Position vector
            deltaT: time of flight
            t_m: transfer-method
                + (short way)
                - (long way)
        Returns:
            v0: initial velocity vector
            v1: final velocity vector
    """
    # Calculate square root of mu parameter
    sqrt_mu = np.sqrt(c.mu2)

    # Calculate the norm of position vectors
    r0_norm = np.linalg.norn(r0)
    r1_norm = np.linalg.norm(r1)

    # Calculate gamma parameter
    gamma = np.dot(r0,r1) / (r0_norm*r1_norm)

    # Calculate beta parameter
    beta = tm * np.sqrt((1-gamma**2))

    # Calculate A parameter
    A = tm * np.sqrt(r0_norm*r1_norm*(1+gamma))

    # Check if A = 0, solution can't be found
    if A == 0.0:
        raise ValueError("Solution cannot be calculated")
        return np.array([0, 0, 0]), np.array([0, 0, 0])
   
    # Initial Values of c2 and c3 parameters and others
    c2 = 0.5
    c3 = 1/6.0
    psi = 0
    psi_u = 4*np.pi**2
    psi_l = -4*np.pi

    # Counter and solved variables
    step = 0
    max_steps = 200
    solved = False

    # While tolerance not met and not at max steps
    for n in range(max_steps):
        
        # Calculate B parameter
        B = r0_norm + r1_norm + A*(psi*c3-1) / np.sqrt(c2)
        
        # if A and B parameters are out of range
        if A > 0.0 and B <0.0:
            # Increase lower psi value
            psi_l+=np.pi

            # recalculate B parameter
            B*=-1

        # Calculate universal variable cubed
        chi3 = np.sqrt(B/c2)**3

        # Calculate deltaT_ variable
        deltaT_ = (chi3*c3+A*np.sqrt(B)) / sqrt_mu
        
        # if difference between deltaT variables is within tolerance
        if abs(deltaT - deltaT_) < tol:

            # Set solved variable to true
            solved = True

            # Break out of for loop
            break

        # Check difference between deltaT and deltaT_
        if deltaT_ <= deltaT:

            # adjust lower psi value
            psi_l = psi

        # else deltaT_ > deltaT
        else:

            # Adjust upper psi value
            psi_u = psi

        # Update psi, c2 and c3 values
        psi = (psi_u + psi_l) / 2.0
        c2 = c2(psi)
        c3 = c3(psi)

    # Check if maximum steps reached
    if not solved:

        # Algorithm didn't converge on a psi value
        print('Lamberts UV variables did not converge')
        return np.array([0,0,0]),np.array([0,0,0])

    # Calculate coefficients
    f = 1 - B/r0_norm
    g = A*np.sqrt(B/c.mu2)
    gdot = 1 - B/r1_norm

    # Calcualte velocity vectors
    v0 = (r1 - f*r0) / g
    v1 = (gdot*r1 - r0) / g


    return v0, v1

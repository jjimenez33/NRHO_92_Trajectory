import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import constants as c

def orbitDynamics(t, y, moon_pos):
    """
    Inputs:
        float t         : time since start (MET) [s]
        array y         : [x, y, z, vx, vy, vz]  Earth-centered J2000 [km, km/s]
        callable moon_pos : moon_pos(t) -> Earth->Moon position [km] (3,)
    Returns:
        array yDOT : [vx, vy, vz, ax, ay, az]
    """

    r_E2SC = y[0:3]
    v_SC = y[3:6]

    r_E2M = moon_pos(t)
    r_M2SC = r_E2SC - r_E2M

    a_SC = (-c.muE * r_E2SC / np.linalg.norm(r_E2SC)**3
            -c.muM * (r_M2SC / np.linalg.norm(r_M2SC)**3
                      + r_E2M / np.linalg.norm(r_E2M)**3))

    return np.concatenate((v_SC, a_SC))

def propagate_spacecraft(y0, lunarState, t_end=None):
    """
    Inputs:
        array y0         : initial state [x, y, z, vx, vy, vz] at lunarState's first epoch
        array lunarState : output of kernel2state (n x 5)
        float t_end      : end MET [s]; defaults to end of lunarState
    Returns:
        n x 8 object array : [x, y, z, MET, UTC, vx, vy, vz]
    """
    lunar_pos = lunarState[:, 0:3].astype(float)
    met = lunarState[:, 3].astype(float)
    utc = lunarState[:, 4]

    moon_pos = CubicSpline(met, lunar_pos)   # moon_pos(t) -> (3,)

    if t_end is None:
        t_end = met[-1]

    sol = solve_ivp(
        orbitDynamics, (met[0], t_end), y0, args=(moon_pos,),
        method="DOP853", rtol=1e-10, atol=1e-10,
        t_eval=met[met <= t_end],            # output at the same epochs as the Moon
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    n = sol.y.shape[1]
    state = np.empty((n, 8), dtype=object)
    state[:, 0:3] = sol.y[0:3].T
    state[:, 3] = sol.t
    state[:, 4] = utc[:n]
    state[:, 5:8] = sol.y[3:6].T
    
    return state









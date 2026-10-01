import numpy as np
import spiceypy as spice

def kernel2state(start_utc, end_utc, step_size):
    """
    Input: 
        string start_utc : start time in ISO
            ex) "2026-01-01T00:00:00"
        string end_utc   : end time in ISO
        double step_size  : deltaT
    Returns:
        nx5 array lunarState : [double x (km), double y (km), double z (km), double MET (s), string UTC]
    """

    # Load SPICE kernels
    spice.furnsh("SPICE_Data/de440s.bsp")
    spice.furnsh("SPICE_Data/naif0012.tls")
    print("SPICE Kernels successfully loaded.")

    # Define time array 
    et_start = spice.utc2et(start_utc)
    et_end = spice.utc2et(end_utc)
    et_array = np.arange(et_start, et_end, step_size)

    # Initialize a lsit to hold state rows
    state_data = []

    for et in et_array:
        # 1. Calculate Mission Elapsed Time (MET) in seconds
        met = et - et_start

        # 2. Get position vector (x, y, z) in (km)
        position, lt = spice.spkpos("MOON", et, "J2000", "NONE", "EARTH")

        #3. Convert current ET back to an ISO UTC string for output
        utc_str = spice.et2utc(et, "ISOC", 3)

        # Combine data
        # [x, y, z, MET, UTC]
        row = [position[0], position[1], position[2], met, utc_str]
        state_data.append(row)

    # Unload kernels to clean up memory
    spice.kclear()

    # Return as a numpy object array to support mixed types (floats and strings)
    return np.array(state_data, dtype=object)
# Constants.py
import numpy as np

# Gravitational Constant
G = 6.67430e-20 # [km^3 kg^-2 s^-2]

# Earth Constants
m1  = 5.972e24 # Earth mass [kg]
mu1 = G*m1 # Earth gravitational parameter
R_EARTH = 6378.137 # Earth equatorial radius [km]

# Moon Constants
m2  = 7.348e22 # Moon Mass [kg]
mu2 = G*m2 # Moon gravitational Parameter
mu2ND = 0.012144
R_MOON = 1737.4 # Moon mean radius [km]

# Characteristic Values
Lstar = 3.844e5 # Avereage distance from Earth to the Moon [km]
Mstar = m1+m2
Tstar = (Lstar**3/(G*Mstar))**(1/2) # Characteristic Time
mu = m2/Mstar
rB1 = mu * Lstar # Earth to barycenter [km]
rB2 = (1 - mu) * Lstar # Moon to barycenter [km]
ws = (G*Mstar/Lstar**3)**(0.5)

# CR3BP normalization
DU = Lstar
TU = 1 / ws
VU = DU / TU

# Parking orbit
LEO_ALT = 200.0 # [km]
R_LEO_ND = (R_EARTH + LEO_ALT) / DU
R_MOON_ND = R_MOON / DU
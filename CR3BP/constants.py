# Constants.py
import numpy as np

# Gravitational Constant
G = 6.67430e-20 # [km^3 kg^-2 s^-2]

# Earth Constants
m1  = 5.972e24 # Earth mass [kg]
mu1 = G*m1

# Moon Constants
m2  = 7.348e22 # Moon Mass [kg]
mu2 = G*m2

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
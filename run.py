from inputparameters import InputParameters
from radpydro import RadPydro
import numpy as np

input = InputParameters()
input.geometry = 'cylindrical'
input.r_half = np.linspace(0, 5, num=6)
input.C_v = 1.0
input.gamma = 1.0
input.kappa = [1, 1, 1, 1]
input.kappa_s = 1
input.a = 0.01372 # [j / (cm3 kev4)]
input.E = lambda r: 1.0
input.rho = lambda r: 1.0
input.T = lambda r: 1.0
input.u = lambda r: 1.0

# Boundary conditions
input.hydro_L = 'u'
input.hydro_L_val = 1
input.hydro_R = 'u'
input.hydro_R_val = 1
input.rad_L = 'reflective'
input.rad_R = 'source'
input.rad_R_val = 0

rp = RadPydro(input)
rp.hydro.solveVelocity(5, True)

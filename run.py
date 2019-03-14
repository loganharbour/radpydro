from inputparameters import InputParameters
from radpydro import RadPydro
import numpy as np

input = InputParameters()
input.geometry = 'cylindrical'
input.r_half = np.linspace(0, 5, num=6)
input.C_v = 1.0
input.gamma = 1.0
input.kappa = [1, 1, 1, 1]
input.a = 1.0
input.E_BC = [0, 0]
input.E = lambda r: 1.0
input.rho = lambda r: 1.0
input.T = lambda r: 1.0
input.u = lambda r: 1.0
input.constrain_u = True

rp = RadPydro(input)
rp.hydro.solveVelocity(5, True)

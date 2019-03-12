from inputparameters import InputParameters
from radpydro import RadPydro
import numpy as np

input = InputParameters()
input.geometry = 'cylindrical'
input.r_half = np.linspace(0, 5, num=6)
input.C_v = lambda r: 1.0
input.rho = lambda r: 1.0
input.m = lambda r: 1.0
input.gamma = lambda r: 1.0
input.k = lambda r: [1, 1, 1, 1]
input.u_BC = [1, 1]
input.T = lambda r: 1.0

rp = RadPydro(input)

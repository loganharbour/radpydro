from inputparameters import InputParameters
from radpydro import RadPydro
from initial_conditions import InitialConditions
import numpy as np
import matplotlib.pyplot as plt

input = InputParameters()
input.enable_radiation = False
input.geometry = 'slab'
input.N = 250
input.R_L = -0.25
input.R_R = 0.25
input.r_half = np.linspace(input.R_L, input.R_R, num=input.N + 1) # cm
input.C_v = 1.66 # jerks / (cm3 eV)

'''
# Initial conditions
ic = InitialConditions(input)
input.rho = ic.rho     # g/cm3
input.T = ic.T         # keV
input.u = ic.u         # cm/sh
input.E = ic.E
'''

# Boundary conditions
input.hydro_L = 'u'
input.hydro_L_val = 1.42601
input.hydro_R = 'u'
input.hydro_R_val = 1.32658
input.rad_L = 'reflective'
input.rad_L_val = None
input.rad_R = 'source'
input.rad_R_val = 0

# Iteration controls
input.CoFactor = 0.5
input.relEFactor = 0.2
input.maxTimeStep = 0.1
input.T_final = 1.5

rp = RadPydro(input)
rp.run(True)
rp.fields.plotFields()

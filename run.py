from inputparameters import InputParameters
from radpydro import RadPydro
from initial_conditions import InitialConditions
import numpy as np
import matplotlib.pyplot as plt

input = InputParameters()
input.enable_radiation = False
<<<<<<< HEAD
input.geometry = 'slab'
input.N = 1000
input.R_L = -0.25
input.R_R = 0.25
input.r_half = np.linspace(input.R_L, input.R_R, num=input.N + 1) # cm
input.C_v = 0.14472799784454 # jerks / (cm3 eV)
=======
input.geometry = 'spherical'
input.N = 250
input.R = 1
input.r_half = np.linspace( 0, input.R, num=input.N + 1) # cm
input.C_v = 1.66 # jerks / (cm3 eV)
>>>>>>> 46eddf48f19a2d70552e9e40bffb8e6912038da7
input.gamma = 5/3 # cm3 / g
input.kappa = [1, 1, 1, 1] # g/cm2
input.kappa_s = 1 # g / cm2


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
input.hydro_L_val = 0.152172533
input.hydro_R = 'u'
input.hydro_R_val = 0.117297805
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

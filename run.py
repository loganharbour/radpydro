from inputparameters import InputParameters
from radpydro import RadPydro
import numpy as np
import matplotlib.pyplot as plt

input = InputParameters()
input.enable_radiation = False
input.geometry = 'spherical'
input.N = 250
input.R = 1
input.r_half = np.linspace( 0, input.R, num=input.N + 1) # cm
input.C_v = 1.66 # jerks / (cm3 eV)
input.gamma = 5/3 # cm3 / g
input.kappa = [1, 1, 1, 1] # g/cm2
input.kappa_s = 1 # g / cm2
input.a = 0.01372 # [jerks / (cm3 kev4)]
input.c = 299.792 # [cm / sh]

# Initial conditions
input.rho = lambda r: 1        # g/cm3
input.T = lambda r: 1          # keV
input.u = lambda r: 1 - r**2  # cm/sh
input.E = lambda r: input.a * input.T(0)**4
# Boundary conditions
input.hydro_L = 'P'
input.hydro_L_val = 10
input.hydro_R = 'P'
input.hydro_R_val = (input.gamma - 1) * input.C_v
input.rad_L = 'reflective'
input.rad_L_val = None
input.rad_R = 'source'
input.rad_R_val = 0

# Iteration controls
input.CoFactor = 0.25
input.relEFactor = 0.2
input.maxTimeStep = 0.01
input.T_final = 0.05

rp = RadPydro(input)
rp.run()
rp.fields.plotFields()

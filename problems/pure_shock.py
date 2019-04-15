import sys
sys.path.insert(0, '..')
from radpydro import *
import matplotlib.pyplot as plt

input = InputParameters()
input.running_mode = 'hydro'
input.geometry = 'slab'
input.N = 1000
input.r_half = np.linspace(-0.25, 0.25, num=input.N + 1) # cm

# done
input.C_v = 1.66 # jerks / (cm3 eV)
input.gamma = 5/3 # cm3 / g

# Unused
input.kappa = [0, 0, 1, 0]
input.kappa_s = 0 # g / cm2
input.a = 0.01372 # [jerks / (cm3 kev4)]
input.c = 299.792 # [cm / sh]

# Initial conditions
input.rho = lambda r: 1 * (r < 0) + 1.07495 * (r >= 0)
input.u = lambda r: 1.42601 * (r < 0) + 1.32658 * (r >= 0)
input.T = lambda r: 1 * (r < 0) + 1.04946 * (r >= 0)
input.E = lambda r: 0
# Boundary conditions
input.hydro_L = 'u'
input.hydro_L_val = None
input.hydro_R = 'u'
input.hydro_R_val = None
input.rad_L = 'reflective'
input.rad_L_val = None
input.rad_R = 'reflective'
input.rad_R_val = None

# Iteration controls
input.CoFactor = 0.5
input.relEFactor = 0.2
input.maxTimeStep = 0.0005
input.T_final = 0.25

rp = RadPydro(input)
rp.run()

# Fields plot
rp.fields.plotFields()

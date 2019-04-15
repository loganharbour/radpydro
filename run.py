from inputparameters import InputParameters
from radpydro import RadPydro
from initial_conditions import InitialConditions
import numpy as np
import matplotlib.pyplot as plt

input = InputParameters()
input.running_mode = 'radhydro'
input.geometry = 'slab'
input.N = 1000
input.r_L = -0.25
input.r_R = 0.25
input.r_half = np.linspace(input.r_L, input.r_R, num=input.N + 1) # cm
input.C_v = 0.14472799784454 # jerks / (cm3 eV)
input.gamma = 5/3 # cm3 / g
input.kappa = [577.35, 0, 1, 0] # g/cm2
input.kappa_s = 1 # g / cm2
input.a = 0.01372 # [jerks / (cm3 kev4)]
input.c = 299.792 # [cm / sh]

# Boundary conditions
input.hydro_L = 'u'
input.hydro_L_val = None
input.hydro_R = 'u'
input.hydro_R_val = None
input.rad_L = 'source'
input.rad_L_val = None
input.rad_R = 'source'
input.rad_R_val = None

# Iteration controls
input.CoFactor = 0.5
input.relEFactor = 0.2
input.maxTimeStep = 0.0005
input.T_final = 1.5
input.T_final = 1.0


rp = RadPydro(input)
rp.run()
rp.fields.plotFields()

fig = plt.figure()
plt.plot(rp.geo.r, rp.fields.T, label='T')
plt.plot(rp.geo.r, (rp.fields.E/rp.input.a)**(1/4), label='$\\theta$')
plt.xlim([-0.02, 0.02])
plt.legend(loc=0)
plt.show()

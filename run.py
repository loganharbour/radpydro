from inputparameters import InputParameters
from radpydro import RadPydro
import numpy as np

input = InputParameters()
input.geometry = 'cylindrical'
input.r_half = np.linspace(0, 5, num=6)
input.C_v = 1.0
input.gamma = 1.5
input.kappa = [1, 1, 1, 1]
input.kappa_s = 1 #
input.a = 0.01372 # [jerks / (cm3 kev4)]
input.c = 299.792 # [cm / sh]
input.E = lambda r: 1.0
input.rho = lambda r: 1.0
input.T = lambda r: 273.15
input.u = lambda r: 1.0

# Boundary conditions
input.hydro_L = 'P'
input.hydro_L_val = 5
input.hydro_R = 'P'
input.hydro_R_val = 5
input.rad_L = 'reflective'
input.rad_R = 'reflective'

# Iteration controls
input.CoFactor = 1
input.relEFactor = 0.2
input.maxTimeStep = 0.5
input.T_final = 1

rp = RadPydro(input)
rp.computeTimeStep()
rp.hydro.solveVelocity(rp.timeSteps[-1], True)
rp.geo.moveMesh(rp.timeSteps[-1], True)
rp.fields.recomputeRho(True)
print(rp.fields.u_old, rp.fields.u_p)
print(rp.fields.rho_old, rp.fields.rho_p)



#rp.radPredictor.solveSystem(rp.timeSteps[-1])
#rp.radCorrector.solveSystem(rp.timeSteps[-1])


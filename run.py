from inputparameters import InputParameters
from radpydro import RadPydro
import numpy as np
import matplotlib.pyplot as plt

input = InputParameters()
input.geometry = 'slab'
input.N = 5
input.r_half = np.linspace(0, 1, num=input.N + 1) # cm
input.C_v = 1.66 # jerks / (cm3 eV)
input.gamma = 1.5 # cm3 / g
input.kappa = [1, 0, 1, 0] # g/cm2
input.kappa_s = 1 # g / cm2
input.a = 0.01372 # [jerks / (cm3 kev4)]
input.c = 299.792 # [cm / sh]

# Initial conditions
input.rho = lambda r: 1.0 #g/cm3
input.T = lambda r: 1.0 # keV
input.u = lambda r: 0.0 # cm / sh
input.E = lambda r: input.a * input.T(0)**4

# Boundary conditions
input.hydro_L = 'u'
input.hydro_L_val = 0.01
input.hydro_R = 'u'
input.hydro_R_val = 0.01
input.rad_L = 'source'
input.rad_L_val = input.a * input.T(0)**4
input.rad_R = 'source'
input.rad_R_val = 0

# Iteration controls
input.CoFactor = 1
input.relEFactor = 0.2
input.maxTimeStep = 0.01
input.T_final = 1

rp = RadPydro(input)
rp.computeTimeStep()

# Predictor step
rp.hydro.solveVelocity(rp.timeSteps[-1], True)
rp.geo.moveMesh(rp.timeSteps[-1], True)
rp.fields.recomputeRho(True)
rp.radPredictor.solveSystem(rp.timeSteps[-1])
rp.fields.recomputeInternalEnergy(rp.timeSteps[-1], True)
rp.fields.recomputeT(True)
rp.fields.recomputeP(True)

# Corrector step
rp.hydro.solveVelocity(rp.timeSteps[-1], False)
rp.geo.moveMesh(rp.timeSteps[-1], False)
rp.fields.recomputeRho(False)
rp.radCorrector.solveSystem(rp.timeSteps[-1])
rp.fields.recomputeInternalEnergy(rp.timeSteps[-1], False)
rp.fields.recomputeT(False)
rp.fields.recomputeP(False)

energy = rp.fields.conservationCheck(rp.timeSteps[-1])

print('\nInitial Cell Edges: ',             rp.geo.r_half_old)
print('Predictor Cell Edges: ',        rp.geo.r_half_p)
print('New Step Cell Edges: ',              rp.geo.r_half)

print('\nInitial Density: ',                rp.fields.rho_old)
print('Predictor Density: ',           rp.fields.rho_p)
print('New Step Density: ',                 rp.fields.rho)

print('\nInitial Velocity: ',               rp.fields.u_old)
print('Predictor Velocity: ',          rp.fields.u_p)
print('New Step Velocity: ',                rp.fields.u)

print('\nInitial Pressure: ',               rp.fields.P_old)
print('Predictor Pressure: ',               rp.fields.P_p)
print('New Step Pressure: ',                rp.fields.P)

print('\nInitial Internal Energy: ',        rp.fields.e_old)
print('Predictor Internal Energy: ',   rp.fields.e_p)
print('New Step Internal Energy: ',         rp.fields.e)

print('\nInitial Radiation Energy: ',       rp.fields.E_old)
print('Predictor Radiation Energy: ',  rp.fields.E_p)
print('New Step Radiation Energy: ',        rp.fields.E)

print('\nInitial Temperature: ',            rp.fields.T_old)
print('Predictor Temperature: ',       rp.fields.T_p)
print('New Step Temperature: ',                rp.fields.T)

print('\nEnergy Conservation Check for Time Step: ', energy)

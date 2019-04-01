from inputparameters import InputParameters
from radpydro import RadPydro
import numpy as np
import matplotlib.pyplot as plt

input = InputParameters()
input.geometry = 'slab'
input.N = 5
input.r_half = np.linspace(0, 5, num=input.N + 1)
input.C_v = 1.0
input.gamma = 1.5
input.kappa = [1, 1, 1, 1]
input.kappa_s = 1 #
input.a = 0.01372 # [jerks / (cm3 kev4)]
input.c = 299.792 # [cm / sh]

# Initial conditions
input.rho = lambda r: 1.0
input.T = lambda r: 273.15
input.u = lambda r: 1.
input.E = lambda r: input.a * input.T(0)**4

# Boundary conditions
input.hydro_L = 'u'
input.hydro_L_val = None
input.hydro_R = 'u'
input.hydro_R_val = None
input.rad_L = 'source'
input.rad_L_val = input.E(0)
input.rad_R = 'source'
input.rad_R_val = input.E(1)

# Iteration controls
input.CoFactor = 1
input.relEFactor = 0.2
input.maxTimeStep = 0.5
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
print('Predictor Step Cell Edges: ',        rp.geo.r_half_p)
print('New Step Cell Edges: ',              rp.geo.r_half)

print('\nInitial Density: ',                rp.fields.rho_old)
print('Predictor Step Density: ',           rp.fields.rho_p)
print('New Step Density: ',                 rp.fields.rho)

print('\nInitial Velocity: ',               rp.fields.u_old)
print('Predictor Step Velocity: ',          rp.fields.u_p)
print('New Step Velocity: ',                rp.fields.u)

print('\nInitial Internal Energy: ',        rp.fields.e_old)
print('Predictor Step Internal Energy: ',   rp.fields.e_p)
print('New Step Internal Energy: ',         rp.fields.e)

print('\nInitial Radiation Energy: ',       rp.fields.E_old)
print('Predictor Step Radiation Energy: ',  rp.fields.E_p)
print('New Step Radiation Energy: ',        rp.fields.E)

print('\nInitial Pressure: ',               rp.fields.P_old)
print('Predictor Step Pressure: ',          rp.fields.P_p)
print('New Step Pressure: ',                rp.fields.P)

print('\nInitial Temperature: ',            rp.fields.T_old)
print('Predictor Step Temperature: ',       rp.fields.T_p)
print('New Step Pressure: ',                rp.fields.T)

print('\nEnergy Conservation Check for Time Step: ', energy)

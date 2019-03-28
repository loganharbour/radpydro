from inputparameters import InputParameters
from radpydro import RadPydro
import numpy as np

input = InputParameters()
input.geometry = 'slab'
input.r_half = np.linspace(0, 5, num=6)
input.C_v = 1.0
input.gamma = 1.5
input.kappa = [1, 1, 1, 1]
input.kappa_s = 1 #
input.a = 0.01372 # [jerks / (cm3 kev4)]
input.c = 299.792 # [cm / sh]
input.E = lambda r: 1.
input.rho = lambda r: 1.0
input.T = lambda r: 273.15
input.u = lambda r: 1.

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
input.CoFactor = 1
input.relEFactor = 0.2
input.maxTimeStep = 0.5
input.T_final = 1

rp = RadPydro(input)
rp.computeTimeStep()
rp.hydro.solveVelocity(rp.timeSteps[-1], True)
rp.geo.moveMesh(rp.timeSteps[-1], True)
rp.fields.recomputeRho(True)
rp.radPredictor.solveSystem(rp.timeSteps[-1])

print('\nInitial Cell Edges: ', rp.geo.r_half_old)
print('Predictor Step Cell Edges: ', rp.geo.r_half_p)

print('\nInitial Density: ', rp.fields.rho_old)
print('Predictor Step Density: ', rp.fields.rho_p)

print('\nInitial Velocity: ', rp.fields.u_old)
print('Predictor Step Velocity: ', rp.fields.u_p)

print('\nInitial Internal Energy', rp.fields.e_old)
print('Predictor Step Internal Energy: ', rp.fields.e_p)

print('\nInitial Radiation Energy: ', rp.fields.E_old)
print('Predictor Step Radiation Energy: ', rp.fields.E_p)

print('\nInitial Pressure: ', rp.fields.P_old)
print('\nInitial Temperature: ', rp.fields.T_old)


#rp.radCorrector.solveSystem(rp.timeSteps[-1])

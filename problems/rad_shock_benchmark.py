import sys
sys.path.insert(0, '..')
from radpydro import *
import matplotlib.pyplot as plt

input = InputParameters()
input.running_mode = 'radhydro'
input.geometry = 'slab'
input.N = 1000
input.r_half = np.linspace(-0.25, 0.25, num=input.N + 1) # cm

# done
input.C_v = 0.14472799784454 # jerks / (cm3 eV)
input.gamma = 5/3 # cm3 / g
input.kappa = [577.35, 0, 1, 0] # g/cm2
input.kappa_s = 0 # g / cm2
input.a = 0.01372 # [jerks / (cm3 kev4)]
input.c = 299.792 # [cm / sh]

# Initial conditions
input.rho = lambda r: 1 * (r < 0) + 1.29731782 * (r >= 0)
input.u = lambda r: 0.152172533 * (r < 0) + 0.117297805 * (r >= 0)
input.T = lambda r: 0.1 * (r < 0) + 0.119475741 * (r >= 0)
input.E = lambda r: 1.37201720e-6 * (r < 0) + 2.79562228e-6 * (r >= 0)
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
input.maxTimeStep = 0.005
input.T_final = 1.5

rp = RadPydro(input)
rp.run()

# Fields plot
rp.fields.plotFields([-0.02, 0.02])

# Somethin
fig = plt.figure()
plt.plot(rp.geo.r, rp.fields.T, label='T')
plt.plot(rp.geo.r, (rp.fields.E/rp.input.a)**(1/4), label='$\\theta$')
plt.xlim([-0.02, 0.02])
plt.tight_layout()
plt.legend()
plt.savefig('gigi.pdf')

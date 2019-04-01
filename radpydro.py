import numpy as np

from fields import Fields
from geometry import SlabGeometry, CylindricalGeometry, SphericalGeometry
from inputparameters import InputParameters
from lagrangian_hydro import LagrangianHydro
from lagrangian_radiation_predictor import LagrangianRadiationPredictor
from lagrangian_radiation_corrector import LagrangianRadiationCorrector

from materials import Materials

class RadPydro:
    def __init__(self, input):

        # Store input parameters and check them
        self.input = input
        self.input.checkInputs()

        # Define geometry based on geometry input type
        if input.geometry == 'slab':
            self.geo = SlabGeometry(self)
        elif input.geometry == 'cylindrical':
            self.geo = CylindricalGeometry(self)
        else:
            self.geo = SphericalGeometry(self)

        # Initialize material handler now that geometry is initialized
        self.mat = Materials(self)

        # Initialize field variables
        self.fields = Fields(self)

        # Time step
        self.timeSteps = []

        # Initialize the radiation and hydro problems
        self.hydro = LagrangianHydro(self)
        self.radPredictor = LagrangianRadiationPredictor(self)
        self.radCorrector = LagrangianRadiationCorrector(self)

    def computeTimeStep(self):

        dr = self.geo.dr
        u = self.fields.u
        F_c = self.input.CoFactor
        relEFactor = self.input.relEFactor

        c_s = (self.mat.gamma * self.fields.P / self.fields.rho)**(1 / 2)

        E_k = (self.fields.E + self.fields.E_old) / 2

        dE_k = np.zeros(self.geo.N)
        if len(self.timeSteps) == 0:
            dE_k = E_k
        else:
            dE_k = abs((self.fields.E + self.fields.E_old) / self.timeSteps[-1])

        u_center = np.zeros(self.geo.N)
        for i in range(self.geo.N):
            u_center = (u[i] + u[i+1]) / 2

        dt_E = min(relEFactor * E_k / dE_k)
        dt_u = min(dr * F_c / u_center)
        dt_cs = min(dr * F_c / c_s)

        self.timeSteps.append(min(self.input.maxTimeStep, dt_E, dt_u, dt_cs))

        print('Computed time step size: ' + str(self.timeSteps[-1]), '\n')

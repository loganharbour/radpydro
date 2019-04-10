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

        # Time parameters
        self.timeSteps = []
        self.time = 0.
        self.timeStep_num = 0
        self.Tf = input.Tf

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
            dE_k = abs((self.fields.E - self.fields.E_old) / self.timeSteps[-1])

        u_center = np.zeros(self.geo.N)
        for i in range(self.geo.N):
            u_center = (u[i] + u[i+1]) / 2

        dt_E = min(relEFactor * E_k / dE_k)
        dt_u = min(dr * F_c / u_center)
        dt_cs = min(dr * F_c / c_s)

        self.timeSteps.append(min(self.input.maxTimeStep, dt_E, dt_u, dt_cs))

    def run(self):
        while self.time < self.Tf:
            # Compute time step size for this time step
            self.computeTimeStep()

            # Update time and time step number
            self.time += self.timeSteps[-1]
            self.timeStep_num += 1
            print('=========================================================')
            print('Starting time step %i,  time = %.3e'  \
                    % (self.timeStep_num, self.time))
            print('=========================================================\n')

            # Add artificial viscosity for this time step
            self.fields.addArtificialViscosity()

            # Predictor step
            self.hydro.solveVelocity(self.timeSteps[-1], True)
            self.geo.moveMesh(self.timeSteps[-1], True)
            self.fields.recomputeRho(True)
            self.radPredictor.solveSystem(self.timeSteps[-1])
            self.fields.recomputeInternalEnergy(self.timeSteps[-1], True)
            self.fields.recomputeT(True)
            self.fields.recomputeP(True)

            # Corrector step
            self.hydro.solveVelocity(self.timeSteps[-1], False)
            self.geo.moveMesh(self.timeSteps[-1], False)
            self.fields.recomputeRho(False)
            self.radCorrector.solveSystem(self.timeSteps[-1])
            self.fields.recomputeInternalEnergy(self.timeSteps[-1], False)
            self.fields.recomputeT(False)
            self.fields.recomputeP(False)

            # Energy conservation check
            energy_diff = self.fields.conservationCheck(self.timeSteps[-1])
            print('Energy Conservation Check for Time Step: ', energy_diff, '\n')

            # Copy to old containers for next time step
            self.fields.stepFields()
            self.geo.stepGeometry()

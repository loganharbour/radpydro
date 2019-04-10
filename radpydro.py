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

        # Initialize hydro problem
        self.hydro = LagrangianHydro(self)

        # Initialize radiation problem (if used)
        if input.enable_radiation:
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

        if self.input.enable_radiation:
            self.timeSteps.append(min(self.input.maxTimeStep, dt_E, dt_u, dt_cs))
        else:
            self.timeSteps.append(min(self.input.maxTimeStep, dt_u, dt_cs))

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
            self.hydro.recomputeVelocity(True)
            self.geo.moveMesh(True)
            self.hydro.recomputeDensity(True)

            if self.input.enable_radiation:
                self.radPredictor.recomputeRadiationEnergy()

            self.hydro.recomputeInternalEnergy(True)
            self.hydro.recomputeTemperature(True)
            self.hydro.recomputePressure(True)

            # Corrector step
            self.hydro.recomputeVelocity(False)
            self.geo.moveMesh(False)
            self.hydro.recomputeDensity(False)

            if self.input.enable_radiation:
                self.radCorrector.recomputeRadiationEnergy()

            self.hydro.recomputeInternalEnergy(False)
            self.hydro.recomputeTemperature(False)
            self.hydro.recomputePressure(False)

            # Energy conservation check
            energy_diff = self.fields.conservationCheck()
            print('Energy Conservation Check for Time Step: ', energy_diff, '\n')

            # Copy to old containers for next time step
            self.fields.stepFields()
            self.geo.stepGeometry()

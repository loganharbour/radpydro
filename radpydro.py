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
        self.dt = self.input.dt

        # Initialize hydro problem
        self.hydro = LagrangianHydro(self)

        # Initialize radiation problem (if used)
        if input.enable_radiation:
            self.radPredictor = LagrangianRadiationPredictor(self)
            self.radCorrector = LagrangianRadiationCorrector(self)

        # Init storage for energies in conservation check
        self.kinetic_energy = []
        self.internal_energy = []
        self.radiation_energy = []
        self.radiation_leakage = []
        self.work_energy = []
        self.total_energy = []

        # Compute initial energies for each
        kinetic = 0
        internal = 0
        radiation = 0
        for i in range(self.geo.N + 1):
            kinetic += 1/2 * self.mat.m_half[i] * self.fields.u_IC[i]**2

            if i < self.geo.N:
                internal += self.mat.m[i] * self.fields.e_IC[i]
                radiation += self.mat.m[i] * self.fields.E_IC[i] / self.fields.rho_IC[i]
        total = kinetic + internal + radiation

        self.kinetic_energy.append(kinetic)
        self.internal_energy.append(internal)
        self.radiation_energy.append(radiation)
        self.radiation_leakage.append(0)
        self.work_energy.append(0)
        self.total_energy.append(total)

        self.total_radiation_leakage = 0
        self.total_work_energy = 0

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

    def run(self, PLOT=False):
        while self.time < self.input.T_final:

            # Compute time step size for this time step
            if self.input.dt is None:
                self.computeTimeStep()
            else:
                self.timeSteps.append(self.input.CoFactor * self.input.dt)

            # Update time and time step number
            self.time += self.timeSteps[-1]
            self.timeStep_num += 1
            print('=========================================================')
            print('Starting time step %i,  time = %.3e'  \
                    % (self.timeStep_num, self.time))
            print('=========================================================\n')

            if PLOT:
                if self.timeStep_num % 100 == 0 or self.timeStep_num == 1:
                    self.fields.plotFields()
                else:
                    pass
            else:
                pass

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
            energy_diff = self.recomputeEnergyConservation()
            print('Energy conservation check: ', energy_diff, '\n')

            # Copy to old containers for next time step
            self.fields.stepFields()
            self.geo.stepGeometry()

    def recomputeEnergyConservation(self):
        kinetic_energy = self.kinetic_energy
        internal_energy = self.internal_energy
        radiation_energy = self.radiation_energy
        radiation_leakage = self.radiation_leakage
        work_energy = self.work_energy
        total_energy = self.total_energy

        c = self.input.c
        dt = self.timeSteps[-1]
        m = self.mat.m
        m_half = self.mat.m_half

        u = self.fields.u
        e = self.fields.e
        E = self.fields.E
        rho = self.fields.rho

        A_k = (self.geo.A + self.geo.A_old) / 2
        A_pk = (self.geo.A_p + self.geo.A_old) / 2

        dr_k = (self.geo.dr + self.geo.dr_old) / 2
        dr_pk = (self.geo.dr_p + self.geo.dr_old) /2

        E_k = (self.fields.E + self.fields.E_old) / 2
        E_pk = (self.fields.E_p + self.fields.E_old) / 2

        T_k = (self.fields.T + self.fields.T_old) / 2
        T_pk = (self.fields.T_p + self.fields.T_old) / 2

        rho_k = (self.fields.rho + self.fields.rho_old) / 2
        rho_pk = (self.fields.rho_p + self.fields.rho_old) / 2

        u_k = (self.fields.u + self.fields.u_old) / 2

        P_pk = (self.fields.P_p + self.fields.P_old) / 2

        # Recomputing kappa_t at the cell edges and cell centers
        self.mat.recomputeKappa_t(T_pk)
        kappa_t_pk_edge = self.mat.kappa_t
        self.mat.recomputeKappa_a(T_pk)
        kappa_t_pk_center = self.mat.kappa_a + self.mat.kappa_s

        # Setting up boundary parameters for the radiation terms
        # in the momentum equation
        if self.input.rad_L is 'source':
            E_bL_k = self.fields.E_bL
            E_bL_pk = self.fields.E_bL
        else:
            E_bL_k = E_k[0]
            E_bL_pk = E_pk[0]
        if self.input.rad_R is 'source':
            E_bR_k = self.fields.E_bR
            E_bR_pk = self.fields.E_bR
        else:
            E_bR_k = E_k[-1]
            E_bR_pk = E_pk[-1]

        # Compute the boundary radiation energies in the momentum eqn
        coeff_E_L = 3 * rho_pk[0] * dr_pk[0] * kappa_t_pk_center[0]
        coeff_E_R = (3 * rho_pk[-1] * dr_pk[-1] * kappa_t_pk_center[-1])

        E_L = (coeff_E_L * E_bL_pk + 4 * E_pk[0]) / (coeff_E_L + 4)
        E_R = (coeff_E_R * E_bR_pk + 4 * E_pk[-1]) / (coeff_E_R + 4)

        # Compute radiation flux at boundaries
        coeff_F_L = -2 * c / (3 * rho_k[0] * dr_k[0] * kappa_t_pk_edge[0] + 4)
        coeff_F_R = -2 * c / (3 * rho_k[-1] * dr_k[-1] * kappa_t_pk_edge[-1] + 4)

        F_L = coeff_F_L * (E_k[0] - E_bL_k)
        F_R = coeff_F_R * (E_bR_k - E_k[-1])

        # Setting up boundary parameters for the pressure boundary values
        if self.input.hydro_L is 'P':
            P_bL_pk = self.fields.P_L
        else:
            P_bL_pk = P_pk[0] + 1 / 3 * (E_pk[0] - E_L)
        if self.input.hydro_R is 'P':
            P_bR_pk = self.fields.P_R
        else:
            P_bR_pk = P_pk[-1] + 1 / 3 * (E_pk[-1] - E_R)

        # Compute kinetic, internal, and radiation energies for this timestep
        kinetic, internal, radiation = 0, 0, 0
        for i in range(self.geo.N + 1):
            kinetic += 1/2 * m_half[i] * u[i]**2

            if i < self.geo.N:
                internal += m[i] * e[i]
                radiation += m[i] * E[i] / rho[i]

        # Compute radiation leakage
        leakage = (A_k[-1] * F_R - A_k[0] * F_L) * dt

        # Compute compressive work
        work = (A_pk[-1] * 1/3 * E_R * u_k[-1] - A_pk[0] * 1/3 * E_L * u_k[0]) * dt

        work += (A_pk[-1] * P_bR_pk * u_k[-1] - A_pk[0] * P_bL_pk * u_k[0]) * dt

        # Compute total energy
        total = kinetic + internal + radiation + leakage + work

        # Compute energy final - initial energies
        dKE = kinetic - kinetic_energy[0]
        dIE = internal- internal_energy[0]
        dRE = radiation - radiation_energy[0]

        # Compute energy losses from pressure work, drift, and leakage
        total_work = self.total_work_energy + work
        total_leak = self.total_radiation_leakage + leakage

        # Update loss terms from pressure work, drift, and leakage
        self.total_work_energy += work
        self.total_radiation_leakage += leakage

        # Append to storage
        kinetic_energy.append(kinetic)
        internal_energy.append(internal)
        radiation_energy.append(radiation)
        radiation_leakage.append(leakage)
        work_energy.append(work)
        total_energy.append(total)

        return dKE + dIE + dRE + total_work + total_leak

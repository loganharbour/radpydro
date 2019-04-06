import numpy as np

class Fields:
    def __init__(self, rp):
        self.rp = rp
        self.input = rp.input
        self.geo = rp.geo
        self.mat = rp.mat
        self.N = rp.geo.N

        # Velocities (spatial cell edges)
        self.u = self.initializeAtEdges(self.input.u)
        self.u_p = np.copy(self.u)
        self.u_old = np.copy(self.u)
        self.u_IC = np.copy(self.u)

        # Set left velocity BCs as necessary
        if self.input.hydro_L == 'u':
            if self.input.hydro_L_val == None:
                self.u_L = self.input.u(0)
            else:
                self.u_L = self.input.hydro_L_val
                self.u_IC[0] = self.u_L;
                self.u_old[0] = self.u_L;
                self.u_p[0] = self.u_L;
                self.u[0] = self.u_L;
        else:
            self.u_L = None

        # Set right velocity BCs as necessary
        if self.input.hydro_R == 'u':
            if self.input.hydro_R_val == None:
                self.u_R = self.input.u(self.geo.r_half_old[-1])
            else:
                self.u_R = self.input.hydro_R_val
                self.u_IC[-1] = self.u_R;
                self.u_old[-1] = self.u_R;
                self.u_p[-1] = self.u_R;
                self.u[-1] = self.u_R;
        else:
            self.u_R = None

        # Temperature (spatial cell centers)
        self.T = self.initializeAtCenters(self.input.T)
        self.T_p = np.copy(self.T)
        self.T_old = np.copy(self.T)

        # Densities (spatial cell centers)
        self.rho = self.initializeAtCenters(self.input.rho)
        self.rho_p = np.copy(self.rho)
        self.rho_old = np.copy(self.rho)
        self.rho_IC = np.copy(self.rho)

        # Pressures (spatial cell centers, IC: Eqs. 22 and 23)
        self.P = (self.mat.gamma - 1) * self.mat.C_v * self.T * self.rho
        self.P_p = np.copy(self.P)
        self.P_old = np.copy(self.P)

        # Set pressure BCs as necessary
        if self.input.hydro_L == 'P':
            self.P_L = self.input.hydro_L_val
        else:
            self.P_L = None
        if self.input.hydro_R == 'P':
            self.P_R = self.input.hydro_R_val
        else:
            self.P_R = None

        # Internal energies (spatial cell centers)
        self.e = self.mat.C_v * self.T_old
        self.e_p = np.copy(self.e)
        self.e_old = np.copy(self.e)
        self.e_IC = np.copy(self.e)

        # Radiation energies (spatial cell centers)
        self.E = self.initializeAtCenters(self.input.E)
        self.E_p = np.copy(self.E)
        self.E_old = np.copy(self.E)
        self.E_IC = np.copy(self.E)

        # Set E boundary conditions
        if self.input.rad_L == 'source':
            self.E_bL = self.input.rad_L_val
        else:
            self.E_bL = None
        if self.input.rad_R == 'source':
            self.E_bR = self.input.rad_R_val
        else:
            self.E_bR = None

        # Init the rest of the materials that depend on field variables
        self.mat.initFromFields(self)

    # Copy over all new fields to old positions
    def stepFields(self):
        np.copyto(self.u_old, self.u)
        np.copyto(self.T_old, self.T)
        np.copyto(self.rho_old, self.rho)
        np.copyto(self.P_old, self.P)
        np.copyto(self.e_old, self.e)
        np.copyto(self.E_old, self.E)

    # Initialize variable with function at the spatial cell centers
    def initializeAtCenters(self, function):
        values = np.zeros(self.N)
        if function is not None:
            for i in range(self.N):
                values[i] = function(self.geo.r[i])
        return values

    # Initialize variable with function at the spatial cell edges
    def initializeAtEdges(self, function):
        values = np.zeros(self.N + 1)
        if function is not None:
            for i in range(self.N + 1):
                values[i] = function(self.geo.r_half[i])
        return values

    # Recmpute density with updated cell volumes
    def recomputeRho(self, predictor):
        m = self.mat.m
        if predictor:
            rho_new = self.rho_p
            V_new = self.geo.V_p
        else:
            rho_new = self.rho
            V_new = self.geo.V
        for i in range(self.N):
            rho_new[i] = m[i] / V_new[i]

    # Recompute radiation energy with updated internal energy
    def recomputeInternalEnergy(self, dt, predictor):
        # Constants
        m = self.mat.m
        a = self.input.a
        c = self.input.c
        C_v = self.mat.C_v

        if predictor:
            e_old = self.e_old
            T_old = self.T_old
            E_pk = (self.E_p + self.E_old) / 2
            xi_old = self.rp.radPredictor.xi
            self.mat.recomputeKappa_a(T_old)
            kappa_a_old = self.mat.kappa_a

            increment = dt * C_v * (m * kappa_a_old * c * (E_pk - a * T_old**4) + xi_old)
            increment /= m * C_v + dt * m * kappa_a_old * c * 2 * a * T_old**3

            self.e_p = e_old + increment

        else:
            e_p = self.e_p
            T_old = self.T_old
            T_p = self.T_p
            T_pk = (T_p + T_old) / 2
            T_pk4 = (T_p**4 + T_old**4) / 2
            E_k = (self.E + self.E_old) / 2
            self.mat.recomputeKappa_a(T_pk)
            kappa_a_pk = self.mat.kappa_a
            xi_k = self.rp.radCorrector.xi

            increment = dt * C_v * (m * kappa_a_pk * c * (E_k - a * T_pk4) + xi_k)
            increment /= m * C_v + dt * m * kappa_a_pk * c * 2 * a * T_p**3

            self.e = e_p + increment

    # Recompute temperature with updated internal energy
    def recomputeT(self, predictor):
        C_v = self.mat.C_v
        if predictor:
            T_new = self.T_p
            e_new = self.e_p
        else:
            T_new = self.T
            e_new = self.e
        for i in range(self.N):
            T_new[i] = e_new[i] / C_v

    # Recompute pressure with updated density and internal energy
    def recomputeP(self, predictor):
        gamma_minus = self.mat.gamma - 1
        if predictor:
            P_new = self.P_p
            e_new = self.e_p
            rho_new = self.rho_p
        else:
            P_new = self.P
            e_new = self.e
            rho_new = self.rho
        for i in range(self.N):
            P_new[i] = gamma_minus * rho_new[i] * e_new[i]

    # Recompute radiation energy density
    def recomputeE(self, dt, predictor):
        if predictor:
            self.radPredictor.computeAuxiliaryFields(dt)
            self.radPredictor.assembleSystem(dt)
            self.radPredictor.solveSystem(dt)
        else:
            self.radCorrector.computeAuxiliaryFields(dt)
            self.radCorrector.assembleSystem(dt)
            self.radCorrector.solveSystem(dt)


    def conservationCheck(self, dt):
        # Centered cell and median mesh cell masses
        m = self.mat.m
        m_half = self.mat.m_half

        # Physical constants
        a = self.input.a
        c = self.input.c
        N = self.geo.N

        # Initial conditions
        u_IC = self.u_IC
        e_IC = self.e_IC
        E_IC = self.E_IC
        rho_IC = self.rho_IC

        # k+1/2'th time-step quantities
        u = self.u
        e = self.e
        E = self.E
        rho = self.rho

        # k'th time-step and predicted k'th time-step variables
        A_k = (self.geo.A + self.geo.A_old) / 2
        A_pk = (self.geo.A_p + self.geo.A_old) / 2

        E_k = (self.E + self.E_old) / 2
        E_pk = (self.E_p + self.E_old) / 2

        rho_k = (self.rho + self.rho_old) / 2
        rho_pk = (self.rho_p + self.rho_old) / 2

        dr_k = (self.geo.dr + self.geo.dr_old) / 2
        dr_pk = (self.geo.dr_p + self.geo.dr_old) / 2

        P_pk = (self.P_p + self.P_old) / 2

        u_k = (self.u + self.u_old) / 2

        T_k = (self.T + self.T_old) / 2
        T_pk = (self.T_p + self.T_old) / 2

        # Recomputing kappa_t at the cell edges and cell centers
        self.mat.recomputeKappa_t(T_pk)
        kappa_t_pk_edge = self.mat.kappa_t
        self.mat.recomputeKappa_a(T_pk)
        kappa_t_pk_center = self.mat.kappa_a + self.mat.kappa_s

        # Setting up boundary parameters for the radiation terms
        # in the momentum equation
        if self.input.rad_L is 'source':
            E_bL_k = self.E_bL
            E_bL_pk = self.E_bL
        else:
            E_bL_k = E_k[0]
            E_bL_pk = E_pk[0]
        if self.input.rad_R is 'source':
            E_bR_k = self.E_bR
            E_bR_pk = self.E_bR
        else:
            E_bR_k = E_k[-1]
            E_bR_pk = E_pk[-1]

        # Computing the boundary radiation energies in the momentum
        # equation from the parameters
        coeff_E_L = 3 * rho_pk[0] * dr_pk[0] * kappa_t_pk_center[0]
        coeff_E_R = 3 * rho_pk[-1] * dr_pk[-1] * kappa_t_pk_center[-1]

        E_L = (coeff_E_L * E_bL_pk + 4 * E_pk[0]) / (coeff_E_L + 4)
        E_R = (coeff_E_R * E_bR_pk + 4 * E_pk[-1]) / (coeff_E_R + 4)

        # Setting up boundary parameters for the pressure boundary values
        if self.input.hydro_L is 'P':
            P_bL_pk = self.P_L
        else:
            P_bL_pk = P_pk[0] + 1 / 3 * (E_pk[0] - E_L)
        if self.input.hydro_R is 'P':
            P_bR_pk = self.P_R
        else:
            P_bR_pk = P_pk[-1] + 1 / 3 * (E_pk[-1] - E_R)

        # Computing the boundary radiation fluxes (used for leakage)
        coeff_F_L = -2 * c / (3 * rho_k[0] * dr_k[0] * kappa_t_pk_edge[0] + 4)
        coeff_F_R = -2 * c / (3 * rho_k[-1] * dr_k[-1] * kappa_t_pk_edge[-1] + 4)

        F_L = coeff_F_L * (E_k[0] - E_bL_k);
        F_R = coeff_F_R * (E_bR_k - E_k[-1]);

        # Computing the rolled balance at this timestep
        energy_diff = 0
        kinetic_energy_IC = 0
        internal_energy_IC = 0
        radiation_energy_IC = 0
        kinetic_energy = 0
        internal_energy = 0
        radiation_energy = 0

        for i in range(N + 1):
            kinetic_energy_IC += 1/2 * m_half[i] * u_IC[i]**2
            kinetic_energy += 1/2 * m_half[i] * u[i]**2
            if i < N:
                internal_energy_IC += m[i] * e_IC[i]
                internal_energy += m[i] * e[i]
                radiation_energy_IC += m[i] * E_IC[i] / rho_IC[i]
                radiation_energy += m[i] * E[i] / rho[i]

        energy_diff += kinetic_energy - kinetic_energy_IC

        energy_diff += internal_energy - internal_energy_IC

        energy_diff += radiation_energy - radiation_energy_IC

        energy_diff += (A_k[-1] * F_R - A_k[0] * F_L) * dt

        energy_diff += (A_pk[-1] * 1/3 * E_R * u_k[-1]  - A_pk[0]  * 1/3 * E_L * u_k[0]) * dt

        energy_diff += (A_pk[-1] * P_bR_pk * u_k[-1] - A_pk[0] * P_bL_pk * u_k[0] ) * dt

        return energy_diff

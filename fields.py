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

        # Set left velocity BCs as necessary
        if self.input.hydro_bL == 'u':
            if self.input.hydro_bL_val == None:
                self.u_bL = self.input.u(0)
            else:
                self.u_bL = self.input.hydro_bL_val
        else:
            self.u_bL = None

        # Set right velocity BCs as necessary
        if self.input.hydro_bR == 'u':
            if self.input.hydro_bR_val == None:
                self.u_bR = self.input.u(self.geo.r_half_old[-1])
            else:
                self.u_bR = self.input.hydro_bR_val
        else:
            self.u_bR = None

        # Temperature (spatial cell centers)
        self.T = self.initializeAtCenters(self.input.T)
        self.T_p = np.copy(self.T)
        self.T_old = np.copy(self.T)

        # Densities (spatial cell centers)
        self.rho = self.initializeAtCenters(self.input.rho)
        self.rho_p = np.copy(self.rho)
        self.rho_old = np.copy(self.rho)

        # Pressures (spatial cell centers, IC: Eqs. 22 and 23)
        self.P = (self.mat.gamma - 1) * self.mat.C_v * self.T * self.rho
        self.P_p = np.copy(self.P)
        self.P_old = np.copy(self.P)

        # Set pressure BCs as necessary
        if self.input.hydro_bL == 'P':
            self.P_bL = self.input.hydro_bL_val
        else:
            self.P_bL = None
        if self.input.hydro_bR == 'P':
            self.P_bR = self.input.hydro_bR_val
        else:
            self.P_bR = None

        # Internal energies (spatial cell centers)
        self.e = self.mat.C_v * self.T_old
        self.e_p = np.copy(self.e)
        self.e_old = np.copy(self.e)

        # Radiation energies (spatial cell centers)
        self.E = self.initializeAtCenters(self.input.E)
        self.E_p = np.copy(self.E)
        self.E_old = np.copy(self.E)
        # Set E boundary conditions
        if self.input.rad_bL == 'source':
            self.E_bL = self.input.rad_bL_val
        else:
            self.E_bL = None
        if self.input.rad_bR == 'source':
            self.E_bR = self.input.rad_bR_val
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
        np.copyto(self.F_L_old, self.F_L)
        np.copyto(self.F_R_old, self.F_R)

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
    def recomputeInternalEnergy(self, dt):
        # Query constants for the problem
        m = self.mat.m
        a = self.input.a
        c = self.input.c
        C_v = self.mat.C_v

        # Query k-1/2'th variables
        T_old = self.T_old
        e_old = self.e_old

        # Query absorption opacities
        kappa_a = self.mat.kappa_a

        # Query parameter from radiation energy solve
        xi = self.rp.radPredictor.xi

        # Compute average of k-1/2'th and predictor radiation energy
        E_k = (self.E_p + self.E_old) / 2

        # Compute factor to be added to k-1/2'th internal energy
        increment = dt * C_v * (m * kappa_a * c * (E_k - a * T_old**4) + xi)
        increment /= m * C_v + dt * m * kappa_a * c * 2 * a * T_old**3

        # Compute predictor internal energy
        self.e_p = e_old + increment

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
            T_new[i] = C_v * e_new[i]

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
    def recomputeE(self, dt):
        # Compute time-averaged parameters, opacities, etc
        self.radPredictor.computeAuxiliaryFields(dt)
        self.radPredictor.assembleSystem(dt)
        self.radPredictor.solveSystem(dt)

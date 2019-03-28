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
        # Set velocity BCs as necessary
        if self.input.hydro_L == 'u':
            if self.input.hydro_L_val == None:
                self.u_L = self.input.u(0)
            else:
                self.u_L = self.input.hydro_L_val
        else:
            self.u_L = None
        if self.input.hydro_R == 'u':
            if self.input.hydro_R_val == None:
                self.u_R = self.input.u(self.geo.r_half_old[-1])
            else:
                self.u_R = self.input.hydro_R_val
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

        # Radiation energies (spatial cell centers)
        self.E = self.initializeAtCenters(self.input.E)
        self.E_p = np.copy(self.E)
        self.E_old = np.copy(self.E)
        # Set E boundary conditions
        if self.input.rad_L == 'source':
            self.E_L = self.input.rad_L_val
        else:
            self.E_L = None
        if self.input.rad_R == 'source':
            self.E_R = self.input.rad_R_val
        else:
            self.E_R = None

        # Initialize the rest of the materials that depend on field variables
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

    # Recmpute rho with an updated V
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

    # Recompute temperature with an updated e
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

    # Recompute pressure with an updated rho and e
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

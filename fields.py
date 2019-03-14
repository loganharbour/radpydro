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
            self.u_BC_L = None
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
        if predictor:
            self.rho_p = self.mat.m / self.V_p
        else:
            self.rho = self.mat.m / self.V

    # Recompute temperature with an updated e
    def recomputeT(self, predictor):
        if predictor:
            self.T_p = self.mat.C_v * self.e_p
        else:
            self.T = self.mat.C_v * self.e

    # Recompute pressure with an updated rho and e
    def recomputeP(self, predictor):
        if predictor:
            self.P_p = (self.mat.gamma - 1) * self.rho_p * self.e_p
        else:
            self.P = (self.mat.gamma - 1) * self.rho * self.e

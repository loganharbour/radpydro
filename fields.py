import numpy as np

class Fields:
    def __init__(self, rp):
        self.rp = rp
        self.input = rp.input
        self.geo = rp.geo
        self.mat = rp.mat
        self.N = rp.geo.N

        # Velocities (spatial cell edges)
        self.u = np.zeros(self.N + 1)
        self.u_old = np.zeros(self.N + 1)
        # Apply velocity initial condition
        self.initializeAtEdges(self.u_old, self.input.u)

        # Temperature (spatial cell centers)
        self.T = np.zeros(self.N)
        self.T_old = np.zeros(self.N)
        # Apply temperature initial condition
        self.initializeAtCenters(self.T_old, self.input.T)
        # Initialize kappa_a and kappa_t with T_old
        self.mat.recomputeKappa_a(self.T_old)
        self.mat.recomputeKappa_t(self.T_old)

        # Densities (spatial cell centers)
        self.rho = np.zeros(self.N)
        self.rho_old = np.zeros(self.N)
        # Initialize density
        self.initializeAtCenters(self.rho_old, self.input.rho)
        # Initialize mass now that we have densities
        self.mat.initializeMasses(self.rho_old)

        # Pressures (spatial cell centers)
        self.P = np.zeros(self.N)
        # Apply pressure initial condition (Eqs. 22 and 23)
        self.P_old = (self.mat.gamma - 1) * self.mat.C_v * self.T_old * self.rho_old

        # Internal energies (spatial cell centers)
        self.e = np.zeros(self.N)
        # Apply energy initial condition (Eq. 22)
        self.e_old = self.mat.C_v * np.copy(self.T_old)

        # Radiation energies (spatial cell centers)
        self.E = np.zeros(self.N)
        self.E_old = np.zeros(self.N)
        # Apply radiation initial condition (if given)
        self.initializeAtCenters(self.E_old, self.input.E)

        # User input boundary conditions ([0] is left value, [1] is right value)
        self.u_BC = self.input.u_BC
        self.P_BC = self.input.P_BC
        # Apply velocity boundary conditions
        if self.u_BC is not None:
            self.u_old[0] = self.u_BC[0]
            self.u_old[-1] = self.u_BC[1]
        # Apply pressure boundary conditions
        if self.P_BC is not None:
            self.P_old[0] = self.P_BC[0]
            self.P_old[-1] = self.P_BC[1]

    # Initialize variable with function at the spatial cell centers
    def initializeAtCenters(self, variable, function):
        if function is not None:
            for i in range(self.N):
                variable[i] = function(self.geo.r_old[i])

    # Initialize variable with function at the spatial cell edges
    def initializeAtEdges(self, variable, function):
        if function is not None:
            for i in range(self.N + 1):
                variable[i] = function(self.geo.r_half_old[i])

    # Recompute densities with newly updated volumes (rho = m / rho)
    def recomputeRho(self):
        np.copyto(self.rho_old, self.rho)
        np.divide(self.mat.m, self.geo.V, out=self.rho)

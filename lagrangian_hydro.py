import numpy as np

class LagrangianHydro:
    def __init__(self, rp):
        self.rp = rp
        self.input = rp.input
        self.geo = rp.geo
        self.mat = rp.mat
        self.rad = None

        # Velocities at cell edges
        self.u = np.zeros(self.geo.N + 1)
        self.u_old = np.zeros(self.geo.N + 1)

        # Temperature at cell edges
        self.T = np.zeros(self.geo.N + 1)
        self.T_old = np.zeros(self.geo.N + 1)
        # Apply temperature initial condition
        for i in range(self.geo.N + 1):
            self.T_old[i] = self.input.T(self.geo.r_half[i])

        # Pressure at cell edges
        self.P = np.zeros(self.geo.N + 1)
        self.P_old = np.zeros(self.geo.N + 1)
        # Apply pressure initial condition (Eqs. 22 and 23)
        for i in range(self.geo.N):
            self.P_old[i] = (self.mat.gamma[i] - 1) * self.mat.rho_old[i] * \
                            self.T_old[i] * self.mat.C_v[i]

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

    # Solve the velocity predictor (Eq. 14)
    def solveVelocityPredictor(self, dt):
        # Copy old velocity
        np.copyto(self.u_old, self.u)

        # No velocity BC given: cell loop includes boundary median mesh cells
        if self.u_BC is None:
            start, end = 0, self.geo.N
        # Velocity BC given: cell loop does not include boundary median mesh cells
        else:
            start, end = 1, self.geo.N - 1

        # Sweep to the right for each median mesh cell
        for i in range(start, end):
            coeff = -self.geo.A_old[i] * dt / self.mat.m_half[i]
            self.u[i] += coeff * (self.P_old[i + 1] - self.P_old[i])
            if self.input.enable_radiation:
                self.u[i] += coeff * (self.rad.E_old[i + 1] - self.rad.E_old[i]) / 3

    def solvePredictorStep(self, dt):
        # Solve velocity predictor (Eq. 14)
        self.solveVelocityPredictor(dt)

        # Move the mesh, which also updates A and V (Eq. 15)
        self.geo.moveMesh(u, u_old, dt)

        # Solve for new mass densities (Eq. 16)
        self.mat.recomputeRho()

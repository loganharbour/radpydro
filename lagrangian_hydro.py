import numpy as np

class LagrangianHydro:
    def __init__(self, rp):
        self.rp = rp
        self.input = rp.input
        self.geo = rp.geo
        self.N = self.geo.N
        self.mat = rp.mat
        self.fields = rp.fields

    # Solve the velocity predictor (Eq. 14)
    def solveVelocityPredictor(self, dt):
        u = self.fields.u
        u_old = self.fields.u_old
        P_old = self.fields.P_old
        E_old = self.fields.E_old

        # Copy old velocity
        np.copyto(u_old, u)

        # No velocity BC given: cell loop includes boundary median mesh cells
        if self.fields.u_BC is None:
            start, end = 0, self.N
        # Velocity BC given: cell loop does not include boundary median mesh cells
        else:
            start, end = 1, self.N - 1

        # Sweep to the right for each median mesh cell
        for i in range(start, end):
            coeff = -self.geo.A_old[i] * dt / self.mat.m_half[i]
            u[i] += coeff * (P_old[i + 1] - P_old[i])
            u[i] += coeff * (E_old[i + 1] - E_old[i]) / 3

    def solvePredictorStep(self, dt):
        # Solve velocity predictor (Eq. 14)
        self.solveVelocityPredictor(dt)

        # Move the mesh, which also updates A and V (Eq. 15)
        self.geo.moveMesh(self.fields.u, self.fields.u_old, dt)

        # Solve for new mass densities (Eq. 16)
        self.fields.recomputeRho()

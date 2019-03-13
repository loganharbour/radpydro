import numpy as np

class LagrangianHydro:
    def __init__(self, rp):
        self.rp = rp
        self.input = rp.input
        self.geo = rp.geo
        self.N = self.geo.N
        self.mat = rp.mat
        self.fields = rp.fields

        # Easy access to boundary conditions
        self.constrain_u = self.input.constrain_u
        self.P_BC = self.input.P_BC

    # Solve the velocity predictor (Eq. 14)
    def solveVelocityPredictor(self, dt):
        A_old = self.geo.A_old
        m_half = self.mat.m_half
        u = self.fields.u
        u_old = self.fields.u_old
        P_old = self.fields.P_old
        E_old = self.fields.E_old

        # Copy old velocity
        np.copyto(u_old, u)

        # Velocity BC, set u at left and u at right explicitly
        if self.constrain_u:
            u[0], u[-1] = self.fields.u_BC
        # Pressure BC, use Eqs. 37 and 38
        else:
            P_L, P_R = self.fields.P_BC
            E_L, E_R = self.computeE_BCs(self.fields.E_old)
            # Left cell
            coeff = -A_old[0] * dt / m_half[0]
            u[0] = u_old[0] + coeff * (P_old[0] - P_L + (E_old[0] - E_L) / 3)
            # Right cell
            coeff = -A_old[-1] * dt / m_half[-1]
            u[-1] = u_old[-1] + coeff * (P_old[-1] - P_R + (E_old[-1] - E_R) / 3)

        # Sweep to the right for each interior median mesh cell
        for i in range(1, self.N - 1):
            coeff = -A_old[i] * dt / m_half[i]
            u[i] = u_old[i] + coeff * (P_old[i + 1] - P_old[i])
            u[i] += coeff * (E_old[i + 1] - E_old[i]) / 3

    def solvePredictorStep(self, dt):
        # Solve velocity predictor (Eq. 14)
        self.solveVelocityPredictor(dt)

        # Move the mesh, which also updates A and V (Eq. 15)
        self.geo.moveMesh(self.fields.u, self.fields.u_old, dt)

        # Solve for new mass densities (Eq. 16)
        self.fields.recomputeRho()

    # Recompute surface intensity boundary conditions
    def computeE_BCs(self, E):
        kappa_t_old = self.mat.kappa_t_old
        rho_old = self.fields.rho_old
        r_half_old = self.geo.r_half_old

        # Reflective BC (use left and right values of E)
        if self.input.E_BC is None:
            E_L, E_R = [E[0], E[-1]]
        # Source BC (Use Eqs. 39 and 40)
        else:
            E_BC_L, E_BC_R = self.input.E_BC
            dr_L, dr_R = r_half_old[1], r_half_old[-1] - r_half_old[-2]
            E_L = 3 * rho_old[0] * dr_L * kappa_t_old[0] * E_BC_L + 4 * E[0]
            E_L /= 3 * rho_old[0] * dr_L * kappa_t_old[0] + 4
            E_R = 3 * rho_old[-1] * dr_R * kappa_t_old[-1] * E_BC_R + 4 * E[-1]
            E_R /= 3 * rho_old[-1] * dr_R * kappa_t_old[-1] + 4

        return E_L, E_R

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

    # Solve for velocities (Eqs. 14 and 24)
    def solveVelocity(self, dt, predictor):
        u_old = self.fields.u_old
        m_half = self.mat.m_half
        if predictor:
            A = self.geo.A_old
            P = self.fields.P_old
            E = self.fields.E_old
            u = self.fields.u_p
        else:
            A = (self.geo.A_old + self.geo.A_p) / 2
            P = (self.geo.P_old + self.geo.P_p) / 2
            E = (self.fields.E_old + self.fields.E_p) / 2
            u = self.fields.u

        # Velocity BC, set u at left and u at right explicitly
        if self.constrain_u:
            u[0], u[-1] = self.fields.u_BC
        # Pressure BC, use Eqs. 37 and 38
        else:
            P_L, P_R = self.fields.P_BC
            E_L, E_R = self.computeE_BCs(E)
            # Left cell
            u[0] = u_old[0] - A[0] * dt / m_half[0] * (P[0] - P_L + (E[0] - E_L) / 3)
            # Right cell
            u[-1] = u_old[-1] - A[-1] * dt / m_half[-1] * (P[-1] - P_R + (E[-1] - E_R) / 3)

        # Sweep to the right for each interior median mesh cell
        for i in range(1, self.N - 1):
            coeff = -A[i] * dt / m_half[i]
            u[i] = u_old[i] - A[i] * dt / m_half[i] * (P[i + 1] - P[i] \
                                                       + (E[i + 1] - E[i]) / 3)

    # Recompute surface intensity boundary conditions
    def computeE_BCs(self, predictor):
        if predictor:
            T = self.fields.T_old
            rho = self.fields.rho_old
            dr = self.geo.dr_old
            E = self.fields.E_old
        else:
            T = (self.fields.T_old + self.fields.T_p) / 2
            rho = (self.fields.rho_old + self.fields.rho_p) / 2
            dr = (self.fields.dr_old + self.fields.dr_p) / 2
            E = (self.fields.E_old + self.fields.E_p) / 2
        self.mat.recomputeKappa_t(T)

        # Boundary values of E ([0,0] for reflective, [val1,val2] for source)
        E_bL, E_bR = self.input.E_BC

        # E_1/2 and E_N+1/2 from Eqs. 39 and 40
        weight = 3 * rho[0] * dr[0] * kappa_t[0]
        E_L = (weight * E_bL + 4 * E[0]) / (weight + 4)
        weight = 3 * rho[-1] * dr[-1] * kappa_t[-1]
        E_R = (weight * E_br + 4 * E[-1]) / (weight + 4)

        return E_L, E_R

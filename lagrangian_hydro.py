import numpy as np

class LagrangianHydro:
    def __init__(self, rp):
        self.rp = rp
        self.input = rp.input
        self.geo = rp.geo
        self.N = self.geo.N
        self.mat = rp.mat
        self.fields = rp.fields

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
            P = (self.fields.P_old + self.fields.P_p) / 2
            E = (self.fields.E_old + self.fields.E_p) / 2
            u = self.fields.u

        # Compute these if a pressure BC exists
        if self.input.hydro_L == 'P' or self.input.hydro_R == 'P':
            E_L, E_R = self.computeE_BCs(predictor)

        # Velocity BC at left
        if self.input.hydro_L == 'u':
            u[0] = self.fields.u_L
        # Pressure BC at left (Eq. 37)
        else:
            P_L = self.fields.P_L
            coeff_L = A[0] * dt / m_half[0]
            u[0] = u_old[0] - coeff_L * (P[0] - P_L + (E[0] - E_L) / 3)
        # Velocity BC at right
        if self.input.hydro_R == 'u':
            u[-1] = self.fields.u_R
        # Pressure BC, use Eqs. 37 and 38
        else:
            P_R = self.fields.P_R
            coeff_R = A[-1] * dt / m_half[-1]
            u[-1] = u_old[-1] - coeff_R * (P_R - P[-1] + (E_R - E[-1]) / 3)

        # Sweep to the right for each interior median mesh cell
        for i in range(1, self.N):
            coeff = A[i] * dt / m_half[i]
            u[i] = u_old[i] - coeff * (P[i] - P[i-1] + (E[i] - E[i-1]) / 3)

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
            dr = (self.geo.dr_old + self.geo.dr_p) / 2
            E = (self.fields.E_old + self.fields.E_p) / 2
        self.mat.recomputeKappa_a(T)
        kappa_t_center = self.mat.kappa_a + self.mat.kappa_s

        # Reflective condition at left, get from E_1
        if self.fields.E_bL is None:
            E_bL = E[0]
        # Source condition at left
        else:
            E_bL = self.fields.E_bL
        # Reflective condition at right, get from E_N+1/2
        if self.fields.E_bR is None:
            E_bR = E[-1]
        # Source condition at right
        else:
            E_bR = self.fields.E_bR

        # E_1/2 and E_N+1/2 (Eqs. 39 and 40)
        weight = 3 * rho[0] * dr[0] * kappa_t_center[0]
        E_L = (weight * E_bL + 4 * E[0]) / (weight + 4)
        weight = 3 * rho[-1] * dr[-1] * kappa_t_center[-1]
        E_R = (weight * E_bR + 4 * E[-1]) / (weight + 4)

        return E_L, E_R

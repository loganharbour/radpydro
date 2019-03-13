import numpy as np
from scipy.sparse import spdiags


class LagrangianRadiation:
    def __init__(self, rp):
        self.rp = rp
        self.input = rp.input
        self.geo = rp.geo
        self.mat = rp.mat
        self.fields = rp.fields

    def assembleSystem(self, dt, predictor):

        m = self.mat.m
        a = self.mat.a
        c = self.mat.c
        C_v = self.mat.C_v
        N = self.geo.N

        rho_old = self.fields.rho_old
        dr_old = self.geo.dr_old
        u_old = self.fields.u_old
        A_old = self.geo.A_old
        E_old = self.fields.E_old
        T_old = self.fields.T_old
        p_old = self.fields.P_old

        if predictor:
            rho = self.fields.rho_p
            dr = self.geo.dr_p
            u = self.fields.u_p
            A = self.geo.A_p
        else:
            rho = self.fields.rho
            dr = self.geo.dr
            u = self.fields.u
            A = self.geo.A


        rho_k = (rho + rho_old) / 2
        dr_k = (dr + dr_old) / 2
        u_k = (u + u_old) / 2
        A_k = (A + A_old) / 2

        self.mat.recomputeKappa_t(T_old)
        self.mat.recomputeKappa_a(T_old)

        kappa_t = self.mat.kappa_t
        kappa_a = self.mat.kappa_a

        diag = np.zeros(self.geo.N)
        lowerdiag = np.zeros(self.geo.N)
        upperdiag = np.zeros(self.geo.N)
        rhs = np.zeros(self.geo.N)

        for i in range(1,N-1):
            diag[i] += m[i] / (dt * rho[i])
            diag[i] += - A_k[i+1] * c / (3 * (rho_k[i] * dr_k[i] * kappa_t[i+1] + rho_k[i+1] * dr_k[i+1] * kappa_t[i+1]))
            diag[i] += A_k[i] * c / (3 * (rho_k[i-1] * dr_k[i-1] * kappa_t[i] + rho_k[i] * dr_k[i] * kappa_t[i]))
            nu = 1 - (dt * kappa_a[i] * c * 2 * a * T_old[i]**3)/(C_v + dt * kappa_a[i] * c * 2 *a * T_old[i]**3)
            diag[i] += 0.5 * nu * m[i] * kappa_a[i] * c

        if self.input.E_BC is None:
            E_L, E_R = [E[0], E[-1]]
        







    def solveSystem(self):
        print("nothing yet")

    def recomputeInternalEnergy(self):
        print("nothing yet")

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
        lowerdiag = np.zeros(self.geo.N-1)
        upperdiag = np.zeros(self.geo.N-1)
        rhs = np.zeros(self.geo.N)

        if self.input.E_BC is None:
            nu = (dt * kappa_a[i] * c * 2 * a * T_old[0]**3)/(C_v + dt * kappa_a[0] * c * 2 *a * T_old[0]**3)
            xi = -P_old[0] * (A_old[1] * u_k[1] - A_old[0] * u_k[0])

            denom1 = 3 * (rho_k[0] * dr_k[0] * kappa_t[1] + rho_k[1] * dr_k[1] * kappa_t[1])

            diag[0] += m[0] / (dt * rho[0]) + A_k[1] * c / denom1    
            diag[0] += m[0] / 2 * (1 - nu) * c * kappa_a[i]

            upperdiag[0] = - A_k[1] * c / denom1

            rhs[0] += (- m[0] / (dt * rho_old[0])  \
                       - m[0] / 2 * kappa_a[0] * c * (1 - nu) \
                       - 1 / 3 * (A_old[1] * u_k[1] - A_old[0] * u_k[0]))*E_old[0] \
                       + nu * xi \
                       + A_k[1] * c / denom1 * (E_old[1] - E_old[0])
        else:
            nu = (dt * kappa_a[0] * c * 2 * a * T_old[0]**3)/(C_v + dt * kappa_a[0] * c * 2 *a * T_old[0]**3)
            xi = -P_old[0] * (A_old[1] * u_k[1] - A_old[0] * u_k[0])

            E_left = self.input.E_BC[0]
            T_left = ((1 / a * E_left + T_old[0]**4) / 2)**(1 / 4)
            kappa_left = self.mat.kappa_func(T_left) + self.kappa_s

            denom1 = 3 * (rho_k[0] * dr_k[0] * kappa_t[1] + rho_k[1] * dr_k[1] * kappa_t[1])
            denom2 = 3 * rho_k[0] * dr_k [0] * kappa_left + 4

            diag[0] += m[0] / (dt * rho[0]) + A_k[1] * c / denom1
            diag[0] += A_k[0] * c / denom2
            diag[0] += m[0] / 2 * (1 - nu) * c * kappa_a[0]

            upperdiag[0] = - A_k[1] * c / denom1

            rhs[0] += (- m[0] / (dt * rho_old[0])  \
                       - m[0] / 2 * kappa_a[0] * c * (1 - nu) \
                       - 1 / 3 * (A_old[1] * u_k[1] - A_old[0] * u_k[0]))*E_old[0] \
                       + nu * xi \
                       + c / denom1 * (E_old[1] - E_old[0]) \
                       - A_k[0] * 2 * c / denom2 * E_old[0] \
                       + A_k[0] * 2 * c / denom2 * E_left

        for i in range(1,N-1):
            nu = (dt * kappa_a[i] * c * 2 * a * T_old[i]**3)/(C_v + dt * kappa_a[i] * c * 2 *a * T_old[i]**3)
            xi = -P_old[i] * (A_old[i+1] * u_k[i+1] - A_old[i] * u_k[i])

            denom1 = 3 * (rho_k[i] * dr_k[i] * kappa_t[i+1] + rho_k[i+1] * dr_k[i+1] * kappa_t[i+1])
            denom2 = 3 * (rho_k[i-1] * dr_k[i-1] * kappa_t[i] + rho_k[i] * dr_k[i] * kappa_t[i])

            diag[i] += m[i] / (dt * rho[i]) + A_k[i+1] * c / denom1 + A_k[i] * c / denom2    
            diag[i] += m[i] / 2 * (1 - nu) * m[i] * c * kappa_a[i]

            upperdiag[i] = - A_k[i+1] * c / denom1
            lowerdiag[i-1] = - A_k[i] * c / denom2

            rhs[i] += (- m[i] / (dt * rho_old[i])  \
                       - m[i] / 2 * kappa_a[i] * c * (1 - nu) \
                       - 1 / 3 * (A_old[i+1] * u_k[i+1] - A_old[i] * u_k[i]))*E_old[i] \
                       + nu * xi \
                       + A_k[i+1] * c / denom1 * (E_old[i+1] - E_old[i]) \
                       - A_k[i] * c / denom2 * (E_old[i] - E_old[i-1])


        if self.input.E_BC is None:
            nu = (dt * kappa_a[N-1] * c * 2 * a * T_old[N-1]**3)/(C_v + dt * kappa_a[N-1] * c * 2 *a * T_old[N-1]**3)
            xi = -P_old[N-1] * (A_old[N] * u_k[N] - A_old[N-1] * u_k[N-1])

            denom2 = 3 * (rho_k[N-2] * dr_k[N-2] * kappa_t[N-1] + rho_k[N-1] * dr_k[N-1] * kappa_t[N-1])

            diag[N-1] += m[N-1] / (dt * rho[N-1]) + A_k[N-1] * c / denom2    
            diag[N-1] += m[N-1] / 2 * (1 - nu) * m[N-1] * c * kappa_a[N-1]

            lowerdiag[N-2] = - A_k[N-1] * c / denom2

            rhs[N-1] += (- m[N-1] / (dt * rho_old[N-1])  \
                       - m[N-1] / 2 * kappa_a[N-1] * c * (1 - nu) \
                       - 1 / 3 * (A_old[N] * u_k[N] - A_old[N-1] * u_k[N-1]))*E_old[N-1] \
                       + nu * xi \
                       - A_k[N-1] * c / denom2 * (E_old[N-1] - E_old[N-2])
        else:
            nu = (dt * kappa_a[N-1] * c * 2 * a * T_old[N-1]**3)/(C_v + dt * kappa_a[N-1] * c * 2 *a * T_old[N-1]**3)
            xi = -P_old[N-1] * (A_old[N] * u_k[N] - A_old[N-1] * u_k[N-1])

            E_right = self.input.E_BC[1]
            T_right = ((1 / a * E_right + T_old[N-1]**4) / 2)**(1 / 4)
            kappa_right = self.mat.kappa_func(T_right) + self.kappa_s

            denom1 = 3 * rho_k[N-1] * dr_k[N-1] * kappa_right + 4
            denom2 = 3 * (rho_k[N-2] * dr_k[N-2] * kappa_t[N-1] + rho_k[N-1] * dr_k[N-1] * kappa_t[N-1])

            diag[N-1] += m[N-1] / (dt * rho[N-1]) + A_k[N] * c / denom1 + A_k[N-1] * c / denom2     
            diag[N-1] += m[N-1] / 2 * (1 - nu) * m[N-1] * c * kappa_a[N-1]

            lowerdiag[N-2] = - A_k[N-1] * c / denom2

            rhs[N-1] += (- m[N-1] / (dt * rho_old[N-1])  \
                       - m[N-1] / 2 * kappa_a[N-1] * c * (1 - nu) \
                       - 1 / 3 * (A_old[N] * u_k[N] - A_old[N-1] * u_k[N-1]))*E_old[N-1] \
                       + nu * xi \
                       - A_k[N-1] * c / denom2 * (E_old[N-1] - E_old[N-2]) \
                       - A_k[N] * c / denom1 * (E_old[N-1]) \
                       + A_k[N] * 2 * c * E_right / denom1

        systemMatrix = diags([lowerdiag, diag, upperdiag], [-1, 0, 1])

    def solveSystem(self):
        print("nothing yet")

    def recomputeInternalEnergy(self):
        print("nothing yet")

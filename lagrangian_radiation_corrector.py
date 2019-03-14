import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve

class LagrangianRadiationCorrector:
    
    def __init__(self, rp):

        self.rp = rp
        self.input = rp.input
        self.geo = rp.geo
        self.mat = rp.mat
        self.fields = rp.fields

        self.diag = np.zeros(self.geo.N)
        self.lowerdiag = np.zeros(self.geo.N-1)
        self.upperdiag = np.zeros(self.geo.N-1)
        self.rhs = np.zeros(self.geo.N)

        self.rho_k = np.zeros(self.geo.N)
        self.dr_k = np.zeros(self.geo.N)
        self.u_k = np.zeros(self.geo.N + 1)
        self.A_k = np.zeros(self.geo.N + 1)

        self.A_pk = np.zeros(self.geo.N + 1)
        self.P_pk = np.zeros(self.geo.N)
        self.T_pk = np.zeros(self.geo.N)
        self.T3_pk = np.zeros(self.geo.N)
        self.T4_pk = np.zeros(self.geo.N)
        self.E_pk = np.zeros(self.goe.N)

        self.nu = np.zeros(self.geo.N)
        self.xi = np.zeros(self.geo.N)

    def assembleSystem(self, dt):

        self.reinitObjects()

        self.computeAuxiliaryFields(dt)

        self.assembleInnerCells(dt)

        self.applyLeftBoundary(dt)

        self.applyRightBoundary(dt)

    def reinitObjects(self):

        self.diag = np.zeros(self.geo.N)
        self.lowerdiag = np.zeros(self.geo.N-1)
        self.upperdiag = np.zeros(self.geo.N-1)
        self.rhs = np.zeros(self.geo.N)

    def computeAuxiliaryFields(self, dt):

        m = self.mat.m
        a = self.mat.a
        c = self.mat.c
        C_v = self.mat.C_v

        rho_old = self.fields.rho_old
        dr_old = self.geo.dr_old
        u_old = self.fields.u_old
        A_old = self.geo.A_old
        P_old = self.fields.P_old
        T_old = self.fields.T_old

        rho = self.fields.rho
        dr = self.geo.dr
        u = self.fields.u
        A = self.geo.A

        A_p = self.geo.A_p
        P_p = self.fields.P_p
        T_p = self.fields.T_p
        E_p = self.fields.E_p

        self.rho_k = (rho + rho_old) / 2
        self.dr_k = (dr + dr_old) / 2
        self.u_k = (u + u_old) / 2
        self.A_k = (A + A_old) / 2

        self.A_pk = (A_p + A_old) / 2
        self.P_pk = (P_p + P_old) / 2
        self.T_pk = (T_p + T_old) / 2
        self.T4_pk = ((T_p**4 + T_old**4) / 2)**(1 / 4)
        self.T3_pk = ((T_p**3 + T_old**3) / 2)**(1 / 3)
        self.E_pk = (E_p + E_old) / 2

        self.mat.recomputeKappa_t(self.T_pk)
        self.mat.recomputeKappa_a(self.T_pk)

        kappa_t = self.mat.kappa_t
        kappa_a = self.mat.kappa_a

        self.nu = dt * kappa_a * c * 2 * a * T3_pk**3
        self.nu /= C_v + dt * kappa_a * c * 2 *a * T3_pk**3

        for i in range(0, self.geo.N):
            self.xi[i] = - m[i] / dt * (self.fields.e_p[i] - self.fields.e_old[i])
            self.xi[i] += - self.P_pk[i] * (self.A_pk[i+1] * self.u_k[i+1] - self.A_pk[i] * self.u_k[i]) 

    def assembleInnerCells(self, dt):

        m = self.mat.m
        a = self.mat.a
        c = self.mat.c
        C_v = self.mat.C_v

        rho_old = self.fields.rho_old
        E_old = self.fields.E_old

        rho_k = self.rho_k
        dr_k = self.dr_k
        u_k = self.u_k
        A_k = self.A_k

        rho = self.fields.rho

        E_pk = self.E_pk
        T4_pk = self.T4_pk

        kappa_t = self.mat.kappa_t
        kappa_a = self.mat.kappa_a

        nu = self.nu
        xi = self.xi

        for i in range(1, N-1):

            denom1 = 3 * (rho_k[i] * dr_k[i] * kappa_t[i+1] + rho_k[i+1] * dr_k[i+1] * kappa_t[i+1])
            denom2 = 3 * (rho_k[i-1] * dr_k[i-1] * kappa_t[i] + rho_k[i] * dr_k[i] * kappa_t[i])

            self.diag[i] += m[i] / (dt * rho[i]) + A_k[i+1] * c / denom1 + A_k[i] * c / denom2    
            self.diag[i] += m[i] / 2 * (1 - nu[i]) * m[i] * c * kappa_a[i]

            self.upperdiag[i] = - A_k[i+1] * c / denom1
            self.lowerdiag[i-1] = - A_k[i] * c / denom2

            self.rhs[i] += (- m[i] / (dt * rho_old[i])  \
                       - m[i] / 2 * kappa_a[i] * c * (1 - nu[i])) * E_old[i]
            self.rhs[i] += - 1 / 3 * (A_pk[i+1] * u_k[i+1] - A_pk[i] * u_k[i]) * E_pk[i]
            self.rhs[i] += m[i] * kappa_a[i] * c * (1 - nu[i]) * a * T4_pk[0]**4
            self.rhs[i] += nu[i] * xi[i]
            self.rhs[i] += A_pk[i+1] * c / denom1 * (E_old[i+1] - E_old[i])
            self.rhs[i] += A_pk[i] * c / denom2 * (E_old[i] - E_old[i-1])

    def applyLeftBoundary(self, dt):

        m = self.mat.m
        a = self.mat.a
        c = self.mat.c
        C_v = self.mat.C_v

        rho_old = self.fields.rho_old
        E_old = self.fields.E_old

        rho_k = self.rho_k
        dr_k = self.dr_k
        u_k = self.u_k
        A_k = self.A_k

        rho = self.fields.rho

        E_pk = self.E_pk
        T4_pk = self.T4_pk

        kappa_t = self.mat.kappa_t
        kappa_a = self.mat.kappa_a

        nu = self.nu
        xi = self.xi

        denom1 = 3 * (rho_k[0] * dr_k[0] * kappa_t[1] + rho_k[1] * dr_k[1] * kappa_t[1])

        if self.input.E_BC is None:            

            self.diag[0] += m[0] / (dt * rho[0]) + A_k[1] * c / denom1    
            self.diag[0] += m[0] / 2 * (1 - nu[0]) * c * kappa_a[0]

            self.upperdiag[0] = - A_k[1] * c / denom1

            self.rhs[0] += (- m[0] / (dt * rho_old[0])  \
                            - m[0] / 2 * kappa_a[0] * c * (1 - nu[0])) * E_old[0]
            self.rhs[0] += - 1 / 3 * (A_pk[1] * u_k[1] - A_pk[0] * u_k[0]) * E_pk[0]
            self.rhs[0] += m[0] * kappa_a[0] * c * (1 - nu[0]) * a * T4_pk[0]**4
            self.rhs[0] += nu[0] * xi[0]
            self.rhs[0] += A_k[1] * c / denom1 * (E_old[1] - E_old[0])

        else:

            E_left = self.input.E_BC[0]
            T_left = ((1 / a * E_left + T_pk[0]**4) / 2)**(1 / 4)
            kappa_left = self.mat.kappa_func(T_left) + self.mat.kappa_s

            denom2 = 3 * rho_k[0] * dr_k [0] * kappa_left + 4

            self.diag[0] += m[0] / (dt * rho[0]) + A_k[1] * c / denom1
            self.diag[0] += A_k[0] * c / denom2
            self.diag[0] += m[0] / 2 * (1 - nu[0]) * c * kappa_a[0]

            self.upperdiag[0] = - A_k[1] * c / denom1

            self.rhs[0] += (- m[0] / (dt * rho_old[0])  \
                            - m[0] / 2 * kappa_a[0] * c * (1 - nu[0])) * E_old[0]
            self.rhs[0] += - 1 / 3 * (A_pk[1] * u_k[1] - A_pk[0] * u_k[0]) * E_pk[0]
            self.rhs[0] += m[0] * kappa_a[0] * c * (1 - nu[0]) * a * T4_pk[0]**4
            self.rhs[0] += nu[0] * xi[0]
            self.rhs[0] += c / denom1 * (E_old[1] - E_old[0]) 
            self.rhs[0] += - A_k[0] * c / denom2 * E_old[0] 
            self.rhs[0] += A_k[0] * 2 * c / denom2 * E_left

    def applyRightBoundary(self, dt):

        m = self.mat.m
        a = self.mat.a
        c = self.mat.c
        C_v = self.mat.C_v

        rho_old = self.fields.rho_old
        E_old = self.fields.E_old

        rho_k = self.rho_k
        dr_k = self.dr_k
        u_k = self.u_k
        A_k = self.A_k

        rho = self.fields.rho

        E_pk = self.E_pk
        T4_pk = self.T4_pk

        kappa_t = self.mat.kappa_t
        kappa_a = self.mat.kappa_a

        nu = self.nu
        xi = self.xi

        denom2 = 3 * (rho_k[N-2] * dr_k[N-2] * kappa_t[N-1] + rho_k[N-1] * dr_k[N-1] * kappa_t[N-1])

        if self.input.E_BC is None:
            
            self.diag[N-1] += m[N-1] / (dt * rho[N-1]) + A_k[N-1] * c / denom2    
            self.diag[N-1] += m[N-1] / 2 * (1 - nu[N-1]) * m[N-1] * c * kappa_a[N-1]

            self.lowerdiag[N-2] = - A_k[N-1] * c / denom2

            self.rhs[N-1] += (- m[N-1] / (dt * rho_old[N-1])  \
                              - m[N-1] / 2 * kappa_a[N-1] * c * (1 - nu[N-1])) * E_old[N-1]
            self.rhs[N-1] += - 1 / 3 * (A_pk[N] * u_k[N] - A_pk[N-1] * u_k[N-1]) * E_pk[N-1]
            self.rhs[N-1] += m[N-1] * kappa_a[N-1] * c * (1 - nu[N-1]) * a * T4_pk[N-1]**4
            self.rhs[N-1] += nu[N-1] * xi[N-1] 
            self.rhs[N-1] += - A_k[N-1] * c / denom2 * (E_old[N-1] - E_old[N-2])

        else:

            E_right = self.input.E_BC[1]
            T_right = ((1 / a * E_right + T_old[N-1]**4) / 2)**(1 / 4)
            kappa_right = self.mat.kappa_func(T_right) + self.mat.kappa_s

            denom1 = 3 * rho_k[N-1] * dr_k[N-1] * kappa_right + 4

            self.diag[N-1] += m[N-1] / (dt * rho[N-1]) + A_k[N] * c / denom1 + A_k[N-1] * c / denom2     
            self.diag[N-1] += m[N-1] / 2 * (1 - nu[N-1]) * m[N-1] * c * kappa_a[N-1]

            self.lowerdiag[N-2] = - A_k[N-1] * c / denom2

            self.rhs[N-1] += (- m[N-1] / (dt * rho_old[N-1])  \
                              - m[N-1] / 2 * kappa_a[N-1] * c * (1 - nu[N-1])) * E_old[N-1]
            self.rhs[N-1] += - 1 / 3 * (A_pk[N] * u_k[N] - A_pk[N-1] * u_k[N-1]) * E_pk[N-1]
            self.rhs[N-1] += nu[N-1] * xi[N-1]
            self.rhs[N-1] += - A_pk[N-1] * c / denom2 * (E_old[N-1] - E_old[N-2])
            self.rhs[N-1] += - A_pk[N] * c / denom1 * (E_old[N-1])
            self.rhs[N-1] += A_pk[N] * 2 * c * E_right / denom1

    def solveSystem(self):

        systemMatrix = diags([lowerdiag, diag, upperdiag], [-1, 0, 1])

        self.fields.E = spsolve(systemMatrix, rhs)

    def recomputeInternalEnergy(self, dt):

        m = self.mat.m
        a = self.mat.a
        c = self.mat.c
        C_v = self.mat.C_v

        T4_pk = self.T4_pk
        T_p = self.fields.T_p
        e_p = self.fields.e_p

        kappa_a = self.mat.kappa_a

        xi = self.xi

        E_k = 0.5 * (self.fields.E + self.fields.E_old)

        increment = dt * C_v * (m * kappa_a * c * (E_k - a * T4_pk**4) + xi)
        increment /= m*C_v + dt * m * kappa_a * c * 2 * a * T_p**3

        self.fields.e = e_p + increment


        


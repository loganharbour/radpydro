import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve


class LagrangianRadiation:
    def __init__(self, rp):
        self.rp = rp
        self.input = rp.input
        self.geo = rp.geo
        self.mat = rp.mat
        self.fields = rp.fields

        diag = np.zeros(self.geo.N)
        lowerdiag = np.zeros(self.geo.N-1)
        upperdiag = np.zeros(self.geo.N-1)
        rhs = np.zeros(self.geo.N)

        rho_k = np.zeros(self.geo.N)
        dr_k = np.zeros(self.geo.N)
        u_k = np.zeros(self.geo.N + 1)
        A_k = np.zeros(self.geo.N + 1)

        nu = np.zeros(self.geo.N)
        xi = np.zeros(self.geo.N)

    def assembleSystem(self, predictor, dt):

        self.mat.recomputeKappa_t(T_old)
        self.mat.recomputeKappa_a(T_old)

        computeAuxiliaryFields(predictor)

        self.assembleInnerCells(dt)

        applyLeftBoundary(dt)

        applyRightBoundary(dt)

    def computeAuxiliaryFields(self, dt, predictor):

        m = self.mat.m
        a = self.mat.a
        c = self.mat.c
        C_v = self.mat.C_v

        rho_old = self.fields.rho_old
        dr_old = self.geo.dr_old
        u_old = self.fields.u_old
        A_old = self.geo.A_old

        T_old = self.fields.T_old

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

        kappa_t = self.mat.kappa_t
        kappa_a = self.mat.kappa_a

        self.rho_k = (rho + rho_old) / 2
        self.dr_k = (dr + dr_old) / 2
        self.u_k = (u + u_old) / 2
        self.A_k = (A + A_old) / 2

        self.nu = dt * kappa_a * c * 2 * a * T_old**3
        self.nu /= C_v + dt * kappa_a * c * 2 *a * T_old**3

        for i in range(0, self.geo.N):
            self.xi[i] = -P_old[i] * (A_old[i+1] * self.u_k[i+1] - A_old[i] * self.u_k[i]) 

    def assembleInnerCells(self, dt):

        m = self.mat.m
        a = self.mat.a
        c = self.mat.c
        C_v = self.mat.C_v

        rho_old = self.fields.rho_old
        A_old = self.geo.A_old
        E_old = self.fields.E_old
        T_old = self.fields.T_old
        p_old = self.fields.P_old

        rho_k = self.rho_k
        dr_k = self.dr_k
        u_k = self.u_k
        A_k = self.A_k

        self.mat.recomputeKappa_t(T_old)
        self.mat.recomputeKappa_a(T_old)

        kappa_t = self.mat.kappa_t
        kappa_a = self.mat.kappa_a

        nu = self.nu
        xi = self.xi

        for i in range(1, N-1):

            denom1 = 3 * (rho_k[i] * dr_k[i] * kappa_t[i+1] + rho_k[i+1] * dr_k[i+1] * kappa_t[i+1])
            denom2 = 3 * (rho_k[i-1] * dr_k[i-1] * kappa_t[i] + rho_k[i] * dr_k[i] * kappa_t[i])

            diag[i] += m[i] / (dt * rho[i]) + A_k[i+1] * c / denom1 + A_k[i] * c / denom2    
            diag[i] += m[i] / 2 * (1 - nu[i]) * m[i] * c * kappa_a[i]

            upperdiag[i] = - A_k[i+1] * c / denom1
            lowerdiag[i-1] = - A_k[i] * c / denom2

            rhs[i] += (- m[i] / (dt * rho_old[i])  \
                       - m[i] / 2 * kappa_a[i] * c * (1 - nu[i]) \
                       - 1 / 3 * (A_old[i+1] * u_k[i+1] - A_old[i] * u_k[i]))*E_old[i]
            rhs[i] +=  nu[i] * xi[i]
            rhs[i] +=  A_k[i+1] * c / denom1 * (E_old[i+1] - E_old[i])
            rhs[i] +=  A_k[i] * c / denom2 * (E_old[i] - E_old[i-1])

    def applyLeftBoundary(self, dt):

        m = self.mat.m
        a = self.mat.a
        c = self.mat.c
        C_v = self.mat.C_v

        rho_old = self.fields.rho_old
        A_old = self.geo.A_old
        E_old = self.fields.E_old
        T_old = self.fields.T_old
        p_old = self.fields.P_old

        rho_k = self.rho_k
        dr_k = self.dr_k
        u_k = self.u_k
        A_k = self.A_k

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
                           - m[0] / 2 * kappa_a[0] * c * (1 - nu[0]) \
                           - 1 / 3 * (A_old[1] * u_k[1] - A_old[0] * u_k[0]))*E_old[0]
            self.rhs[0] += nu[0] * xi[0]
            self.rhs[0] += A_k[1] * c / denom1 * (E_old[1] - E_old[0])

        else:

            E_left = self.input.E_BC[0]
            T_left = ((1 / a * E_left + T_old[0]**4) / 2)**(1 / 4)
            kappa_left = self.mat.kappa_func(T_left) + self.kappa_s

            denom2 = 3 * rho_k[0] * dr_k [0] * kappa_left + 4

            self.diag[0] += m[0] / (dt * rho[0]) + A_k[1] * c / denom1
            self.diag[0] += A_k[0] * c / denom2
            self.diag[0] += m[0] / 2 * (1 - nu[0]) * c * kappa_a[0]

            self.upperdiag[0] = - A_k[1] * c / denom1

            self.rhs[0] += (- m[0] / (dt * rho_old[0])  \
                            - m[0] / 2 * kappa_a[0] * c * (1 - nu[0]) \
                            - 1 / 3 * (A_old[1] * u_k[1] - A_old[0] * u_k[0]))*E_old[0]
            self.rhs[0] += nu[0] * xi[0] \
            self.rhs[0] += c / denom1 * (E_old[1] - E_old[0]) \
            self.rhs[0] += - A_k[0] * 2 * c / denom2 * E_old[0] \
            self.rhs[0] += A_k[0] * 2 * c / denom2 * E_left

    def applyRightBoundary(self, dt):

        m = self.mat.m
        a = self.mat.a
        c = self.mat.c
        C_v = self.mat.C_v

        rho_old = self.fields.rho_old
        A_old = self.geo.A_old
        E_old = self.fields.E_old
        T_old = self.fields.T_old
        p_old = self.fields.P_old

        rho_k = self.rho_k
        dr_k = self.dr_k
        u_k = self.u_k
        A_k = self.A_k

        kappa_t = self.mat.kappa_t
        kappa_a = self.mat.kappa_a

        nu = self.nu
        xi = self.xi

        denom2 = 3 * (rho_k[N-2] * dr_k[N-2] * kappa_t[N-1] + rho_k[N-1] * dr_k[N-1] * kappa_t[N-1])

        if self.input.E_BC is None:
            
            diag[N-1] += m[N-1] / (dt * rho[N-1]) + A_k[N-1] * c / denom2    
            diag[N-1] += m[N-1] / 2 * (1 - nu[N-1]) * m[N-1] * c * kappa_a[N-1]

            lowerdiag[N-2] = - A_k[N-1] * c / denom2

            rhs[N-1] += (- m[N-1] / (dt * rho_old[N-1])  \
                        - m[N-1] / 2 * kappa_a[N-1] * c * (1 - nu[N-1]) \
                        - 1 / 3 * (A_old[N] * u_k[N] - A_old[N-1] * u_k[N-1]))*E_old[N-1]
            rhs[N-1] += nu[N-1] * xi[N-1] 
            rhs[N-1] += - A_k[N-1] * c / denom2 * (E_old[N-1] - E_old[N-2])
        else:

            E_right = self.input.E_BC[1]
            T_right = ((1 / a * E_right + T_old[N-1]**4) / 2)**(1 / 4)
            kappa_right = self.mat.kappa_func(T_right) + self.kappa_s

            denom1 = 3 * rho_k[N-1] * dr_k[N-1] * kappa_right + 4

            diag[N-1] += m[N-1] / (dt * rho[N-1]) + A_k[N] * c / denom1 + A_k[N-1] * c / denom2     
            diag[N-1] += m[N-1] / 2 * (1 - nu[N-1]) * m[N-1] * c * kappa_a[N-1]

            lowerdiag[N-2] = - A_k[N-1] * c / denom2

            rhs[N-1] += (- m[N-1] / (dt * rho_old[N-1])  \
                       - m[N-1] / 2 * kappa_a[N-1] * c * (1 - nu[N-1]) \
                       - 1 / 3 * (A_old[N] * u_k[N] - A_old[N-1] * u_k[N-1]))*E_old[N-1]
            rhs[N-1] += nu[N-1] * xi[N-1]
            rhs[N-1] += - A_k[N-1] * c / denom2 * (E_old[N-1] - E_old[N-2])
            rhs[N-1] += - A_k[N] * c / denom1 * (E_old[N-1])
            rhs[N-1] += A_k[N] * 2 * c * E_right / denom1

    def solveSystem(self, predictor):

        systemMatrix = diags([lowerdiag, diag, upperdiag], [-1, 0, 1])

        if predictor:
            self.fields.E_p = spsolve(systemMatrix, rhs)
        else:
            self.fields.E = spsolve(systemMatrix, rhs)

    def recomputeInternalEnergy(self, dt, predictor):

        m = self.mat.m
        a = self.mat.a
        c = self.mat.c
        C_v = self.mat.C_v

        T_old = self.fields.T_old
        e_old = self.fields.e_old

        kappa_a = self.mat.kappa_a

        xi = self.xi

        if predictor:
            E_k = 0.5 * (self.fields.E_p + self.fields.E_old)
        else:
            E_k = 0.5 * (self.fields.E + self.fields.E_old)

        increment = dt * C_v * (m * kappa_a * c * (E_k - a * T_old**4) + xi)
        increment /= m*C_v + dt * m * kappa_a * c * 2 * a * T_old**3

        if predictor:
            self.fields.e_p = e_old + increment
        else:
            self.fields.e = e_old + increment


        


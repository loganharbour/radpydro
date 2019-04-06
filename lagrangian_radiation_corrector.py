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

        # Init containers for linear system
        self.diag = np.zeros(self.geo.N)
        self.lowerdiag = np.zeros(self.geo.N)
        self.upperdiag = np.zeros(self.geo.N)
        self.rhs = np.zeros(self.geo.N)

        # Init containers for k'th time-step quantities
        self.rho_k = np.zeros(self.geo.N)
        self.dr_k = np.zeros(self.geo.N)
        self.u_k = np.zeros(self.geo.N + 1)
        self.A_k = np.zeros(self.geo.N + 1)

        # Init containers for predicted k'th time-step quantities
        self.A_pk = np.zeros(self.geo.N + 1)
        self.P_pk = np.zeros(self.geo.N)
        self.T_pk = np.zeros(self.geo.N)
        self.E_pk = np.zeros(self.geo.N)

        # Init containers for parameters used in linear system
        self.nu = np.zeros(self.geo.N)
        self.xi = np.zeros(self.geo.N)

    def computeAuxiliaryFields(self, dt):
        # Constants
        m = self.mat.m
        a = self.input.a
        c = self.input.c
        C_v = self.mat.C_v

        # k-1/2'th time step quantities
        rho_old = self.fields.rho_old
        dr_old = self.geo.dr_old
        u_old = self.fields.u_old
        A_old = self.geo.A_old
        P_old = self.fields.P_old
        T_old = self.fields.T_old
        E_old = self.fields.E_old

        # k+1/2'th time-step quantities
        rho = self.fields.rho
        dr = self.geo.dr
        u = self.fields.u
        A = self.geo.A

        # Predicted k_1/2'th time-step quantities
        A_p = self.geo.A_p
        P_p = self.fields.P_p
        T_p = self.fields.T_p
        E_p = self.fields.E_p

        # Compute k'th time-step quantities
        self.rho_k = (rho + rho_old) / 2
        self.dr_k = (dr + dr_old) / 2
        self.u_k = (u + u_old) / 2
        self.A_k = (A + A_old) / 2

        # Compute predicted k'th time-step quantities
        self.A_pk = (A_p + A_old) / 2
        self.P_pk = (P_p + P_old) / 2
        self.T_pk = (T_p + T_old) / 2
        self.E_pk = (E_p + E_old) / 2

        # Compute opacities at predicted k'th time-step temperatures
        self.mat.recomputeKappa_t(self.T_pk)
        self.mat.recomputeKappa_a(self.T_pk)
        kappa_t = self.mat.kappa_t
        kappa_a = self.mat.kappa_a

        # Compute parameters for linear system
        weight = 2 * dt * kappa_a * a * c
        self.nu = weight * T_p**3 / (C_v + weight * T_p**3)
        for i in range(0, self.geo.N):
            self.xi[i] = -m[i] / dt * (self.fields.e_p[i] - self.fields.e_old[i])
            self.xi[i] -= self.P_pk[i] * (self.A_pk[i+1] * self.u_k[i+1] - self.A_pk[i] * self.u_k[i])

    def assembleSystem(self, dt):
        # Constants
        m = self.mat.m
        a = self.input.a
        c = self.input.c
        C_v = self.mat.C_v
        N = self.geo.N

        # k-1/2'th time-step quantities
        rho_old = self.fields.rho_old
        E_old = self.fields.E_old
        T_old = self.fields.T_old

        # k'th time-step quantities
        rho_k = self.rho_k
        dr_k = self.dr_k
        u_k = self.u_k
        A_k = self.A_k

        # k+1/2'th time-step density
        rho = self.fields.rho

        # Predicted k+1/2'th time-step temperature
        T_p = self.fields.T_p

        # Predicted k'th time-step quantities
        A_pk = self.A_pk
        E_pk = self.E_pk

        # Opacities evaluated at predicted k'th time-step temperatures
        kappa_t = self.mat.kappa_t
        kappa_a = self.mat.kappa_a

        # Parameters for linear system
        nu = self.nu
        xi = self.xi

        for i in range(N):
            # Time derivative contributions
            self.diag[i] += m[i] / dt * (1 / rho[i])
            self.rhs[i] += m[i] / dt * (E_old[i] / rho_old[i])

            # Absorption contributions
            self.diag[i] += m[i] * kappa_a[i] * c * (1 - nu[i]) / 2
            self.rhs[i] -= m[i] * kappa_a[i] * c * (1 - nu[i]) * E_old[i] / 2

            # Material emmision and nu * xi parameter contributions
            self.rhs[i] += m[i] * kappa_a[i] * c * (1 - nu[i]) * a * (T_p[i]**4 + T_old[i]**4) / 2
            self.rhs[i] += nu[i] * xi[i]

            # Drift term contributions
            self.rhs[i] -= 1/3 * E_pk[i] * (A_pk[i+1] * u_k[i+1] - A_pk[i] * u_k[i])

            # If past leftmost cell
            if i > 0:
                # left edge flux coefficient
                coeff_F_L = -2 * c / (3 * (rho_k[i-1] * dr_k[i-1] * kappa_t[i] \
                                           + rho_k[i] * dr_k[i] * kappa_t[i]))

                # left edge flux contributions
                self.diag[i] -= A_k[i] * coeff_F_L / 2
                self.lowerdiag[i-1] += A_k[i] * coeff_F_L / 2
                self.rhs[i] += A_k[i] * coeff_F_L * E_old[i] / 2
                self.rhs[i] -= A_k[i] * coeff_F_L * E_old[i-1] / 2

            # if before rightmost cell
            if i < N-1:
                # right edge flux coefficient
                coeff_F_R = -2 * c / (3 * (rho_k[i] * dr_k[i] * kappa_t[i+1] \
                                          + rho_k[i+1] * dr_k[i+1] * kappa_t[i+1]))

                # right edge flux contributions
                self.diag[i] -= A_k[i+1] * coeff_F_R / 2
                self.upperdiag[i+1] += A_k[i+1] * coeff_F_R / 2
                self.rhs[i] += A_k[i+1] * coeff_F_R * E_old[i] / 2
                self.rhs[i] -= A_k[i+1] * coeff_F_R * E_old[i+1] / 2

        # Left BC handling
        if self.input.rad_L is 'source':
            coeff_F_bL = - 2 * c / (3 * rho_k[0] * dr_k[0] * kappa_t[0] + 4)
            self.diag[0] -= A_k[0] * coeff_F_bL / 2
            self.rhs[0] += A_k[0] * coeff_F_bL * E_old[0] / 2
            self.rhs[0] -= A_k[0] * coeff_F_bL * self.fields.E_bL

        # Right BC handline
        if self.input.rad_R is 'source':
            coeff_F_bR = - 2 * c / (3 * rho_k[-1] * dr_k[-1] * kappa_t[-1] + 4)
            self.diag[-1] -= A_k[-1] * coeff_F_bR / 2
            self.rhs[-1] += A_k[-1] * coeff_F_bR * E_old[-1] / 2
            self.rhs[-1] -= A_k[-1] * coeff_F_bR * self.fields.E_bR 

    def solveSystem(self, dt):
        self.computeAuxiliaryFields(dt)
        self.assembleSystem(dt)

        data = np.array([self.lowerdiag, self.diag, self.upperdiag])
        diags = np.array([-1, 0, 1])

        systemMatrix = spdiags(data, diags, self.geo.N, self.geo.N, format = 'csr')

        self.fields.E = spsolve(systemMatrix, self.rhs)

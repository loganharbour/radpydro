import numpy as np
from scipy.sparse import diags, spdiags
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve_banded

class LagrangianRadiationPredictor:

    def __init__(self, rp):
        self.rp = rp
        self.input = rp.input
        self.geo = rp.geo
        self.mat = rp.mat
        self.fields = rp.fields

        # init containers for linear system
        self.diag = np.zeros(self.geo.N)
        self.lowerdiag = np.zeros(self.geo.N)
        self.upperdiag = np.zeros(self.geo.N)
        self.rhs = np.zeros(self.geo.N)

        # init containers for k'th time-step variables
        self.rho_pk = np.zeros(self.geo.N)
        self.dr_pk = np.zeros(self.geo.N)
        self.u_pk = np.zeros(self.geo.N + 1)
        self.A_pk = np.zeros(self.geo.N + 1)

        # additional parameters
        self.nu = np.zeros(self.geo.N)
        self.xi = np.zeros(self.geo.N)

    def computeAuxiliaryFields(self, dt):
        a = self.input.a
        c = self.input.c
        C_v = self.mat.C_v

        m = self.mat.m
        rho_old = self.fields.rho_old
        dr_old = self.geo.dr_old
        u_old = self.fields.u_old
        A_old = self.geo.A_old
        P_old = self.fields.P_old
        T_old = self.fields.T_old

        rho_p = self.fields.rho_p
        dr_p = self.geo.dr_p
        u_p = self.fields.u_p
        A_p = self.geo.A_p

        # compute k'th time-step variables
        self.rho_pk = (rho_p + rho_old) / 2
        self.dr_pk = (dr_p + dr_old) / 2
        self.u_pk = (u_p + u_old) / 2
        self.A_pk = (A_p + A_old) / 2

        # recompute opacities at old temperatures
        self.mat.recomputeKappa_t(T_old)
        self.mat.recomputeKappa_a(T_old)

        kappa_t = self.mat.kappa_t
        kappa_a = self.mat.kappa_a

        # parameter in radiation energy system (Eq. 20a)
        weight = 2 * a * c * dt * kappa_a
        self.nu = weight * T_old**3 / (C_v + weight * T_old**3)

        # parameter in radiation energy system (Eq. 20b)
        for i in range(0, self.geo.N):
            self.xi[i] = - P_old[i] * (A_old[i+1] * self.u_pk[i+1] - A_old[i] * self.u_pk[i])

    def assembleSystem(self, dt):
        N = self.geo.N
        a = self.input.a
        c = self.input.c
        C_v = self.mat.C_v

        m = self.mat.m
        rho_old = self.fields.rho_old
        A_old = self.geo.A_old
        E_old = self.fields.E_old
        T_old = self.fields.T_old

        rho_pk = self.rho_pk
        dr_pk = self.dr_pk
        u_pk = self.u_pk
        A_pk = self.A_pk

        rho_p = self.fields.rho_p

        kappa_t = self.mat.kappa_t
        kappa_a = self.mat.kappa_a

        nu = self.nu
        xi = self.xi

        for i in range(N):

            # Time derivative contributions
            self.diag[i] += m[i] / dt * (1 / rho_p[i])
            self.rhs[i] += m[i] / dt * (E_old[i] / rho_old[i])

            # Absorption contributions
            self.diag[i] += m[i] * kappa_a[i] * c * (1 - nu[i]) / 2
            self.rhs[i] -= m[i] * kappa_a[i] * c * (1 - nu[i]) * E_old[i] / 2

            # Material emmision and nu * xi contributions
            self.rhs[i] += m[i] * kappa_a[i] * c * (1 - nu[i]) * a * T_old[i]**4
            self.rhs[i] += nu[i] * xi[i]

            # Drift term contributions
            self.rhs[i] -= 1/3 * E_old[i] * (A_old[i+1] * u_pk[i+1] - A_old[i] * u_pk[i])

            # if past leftmost cell
            if i > 0:
                # left edge flux coefficient
                coeff_F_L = -2 * c / (3 * (rho_pk[i-1] * dr_pk[i-1] * kappa_t[i] \
                                           + rho_pk[i] * dr_pk[i] * kappa_t[i]))

                # left edge flux contributions
                self.diag[i] -= A_pk[i] * coeff_F_L / 2
                self.lowerdiag[i-1] += A_pk[i] * coeff_F_L / 2
                self.rhs[i] += A_pk[i] * coeff_F_L * E_old[i] / 2
                self.rhs[i] -= A_pk[i] * coeff_F_L * E_old[i-1] / 2

            # if before rightmost cell
            if i < N-1:
                # right edge flux coefficient
                coeff_F_R = -2 * c / (3 * (rho_pk[i] * dr_pk[i] * kappa_t[i+1] \
                                          + rho_pk[i+1] * dr_pk[i+1] * kappa_t[i+1]))

                # right edge flux contributions
                self.diag[i] -= A_pk[i+1] * coeff_F_R / 2
                self.upperdiag[i+1] += A_pk[i+1] * coeff_F_R / 2
                self.rhs[i] += A_pk[i+1] * coeff_F_R * E_old[i] / 2
                self.rhs[i] -= A_pk[i+1] * coeff_F_R * E_old[i+1] / 2

        # Left BC handling
        if self.input.rad_L is 'source':
            coeff_F_L = -2 * c / (3 * rho_pk[0] * dr_pk[0] * kappa_t[0] + 4)
            self.diag[0] -= A_pk[0] * coeff_F_L / 2
            self.rhs[0] += A_pk[0] * coeff_F_L / 2 * E_old[0]
            self.rhs[0] -= A_pk[0] * coeff_F_L * self.fields.E_bL

        # Right BC handline
        if self.input.rad_R is 'source':
            coeff_F_R = -2 * c / (3 * rho_pk[-1] * dr_pk[-1] * kappa_t[-1] + 4)
            self.diag[-1] -= A_pk[-1] * coeff_F_R / 2
            self.rhs[-1] += A_pk[-1] * coeff_F_R * E_old[-1] / 2
            self.rhs[-1] -= A_pk[-1] * coeff_F_R * self.fields.E_bR

    def solveSystem(self, dt):
        self.computeAuxiliaryFields(dt)
        self.assembleSystem(dt)

        data = np.array([self.lowerdiag, self.diag, self.upperdiag])
        diags = np.array([-1, 0, 1])

        systemMatrix = spdiags(data, diags, self.geo.N, self.geo.N, format = 'csr')

        self.fields.E_p = spsolve(systemMatrix, self.rhs)

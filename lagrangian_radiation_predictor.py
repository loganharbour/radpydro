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
        self.nu = dt * kappa_a * c * 2 * a * T_old**3
        self.nu /= C_v + dt * kappa_a * c * 2 *a * T_old**3

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

        for i in range(0, N):
            # time derivative contributions
            self.diag[i] += m[i] / (dt * rho_p[i])
            self.rhs[i] += m[i] / (dt * rho_p[i]) * E_old[i]

            # absorption contributions
            self.diag[i] += m[i] * kappa_a[i] * c * (1 - nu[i]) / 2
            self.rhs[i] -= m[i] * kappa_a[i] * c * (1 - nu[i]) * E_old[i] / 2

            # material emmision and nu * xi contributions
            self.rhs[i] += m[i] * kappa_a[i] * c * (1 - nu[i]) * a * T_old[i]**4
            self.rhs[i] += nu[i] * xi[i]

            # drift term contributions
            self.rhs[i] -= 1/3 * E_old[i] * (A_pk[i+1] * u_pk[i+1] \
                                             - A_pk[i] * u_pk[i])

            # if past leftmost cell
            if (i > 0):
                # left edge flux coefficient
                coeff_F_L = 2 * c / (3 * (rho_pk[i-1] * dr_pk[i-1] * kappa_t[i] \
                                          + rho_pk[i] * dr_pk[i] * kappa_t[i]))

                # left edge flux contributions
                self.diag[i] += A_pk[i] * coeff_F_L / 2
                self.lowerdiag[i-1] -= A_pk[i] * coeff_F_L / 2
                self.rhs[i] -= A_pk[i] * coeff_F_L * E_old[i] / 2
                self.rhs[i-1] += A_pk[i] * coeff_F_L * E_old[i-1] / 2

            # if before rightmost cell
            if (i < N-1):
                # right edge flux coefficient
                coeff_F_R = 2 * c / (3 * (rho_pk[i] * dr_pk[i] * kappa_t[i+1] \
                                          + rho_pk[i+1] * dr_pk[i+1] * kappa_t[i+1]))

                # right edge flux contributions
                self.diag[i] += A_pk[i+1] * coeff_F_R / 2
                self.upperdiag[i+1] -= A_pk[i+1] * coeff_F_R / 2
                self.rhs[i] -= A_pk[i+1] * coeff_F_R * E_old[i] / 2
                self.rhs[i+1] += A_pk[i+1] * coeff_F_R * E_old[i+1] / 2

            # if on left boundary cell
            if (i == 0):
                if self.input.rad_L is 'source':
                    E_L = self.input.rad_L_val

                    # compute left bdry temp with Eq. 34
                    T_L = ((E_L / a + T_old[i]**4) / 2)**(1/4)
                    # compute left bdry total opacity with T_L
                    kappa_t_L = self.mat.kappa_func(T_L) + self.mat.kappa_s

                    # left bdry flux coefficient
                    coeff_F_Lb = 2 * c / (3 * rho_pk[i] * dr_pk[i] * kappa_t_L + 4)

                    # left bdry flux contributions
                    self.diag[i] += A_pk[i] * coeff_F_Lb / 2
                    self.rhs[i] += A_pk[i] * coeff_F_Lb * E_L / 2

                elif self.input.rad_L is 'reflective':
                    pass

            # if on right boundary cell
            if (i == N-1):
                    if self.input.rad_R is 'source':
                        E_R = self.input.rad_R_val

                        # compute right bdry temp with Eq. 36
                        T_R = ((E_R / a + T_old[i]**4) / 2)**(1/4)
                        # compute right bdry total opacity with T_R
                        kappa_t_R = self.mat.kappa_func(T_R) + self.mat.kappa_s

                        # right bdry flux coefficient
                        coeff_F_Rb = 2 * c / (3 * rho_pk[i] * dr_pk[i] * kappa_t_R + 4)

                        #right bdry flux contributions
                        self.diag[i] += A_pk[i+1] * coeff_F_Rb / 2
                        self.rhs[i] += A_pk[i+1] * coeff_F_Rb * E_R / 2
                        print('\n', A_pk[i+1] * coeff_F_Rb * E_R / 2)

                    elif self.input.rad_R is 'reflective':
                        pass

    def solveSystem(self, dt):

        self.computeAuxiliaryFields(dt)
        self.assembleSystem(dt)

        data = np.array([self.lowerdiag, self.diag, self.upperdiag])
        diags = np.array([-1, 0, 1])

        systemMatrix = spdiags(data, diags, self.geo.N, self.geo.N, format = 'csr')
        self.fields.E_p = spsolve(systemMatrix, self.rhs)

        self.recomputeInternalEnergy(dt)

    def recomputeInternalEnergy(self, dt):

        m = self.mat.m
        a = self.input.a
        c = self.input.c
        C_v = self.mat.C_v

        T_old = self.fields.T_old
        e_old = self.fields.e_old

        kappa_a = self.mat.kappa_a

        xi = self.xi

        E_k = (self.fields.E_p + self.fields.E_old) / 2

        increment = dt * C_v * (m * kappa_a * c * (E_k - a * T_old**4) + xi)
        increment /= m * C_v + dt * m * kappa_a * c * 2 * a * T_old**3

        self.fields.e_p = e_old + increment

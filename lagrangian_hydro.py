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
    def recomputeVelocity(self, predictor):
        u_old = self.fields.u_old
        m_half = self.mat.m_half
        dt = self.rp.timeSteps[-1]
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

    # Recmpute density with updated cell volumes
    def recomputeDensity(self, predictor):
        m = self.mat.m
        if predictor:
            rho_new = self.fields.rho_p
            V_new = self.geo.V_p
        else:
            rho_new = self.fields.rho
            V_new = self.geo.V
        for i in range(self.geo.N):
            rho_new[i] = m[i] / V_new[i]

    # Recompute radiation energy with updated internal energy
    def recomputeInternalEnergy(self, predictor):
        # Constants
        a = self.input.a
        c = self.input.c
        C_v = self.mat.C_v
        m = self.mat.m
        dt = self.rp.timeSteps[-1]
        # Predictor step routine
        if predictor:
            e_old = self.fields.e_old

            # If running a rad-hydro problem
            if self.input.enable_radiation:
                T_old = self.fields.T_old
                E_pk = (self.fields.E_p + self.fields.E_old) / 2
                xi_old = self.rp.radPredictor.xi
                self.mat.recomputeKappa_a(T_old)
                kappa_a_old = self.mat.kappa_a

                increment = dt * C_v * (m * kappa_a_old * c * (E_pk - a * T_old**4) + xi_old)
                increment /= m * C_v + dt * m * kappa_a_old * c * 2 * a * T_old**3

            # If running a purely hydro problem
            else:
                 P_old = self.fields.P_old
                 A_old = self.geo.A_old
                 u_pk  = (self.fields.u_old + self.fields.u_p) / 2

                 xi_old = np.zeros(self.geo.N)
                 for i in range(self.geo.N):
                     xi_old[i] = -P_old[i] * (A_old[i+1] * u_pk[i+1] - A_old[i] * u_pk[i])

                 increment = dt / m * xi_old

            self.fields.e_p = e_old + increment

        # If corrector step
        else:
            e_p = self.fields.e_p

            # If running a rad-hydro problem
            if self.input.enable_radiation:
                T_old = self.fields.T_old
                T_p = self.fields.T_p
                T_pk = (T_p + T_old) / 2
                T_pk4 = (T_p**4 + T_old**4) / 2
                E_k = (self.fields.E + self.fields.E_old) / 2
                self.mat.recomputeKappa_a(T_pk)
                kappa_a_pk = self.mat.kappa_a
                xi_k = self.rp.radCorrector.xi

                increment = dt * C_v * (m * kappa_a_pk * c * (E_k - a * T_pk4) + xi_k)
                increment /= m * C_v + dt * m * kappa_a_pk * c * 2 * a * T_p**3

            # If running a purely hydro problem
            else:
                e_old = self.fields.e_old
                P_pk  = (self.fields.P_old + self.fields.P_p) / 2
                A_pk  = (self.geo.A_old + self.geo.A_p) / 2
                u_k   = (self.fields.u + self.fields.u_old) / 2

                xi_k = np.zeros(self.geo.N)
                for i in range(self.geo.N):
                    xi_k[i]  = -(m[i] / dt) * (e_p[i] - e_old[i])
                    xi_k[i] -= P_pk[i] * (A_pk[i+1] * u_k[i+1] - A_pk[i] * u_k[i])

                increment = dt / m * xi_k

            self.fields.e = e_p + increment

    # Recompute temperature with updated internal energy
    def recomputeTemperature(self, predictor):
        C_v = self.mat.C_v
        if predictor:
            T_new = self.fields.T_p
            e_new = self.fields.e_p
        else:
            T_new = self.fields.T
            e_new = self.fields.e
        for i in range(self.geo.N):
            T_new[i] = e_new[i] / C_v

    # Recompute pressure with updated density and internal energy
    def recomputePressure(self, predictor):
        gamma_minus = self.mat.gamma - 1
        if predictor:
            P_new = self.fields.P_p
            e_new = self.fields.e_p
            rho_new = self.fields.rho_p
        else:
            P_new = self.fields.P
            e_new = self.fields.e
            rho_new = self.fields.rho
        for i in range(self.geo.N):
            P_new[i] = gamma_minus * rho_new[i] * e_new[i]

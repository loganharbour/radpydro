import numpy as np
from sys import exit

class Materials:
    def __init__(self, rp):
        self.rp = rp
        self.input = rp.input
        self.geo = rp.geo
        self.N = rp.geo.N

        # Densities (defined on spatial cells)
        self.rho = np.zeros(self.N)
        self.rho_old = np.zeros(self.N)
        # Specific energy densities (defined on spatial cells)
        self.C_v = np.zeros(self.N)
        # Compressability coefficient (defined on spatial cells)
        self.gamma = np.zeros(self.N)
        # Kappa coefficients (defined on spatial cells)
        self.k_1 = np.zeros(self.N)
        self.k_2 = np.zeros(self.N)
        self.k_3 = np.zeros(self.N)
        self.k_n = np.zeros(self.N)

        # Fill spatial cell material properties
        for i in range(self.N):
            r = self.geo.r[i]
            self.rho_old[i] = self.input.rho(r)
            self.C_v[i] = self.input.C_v(r)
            self.gamma[i] = self.input.gamma(r)
            self.k_1[i], self.k_2[i], self.k_3[i], self.k_n[i] = self.input.k(r)
        np.copyto(self.rho, self.rho_old)

        # Initialize spatial cell masses (m = rho * v)
        self.m = np.zeros(self.N)
        np.multiply(self.geo.V, self.rho, out=self.m)
        # Initialize median mesh masses
        self.m_half = np.zeros(self.N + 1)
        self.m_half[0] = self.m[0] / 2 # see below Eq. 38
        self.m_half[-1] = self.m[-1] / 2 # see below Eq. 38
        for i in range(1, self.N - 1):
            self.m_half[i] = self.geo.V[i - 1] * self.rho[i - 1] + self.geo.V[i] * self.rho[i]

    # Recompute densities with newly updated volumes
    def recomputeRho(self):
        # Copy over to old
        np.copyto(self.rho_old, self.rho)

        # And recompute (rho = m / rho)
        np.divide(self.m, self.geo.V, out=self.rho)

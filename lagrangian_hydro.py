import numpy as np

class LagrangianHydro:
    def __init__(self, rp):
        self.rp = rp
        self.input = rp.input
        self.geo = rp.geo

        self.u = np.zeros(self.geo.N + 1)
        self.u_old = np.zeros(self.geo.N + 1)

        self.P = np.zeros(self.geo.N + 1)
        self.P_old = np.zeros(self.geo.N + 1)

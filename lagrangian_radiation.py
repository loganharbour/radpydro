import numpy as np

class LagrangianRadiation:
    def __init__(self, rp):
        self.rp = rp
        self.input = rp.input
        self.geo = rp.geo
        self.mat = rp.mat
        self.hydro = None

        self.E = np.zeros(self.geo.N + 1)
        self.E_old = np.zeros(self.geo.N + 1)

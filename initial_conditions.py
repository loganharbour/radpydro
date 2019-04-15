import numpy as np

class InitialConditions:
    def __init__(self, r_L, r_R):
        self.r_L = r_L
        self.r_R = r_R

    def rho(self, r):
        if ((r >= self.r_L) and (r < 0)):
             return 1
        else:
            return 1.29731782

    def u(self, r):
        if ((r >= self.r_L) and (r < 0)):
            return 1.52172533e-1
        else:
            return 1.17297805e-1

    def T(self, r):
        if ((r >= self.r_L) and (r < 0)):
            return 0.1
        else:
            return 1.19475741e-1

    def E(self, r):
        if ((r >= self.r_L) and (r < 0)):
            return 1.37201720e-6
        else:
            return 2.79562228e-6

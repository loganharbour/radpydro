import numpy as np

class InitialConditions:
    def __init__(self, R_L, R_R):
        self.R_L = R_L
        self.R_R = R_R

    def rho(self, r):
        if ((r >= self.R_L) and (r < 0)):
             return 1
        else:
            return 1.29731782

    def u(self, r):
        if ((r >= self.R_L) and (r < 0)):
            return 0.152172533
        else:
            return 0.117297805

    def T(self, r):
        if ((r >= self.R_L) and (r < 0)):
            return 0.1
        else:
            return 0.119475741

    def E(self, r):
        if ((r >= self.R_L) and (r < 0)):
            return 1.37201720e-6
        else:
            return 2.79562228e-6

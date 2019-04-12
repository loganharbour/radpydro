import numpy as np

class InitialConditions:
    def __init__(self, R_L, R_R):
        self.R_L = R_L
        self.R_R = R_R

    def rho(self, r):
        if ((r >= self.R_L) and (r < 0)):
             return 1
        else:
            return 1.07495

    def u(self, r):
        if ((r >= self.R_L) and (r < 0)):
            return 1.42601
        else:
            return 1.32658

    def T(self, r):
        if ((r >= self.R_L) and (r < 0)):
            return 1.
        else:
            return 1.04946

    def E(self, r):
        if ((r >= self.R_L) and (r < 0)):
            return 1.37201720e-6
        else:
            return 2.79562228e-6

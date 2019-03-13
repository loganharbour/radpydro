import numpy as np

class LagrangianRadiation:
    def __init__(self, rp):
        self.rp = rp
        self.input = rp.input
        self.geo = rp.geo
        self.mat = rp.mat
        self.fields = rp.fields

    def assemblePredictorSystem(self, dt):

        rho = self.fields.rho
        rho_old = self.fields.rho_old
        rho_k = (rho + rho_old) / 2

        dr_old = 0





    def solveSystem(self):
        print("nothing yet")

    def recomputeInternalEnergy(self):
        print("nothing yet")

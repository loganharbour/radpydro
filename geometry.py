import numpy as np
from sys import exit
from tools import checkInteger, checkNumber

class Geometry:
    def __init__(self, rp):
        self.rp = rp
        self.input = rp.input

        # Spatial domain size
        self.r_max = checkNumber(self.input.r_max, "r_max")
        self.r_max_old = self.r_max

        # Number of cells
        self.N = checkInteger(self.input.N, "N")

        # Radius at cell centers (uniform spacing)
        self.r = np.linspace(0, self.r_max, num=self.N + 1)
        self.r_old = np.copy(self.r)

        # Empty containers for areas, volumes, and masses
        self.A = np.zeros(self.N + 1)
        self.A_old = np.zeros(self.N + 1)
        self.V = np.zeros(self.N + 1)
        self.V_old = np.zeros(self.N + 1)
        self.rho = np.zeros(self.N + 1)
        self.m = np.zeros(self.N + 1)

        # Fill areas and volumes
        self.recompute(True)

    def recompute(self, init = False):
        self.A_old = np.copy(self.A)
        self.V_old = np.copy(self.V)

        self.computeAreas()
        self.computeVolumes()

        # Copy to old on init
        if (init):
            self.A_old = np.copy(self.A)
            self.V_old = np.copy(self.V)

    def computeAreas(self):
        sys.exit('Can only call computeAreas() from inherited geometry classes')
        pass

    def computeVolumes(self):
        sys.exit('Can only call computeVolumes() from inherited geometry classes')
        pass

class SlabGeometry(Geometry):
    def computeAreas(self):
        self.A = 1

    def computeVolumes(self):
        for i in range(self.N):
            self.V[i] = self.r[i + 1] - self.r[i]

class CylindricalGeometry(Geometry):
    def computeAreas(self):
        self.A = 2 * np.pi * self.r

    def computeVolumes(self):
        for i in range(self.N):
            self.V[i] = np.pi * (self.r[i + 1]**2 - self.r[i]**2)

class SphericalGeometry(Geometry):
    def computeAreas(self):
        self.A = 4 * np.pi * np.square(self.r)

    def computeVolumes(self):
        for i in range(self.N):
            self.V[i] = 4 * np.pi * (self.r[i + 1]**3 - self.r[i]**3) / 3

import numpy as np
from sys import exit
from tools import checkInteger, checkNumber

class Geometry:
    def __init__(self, r_max, N):
        # Spatial domain size
        self.r_max = checkNumber(r_max, "r_max")

        # Number of cells
        self.N = checkInteger(N, "N")

        # Radius at cell centers (uniform spacing)
        self.r = np.linspace(0, self.r_max, num=self.N + 1)

        # Empty containers for areas and volumes
        self.A = None
        self.V = None

        # Setup the rest (dependent on geometry type)
        self.setup()

    # Setup to be called by parent geometry classes
    def setup(self):
        sys.exit('Can only call Geometry.setup() from an inherited class')
        pass

class SlabGeometry(Geometry):
    def setup(self):
        self.A = np.ones(self.N + 1)

        self.V = np.zeros(self.N)
        for i in range(self.N):
            self.V[i] = self.r[i + 1] - self.r[i]

class CylindricalGeometry(Geometry):
    def setup(self):
        self.A = 2 * np.pi * self.r

        self.V = np.zeros(self.N)
        for i in range(self.N):
            self.V[i] = np.pi * (self.r[i + 1]**2 - self.r[i]**2)

class SphericalGeometry(Geometry):
    def setup(self):
        self.A = 4 * np.pi * np.square(self.r)

        self.V = np.zeros(self.N)
        for i in range(self.N):
            self.V[i] = 4 * np.pi * (self.r[i + 1]**3 - self.r[i]**3) / 3

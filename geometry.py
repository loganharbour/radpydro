import numpy as np
from sys import exit

class Geometry:
    def __init__(self, rp):
        self.rp = rp
        self.input = rp.input

        # Radius at cell edges (user defined)
        self.r_half = np.copy(self.input.r_half)
        self.r_half_old = np.copy(self.input.r_half)
        # Number of cells
        self.N = self.r_half.size - 1
        # Spatial domain size
        self.r_max = self.r_half[-1]
        self.r_max_old = self.r_half[-1]

        # Areas (defined at edges)
        self.A = np.zeros(self.N + 1)
        self.A_old = np.zeros(self.N + 1)
        # Volumes (defined on spatial cells)
        self.V = np.zeros(self.N)
        self.V_old = np.zeros(self.N)
        # Radius at cell centers
        self.r = np.zeros(self.N)
        self.r_old = np.zeros(self.N)

        # Cells widths 
        self.dr = np.zeros(self.N)
        self.dr_old = np.zero(self.N)

        # Initialize A, V, r and copy to old
        self.recomputeGeometry()
        np.copyto(self.A_old, self.A)
        np.copyto(self.V_old, self.V)
        np.copyto(self.r_old, self.r)

    def moveMesh(self, u, u_old, dt):
        # Recompute radii at edges
        np.copyto(self.r_half_old, self.r_half)
        self.r_half += 0.5 * (u + u_old) * dt

        # Recompute A, V, and r using newly obtained r_half
        self.recomputeGeometry()

    # Recompute A, V, and r using r_half
    def recomputeGeometry(self):
        # Copy over to old
        np.copyto(self.A_old, self.A)
        np.copyto(self.V_old, self.V)
        np.copyto(self.r_old, self.r)

        # Recompute areas and volumes (dependent on geometry type)
        self.recomputeAreas()
        self.recomputeVolumes()

        # Recompute new cell centered radii
        for i in range(self.N):
            self.r[i] = (self.r_half[i] + self.r_half[i + 1]) / 2

    # Recompute area function to be overridden by children classes
    def recomputeAreas(self):
        pass

    # Recompute volume function to be overridden by children classes
    def recomputeVolumes(self):
        pass

class SlabGeometry(Geometry):
    def recomputeAreas(self):
        self.A.fill(1.0)

    def recomputeVolumes(self):
        for i in range(self.N):
            self.V[i] = self.r_half[i + 1] - self.r_half[i]

class CylindricalGeometry(Geometry):
    def recomputeAreas(self):
        self.A = 2 * np.pi * self.r_half

    def recomputeVolumes(self):
        for i in range(self.N):
            self.V[i] = np.pi * (self.r_half[i + 1]**2 - self.r_half[i]**2)

class SphericalGeometry(Geometry):
    def recomputeAreas(self):
        self.A = 4 * np.pi * np.square(self.r_half)

    def recomputeVolumes(self):
        for i in range(self.N):
            self.V[i] = 4 * np.pi * (self.r_half[i + 1]**3 - self.r_half[i]**3) / 3

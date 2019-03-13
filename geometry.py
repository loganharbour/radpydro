import numpy as np
from sys import exit

class Geometry:
    def __init__(self, rp):
        self.rp = rp
        self.input = rp.input

        # Radius at cell edges (user defined)
        self.r_half = np.copy(self.input.r_half)
        self.r_half_p = np.copy(self.r_half)
        self.r_half_old = np.copy(self.r_half)
        # Number of cells
        self.N = self.r_half.size - 1

        # Areas (defined at edges)
        self.A = np.zeros(self.N + 1)
        self.A_p = np.zeros(self.N + 1)
        self.A_old = np.zeros(self.N + 1)
        # Volumes (defined on spatial cells)
        self.V = np.zeros(self.N)
        self.V_p = np.zeros(self.N)
        self.V_old = np.zeros(self.N)
        # Radius at cell centers
        self.r = np.zeros(self.N)
        self.r_p = np.zeros(self.N)
        self.r_old = np.zeros(self.N)
        # Cells widths
        self.dr = np.zeros(self.N)
        self.dr_p = np.zeros(self.N)
        self.dr_old = np.zeros(self.N)

        # Initialize A, V, r and copy to old
        self.recomputeGeometry(False)
        self.stepGeometry()

    def moveMesh(self, dt, predictor):
        if predictor:
            u_avg = (self.rp.fields.u_p + self.rp.fields.u_old) / 2
            self.r_half_p = self.r_half_old + u_avg * dt
            self.recomputeGeometry(True)
            self.rp.fields.rho_p = self.rp.mat.m / self.V_p
        else:
            u_avg = (self.rp.fields.u + self.rp.fields.u_old) / 2
            self.r_half = self.r_half_old + u_avg * dt
            self.recomputeGeometry(False)
            self.rp.fields.rho = self.rp.mat.m / self.V

        # Recompute A, V, and r using newly obtained r_half
        self.recomputeGeometry(predictor)

    # Recompute A, V, dr, and r using r_half_p (predictor = true) or
    # r_half (predictor = false)
    def recomputeGeometry(self, predictor):
        if predictor:
            A = self.A_p
            V = self.V_p
            r_half = self.r_half_p
            r = self.r_p
            dr = self.dr_p
        else:
            A = self.A
            V = self.V
            r_half = self.r_half
            r = self.r
            dr = self.dr

        # Update areas
        self.recomputeAreas(A, r_half)
        # Update volumes
        self.recomputeVolumes(V, r_half)
        # Update cell centered radii and cell widths
        for i in range(self.N):
            r[i] = (r_half[i] + r_half[i + 1]) / 2
            dr[i] = (r_half[i + 1] - r_half[i])

    # Copy over all new values to old positions
    def stepGeometry(self):
        np.copyto(self.r_half_old, self.r_half)
        np.copyto(self.A_old, self.A)
        np.copyto(self.V_old, self.V)
        np.copyto(self.r_old, self.r)
        np.copyto(self.dr_old, self.dr)

class SlabGeometry(Geometry):
    def recomputeAreas(self, A, r_half):
        A.fill(1.0)

    def recomputeVolumes(self, V, r_half):
        for i in range(self.N):
            V[i] = r_half[i + 1] - r_half[i]

class CylindricalGeometry(Geometry):
    def recomputeAreas(self, A, r_half):
        A = 2 * np.pi * r_half

    def recomputeVolumes(self, V, r_half):
        for i in range(self.N):
            V[i] = np.pi * (r_half[i + 1]**2 - r_half[i]**2)

class SphericalGeometry(Geometry):
    def recomputeAreas(self, A, r_half):
        A = 4 * np.pi * np.square(r_half)

    def recomputeVolumes(self, V, r_half):
        for i in range(self.N):
            V[i] = 4 * np.pi * (r_half[i + 1]**3 - r_half[i]**3) / 3

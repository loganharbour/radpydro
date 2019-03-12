import numpy as np
from sys import exit

class InputParameters:
    def __init__(self):
        # Geometry specficiations
        self.geometry = 'slab'
        self.r_half = None

        # Whether or not to enable the hydro/radiation run
        self.enable_hydro = True
        self.enable_radiation = True

        # Material properties
        self.C_v = None
        self.rho = None
        self.gamma = None
        self.k = None

        # Initial conditions
        self.T = None

        # Hydro boundary conditions
        self.u_BC = None
        self.P_BC = None

    def checkInputs(self):
        # Geometry checks
        if (self.r_half is None):
            exit("The cell edges (r_half) must be specified")
        if (self.geometry not in ['slab', 'cylindrical', 'spherical']):
            exit("Geometry type {} not supported".format(self.geometry))

        # Need to run at least one problem
        if (not self.enable_hydro and not self.enable_radiation):
            exit("No problem to run")

        # Material properties
        if not callable(self.C_v):
            exit("Need to specify C_v property as a function")
        if not callable(self.rho):
            exit("Need to specify rho initial condition as a function")
        if not callable(self.gamma):
            exit("Need to specify gamma property as a function")
        if not callable(self.k):
            exit("Need to specify k property as a function")
        if len(self.k(0)) != 4:
            exit("k needs to return length 4 (k1, k2, k3, n)")

        # Initial conditions
        if not callable(self.T):
            exit("Need to specify T initial condition as a function")

        # Velocity and pressure BC checks
        if (self.u_BC is None and self.P_BC is None):
            exit("Either a velocity BC (u_BC) or a pressure BC (P_BC) is required")
        if (self.u_BC is not None and self.P_BC is not None):
            exit("Velocity (u_BC) and pressure (P_BC) cannot be specified together")

import numpy as np
from sys import exit

def isScalar(value):
    return isinstance(value, float) or isinstance(value, int)

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
        self.gamma = None
        self.kappa = None
        self.kappa_s = None

        # Constants
        self.a = None
        self.c = None

        # Initial conditions
        self.E = None
        self.rho = None
        self.T = None
        self.u = None

        # Hydro boundary conditions
        self.hydro_L = None
        self.hydro_R = None
        self.hydro_L_val = None
        self.hydro_R_val = None

        # Radiation boundary conditions
        self.rad_L = None
        self.rad_R = None
        self.rad_L_val = None
        self.rad_R_val = None

        # Iteration parameters
        self.CoFactor = None
        self.relEFactor = None
        self.maxTimeStep = None
        self.T_final = None

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
        if not isScalar(self.C_v):
            exit("Need to specify C_v as a scalar")
        if not isScalar(self.gamma) or self.gamma <=1.0:
            exit("Need to specify gamma as a scalar larger than 1.0")
        if not isScalar(self.kappa_s):
            exit("Need to specify kappa_s as a scalar")
        if self.kappa is None or len(self.kappa) != 4:
            exit("Need to specify k property as a list of 4 scalars (1, 2, 3, n)")

        # Constants
        if not isScalar(self.a):
            exit("Need to specify a as a scalar")
        if not isScalar(self.c):
            exit("Need to specify c as a scalar")

        # Initial conditions
        if not callable(self.rho):
            exit("Need to specify rho initial condition as a function")
        if not callable(self.T):
            exit("Need to specify T initial condition as a function")
        if callable(self.E) and not self.enable_radiation:
            print("E initial condition is provided but radiation is disabled")
        if not callable(self.E) and self.enable_radiation:
            exit("Need to specify E initial coniditon as a function")
        if not callable(self.u):
            exit("Need to specify u initial condition as a function")

        # Hydro boundary conditions
        if self.hydro_L is None or self.hydro_L not in ['u', 'P']:
            exit("Need to specify u or P for hydro_L")
        if self.hydro_L is 'P' and not isScalar(self.hydro_L_val):
            exit("With hydro_L = P, hydro_L_val must be a scalar")
        if self.hydro_R is None or self.hydro_R not in ['u', 'P']:
            exit("Need to specify u or P for hydro_R")
        if self.hydro_R is 'P' and not isScalar(self.hydro_R_val):
            exit("With hydro_R = P, hydro_R_val must be a scalar")

        # Radiation boundary conditions
        if self.rad_L is None or self.rad_L not in ['source', 'reflective']:
            exit("Need to specify source or reflective for rad_L")
        if self.rad_L is 'source' and not isScalar(self.rad_L_val):
            exit("With rad_L = source, rad_L_val must be a scalar")
        if self.rad_L is 'reflective' and self.rad_L_val is not None:
            exit("rad_L_val should not be specified with rad_L = reflective")
        if self.rad_R is None or self.rad_R not in ['source', 'reflective']:
            exit("Need to specify source or reflective for rad_R")
        if self.rad_R is 'source' and not isScalar(self.rad_R_val):
            exit("With rad_R = source, rad_R_val must be a scalar")
        if self.rad_R is 'reflective' and self.rad_R_val is not None:
            exit("rad_R_val should not be specified with rad_R = reflective")

        # Iteration parameters
        if not isScalar(self.CoFactor):
            exit("Need to specify a positive Courant factor")
        if not isScalar(self.relEFactor):
            exit("Need to specify a maximum relative radiative energy change")
        if not isScalar(self.maxTimeStep):
            exit("Need to specify a maximum time step (must be lower than T_final)")
        if not isScalar(self.T_final):
            exit("Need to specify the end time of the iteration")

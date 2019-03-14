import numpy as np
from sys import exit

# Ensure value is of type float or integer
def checkScalar(value, prefix):
    if isinstance(value, float) or isinstance(value, int):
        return value
    else:
        sys.exit('{} needs to be specified as a scalar'.format(prefix))

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
        self.a = None
        self.kappa = None

        # Initial conditions
        self.E = None
        self.rho = None
        self.T = None
        self.u = None

        # Hydro boundary conditions
        self.contrain_u = None
        self.P_BC = None
        self.E_BC = None

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
        checkScalar(self.C_v, "C_v")
        checkScalar(self.gamma, "gamma")
        checkScalar(self.a, "a")
        if self.kappa is None or len(self.kappa) != 4:
            exit("Need to specify k property as a list of 4 scalars (1, 2, 3, n)")

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

        # Boundary conditions
        if type(self.constrain_u) != bool:
            exit("constrain_u needs to be either True or False")
        if self.constrain_u and self.P_BC is not None:
            exit("Velocity (constain_u) and pressure (P_BC) cannot be specified together")
        if self.E_BC is None:
            exit("E_BC must be set (fix this later! what do we do w/o E_BC?)")
        if self.E_BC is not None and len(self.E_BC) != 2:
            exit("E_BC must be two values [0, 0] for reflective, [val1, val2] for source")
        if self.P_BC is not None and len(self.P_BC) != 2:
            exit("If set, P_BC must be two values (left and right)")

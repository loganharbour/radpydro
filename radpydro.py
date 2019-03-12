from geometry import SlabGeometry, CylindricalGeometry, SphericalGeometry
from inputparameters import InputParameters
from lagrangian_hydro import LagrangianHydro
from lagrangian_radiation import LagrangianRadiation
from materials import Materials
from sys import exit

class RadPydro:
    def __init__(self, input):
        # Store input parameters and check them
        self.input = input
        self.input.checkInputs()

        # Define geometry based on geometry input type
        if input.geometry == 'slab':
            self.geo = SlabGeometry(self)
        elif input.geometry == 'cylindrical':
            self.geo = CylindricalGeometry(self)
        else:
            self.geo = SphericalGeometry(self)

        # Initialize material handler now that geometry is initialized
        self.mat = Materials(self)

        # Initialize the radiation and hydro problems
        self.rad = LagrangianRadiation(self)
        self.hydro = LagrangianHydro(self)

        # Define easy access in each problem to the other
        self.rad.hydro = self.hydro
        self.hydro.rad = self.rad

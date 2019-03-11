from geometry import SlabGeometry, CylindricalGeometry, SphericalGeometry
from inputparameters import InputParameters
from lagrangian_hydro import LagrangianHydro
from lagrangian_radiation import LagrangianRadiation
from sys import exit

class RadPydro:
    def __init__(self, input):
        self.input = input

        if input.geometry == 'slab':
            self.geo = SlabGeometry(self)
        elif input.geometry == 'cylindrical':
            self.geo = CylindricalGeometry(self)
        elif input.geometry == 'spherical':
            self.geo = SphericalGeometry(self)
        else:
            exit('Geometry type {} not supported'.format(input.geometry))

        self.rad = LagrangianRadiation(self)
        self.hydro = LagrangianHydro(self)

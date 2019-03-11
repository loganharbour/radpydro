from inputparameters import InputParameters
from radpydro import RadPydro

input = InputParameters()
input.geometry = 'spherical'
input.r_max = 5
input.N = 5

rp = RadPydro(input)

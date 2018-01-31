import cantera as ct
import numpy as np
from numpy import array
gas = ct.Solution('gri30.xml')

# dimension of many variables
NSP = gas.n_species
NE = gas.n_elements
NR = gas.n_reactions

# matrix
# elemental composition of species
E = array([[sp.composition.get(e,0.0) for e in gas.element_names] for sp in gas.species()])
# dict of it
Ed = dict(zip(gas.species_names,E))
# atom weights
AW = gas.atomic_weights
# dict of it
AWd = dict(zip(gas.element_names,AW))
# MW weight
MW = E.dot(AW)
assert np.allclose(MW,gas.molecular_weights),"Calculated molecular weights do not match cantera's value."
# dict of MW weight
MWd = dict(zip(gas.species_names,MW))

# T terms
# [1,T,T**2,T**3,T**4]
def generate_thermo_table(sp:ct.Species):
    thermo = sp.thermo
    type_name = thermo.__class__.__name__
    if type_name == 'NasaPoly2': # coeffs[15]: [0:Tmid, 1-7:lowT coeffs, 8-14:highT coeffs]
        T_sections = [thermo.min_temp,thermo.coeffs[0],thermo.max_temp]
        coeffs = [thermo.coeffs[8:15],thermo.coeffs[1:8]]
        return (T_sections, coeffs)
    else:
        raise NotImplementedError
        
## thermo tables
## every item is a tuple of (T_sections, [coeffs_of_T_section[0], coeffs_of_T_section[1], ... ])
TT = [generate_thermo_table(sp) for sp in SPs]

from bisect import bisect_left
def Cp0s(T):
    T_Terms = array([T**i for i in range(5)]+[0.0,0.0])
    return array([coeffs[bisect_left(T_sec,T)-1] for (T_sec,coeffs) in TT]).dot(T_Terms)*ct.gas_constant
def H0s(T):
    T_Terms = array([T**(i+1)/(i+1) for i in range(5)]+[1.0,0.0])
    return array([coeffs[bisect_left(T_sec,T)-1] for (T_sec,coeffs) in TT]).dot(T_Terms)*ct.gas_constant
from math import log
def S0s(T):
    T_Terms = array([log(T)]+[T**i/i for i in range(1,5)]+[0.0,1.0])
    return array([coeffs[bisect_left(T_sec,T)-1] for (T_sec,coeffs) in TT]).dot(T_Terms)*ct.gas_constant

# assert test
Cp0_1900_0 = array([sp.thermo.cp(1900) for sp in gas.species()])
Cp0_1900_1 = Cp0s(1900)
assert np.allclose(Cp_1900_0,Cp_1900_1), "Cp0 calculation is wrong!"
H0_1900_0 = array([sp.thermo.h(1900) for sp in gas.species()])
H0_1900_1 = H0s(1900)
assert np.allclose(H0_1900_0,H0_1900_1), "H0 calculation is wrong!"
S0_1900_0 = array([sp.thermo.s(1900) for sp in gas.species()])
S0_1900_1 = S0s(1900)
assert np.allclose(S0_1900_0,S0_1900_1), "S0 calculation is wrong!"

SR = gas.reactant_stoich_coeffs()
assert SR.shape == (NSP,NR), "Reactant stoichiometric coefficients do not match!"
SP = gas.product_stoich_coeffs()
assert SP.shape == (NSP,NR), "Reactant stoichiometric coefficients do not match!"
SNet = SP - SR

# reaction order is equal to stoichiometric coeffs by default
if all([len(R.orders)==0 for R in Reactions]):
    ## Reaction order is equal to stoichiometric coeffs
    OR = SR
    OP = SP
else:
    raise NotImplementedError




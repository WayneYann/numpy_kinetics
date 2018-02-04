# -*- coding: utf-8 -*-

"""
Author: Di Cheng
"""
import cantera as ct
import numpy as np
# use np.log and np.exp by default
from numpy import array,log,exp
from collections import Counter,defaultdict
from scipy.sparse import csr_matrix as sp_mat
from numpy.polynomial.chebyshev import chebval2d
import unittest


# for example
# C10 Methyl Ester Surrogates for Biodiesel of LLNL is used as an example
# mech: https://combustion.llnl.gov/content/assets/docs/combustion/md9dnc7_mech_v1.txt
# thermo: https://combustion.llnl.gov/content/assets/docs/combustion/md9dnc7_therm_v1.txt
# converted using the following cmd:
# ck2cti --input=md9dnc7_mech_v1.txt --thermo=md9dnc7_therm_v1.txt --permissive
# output is md9dnc7_mech_v1.cti
gas = ct.Solution('GRI30.cti')
#gas = ct.Solution('md9dnc7_mech_v1.cti')

## constants preparation

#reaction rearrangement 
reactions = gas.reactions()
ElementaryReactions = []
ThreeBodyReactions = []
FalloffReactions_SIMPLE = []
FalloffReactions_TROE = []
FalloffReactions_SRI = []
ChemicallyActivatedReactions_SIMPLE = []
ChemicallyActivatedReactions_TROE = []
ChemicallyActivatedReactions_SRI = []
PlogReactions = []
PlogReactionRatesNo = []
ChebyshevReactions = []

for R in reactions:
    reaction_type = R.__class__.__name__
    if (reaction_type == "ElementaryReaction"):
        ElementaryReactions.append(R)
    elif (reaction_type == "ThreeBodyReaction"):
        ThreeBodyReactions.append(R)
    elif (reaction_type == "FalloffReaction"):
        if (R.falloff.falloff_type == 100):
            FalloffReactions_SIMPLE.append(R)
        elif (R.falloff.falloff_type == 110):
            FalloffReactions_TROE.append(R)
        elif (R.falloff.falloff_type == 112):
            FalloffReactions_SRI.append(R)
        else:
            raise NotImplementedError
    elif (reaction_type == "ChemicallyActivatedReaction"):
        if (R.falloff.falloff_type == 100):
            ChemicallyActivatedReactions_SIMPLE.append(R)
        elif (R.falloff.falloff_type == 110):
            ChemicallyActivatedReactions_TROE.append(R)
        elif (R.falloff.falloff_type == 112):
            ChemicallyActivatedReactions_SRI.append(R)
        else:
            raise NotImplementedError
    elif (reaction_type == "PlogReaction"):
        PlogReactions.append(R)
        PlogReactionRatesNo.append(len(R.rates))
    elif (reaction_type == "ChebyshevReaction"):
        ChebyshevReactions.append(R)
    else:
        raise NotImplementedError

FalloffReactions = FalloffReactions_SIMPLE +FalloffReactions_TROE +FalloffReactions_SRI
ChemicallyActivatedReactions = ChemicallyActivatedReactions_SIMPLE +ChemicallyActivatedReactions_TROE +ChemicallyActivatedReactions_SRI
reactions = ElementaryReactions+ThreeBodyReactions+FalloffReactions+ChemicallyActivatedReactions+PlogReactions+ChebyshevReactions
PlogReactionRatesNo = array(PlogReactionRatesNo)
gas = ct.Solution(thermo='IdealGas', kinetics='GasKinetics', species=gas.species(), reactions=reactions)

#set an initial state
gas.TPX = 1900, ct.one_atm*10, "ch4:1, o2:2, n2:7.52"
gas.equilibrate('TP')

# dimension of many variables
NSP = gas.n_species
NE = gas.n_elements

NR = len(reactions)
NR_E = len(ElementaryReactions)
NR_TB = len(ThreeBodyReactions)
NR_FO = len(FalloffReactions)
NR_FO_SIMPLE = len(FalloffReactions_SIMPLE)
NR_FO_TROE = len(FalloffReactions_TROE)
NR_FO_SRI = len(FalloffReactions_SRI)
NR_CA = len(ChemicallyActivatedReactions)
NR_CA_SIMPLE = len(ChemicallyActivatedReactions_SIMPLE)
NR_CA_TROE = len(ChemicallyActivatedReactions_TROE)
NR_CA_SRI = len(ChemicallyActivatedReactions_SRI)
NR_PLOG = len(PlogReactions)
NR_PLOG_RATES = sum(PlogReactionRatesNo)
NR_CHEB = len(ChebyshevReactions)


# In[4]:


# element composition matrix
E = array([[sp.composition.get(e,0.0) for e in gas.element_names] for sp in gas.species()])
# dict of it
E_D = dict(zip(gas.species_names,E))
# atom weight vector
AW = gas.atomic_weights
# dict of it
AW_D = dict(zip(gas.element_names,AW))
# MW weight vector, for convenience
MW = E.dot(AW)
# test
assert np.allclose(MW,gas.molecular_weights),"Calculated molecular weights do not match cantera's value."
# dict of MW weight
MW_D = dict(zip(gas.species_names,MW))


# In[5]:


## define flatten function for list of list
flatten = lambda a: [subitem for item in a for subitem in (item if isinstance(item, list) else [item])]


# In[6]:


## memory arrangement: to maximize largest usable continuous block


# generate Arrhenius equation matrix elements

ElementaryReactionArrs = [R.rate for R in ElementaryReactions]
ThreeBodyReactionArrs = [R.rate for R in ThreeBodyReactions]
FalloffReactionArrs_low = [R.low_rate for R in FalloffReactions]
FalloffReactionArrs_high = [R.high_rate for R in FalloffReactions]
ChemicallyActivatedReactionArrs_high = [R.high_rate for R in ChemicallyActivatedReactions]
ChemicallyActivatedReactionArrs_low = [R.low_rate for R in ChemicallyActivatedReactions]
PlogReactionArrs = flatten([[t[1] for t in R.rates] for R in ChemicallyActivatedReactions])
# ChebyshevReactionArrs = []

Arrs = ElementaryReactionArrs+ThreeBodyReactionArrs+ FalloffReactionArrs_high+ChemicallyActivatedReactionArrs_low+ FalloffReactionArrs_low+ChemicallyActivatedReactionArrs_high+ PlogReactionArrs

# Ta, logA, beta
# rate's const table
ARR_CONST_TABLE = array([(log(r.pre_exponential_factor),r.activation_energy/ct.gas_constant,r.temperature_exponent) for r in Arrs])


# In[7]:


## thermodynamic calculations

if not all([sp.thermo.reference_pressure == ct.one_atm for sp in gas.species()]):
    raise NotImplementedError

# T terms is
# [1,T,T**2,T**3,T**4]
def generate_thermo_table(sp:ct.Species):
    """
    generate thermodynamic tables for calculating ideal gas properties
    """
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
TT = [generate_thermo_table(sp) for sp in gas.species()]

from bisect import bisect_left
def Cp0s(T):
    T_Terms = array([T**i for i in range(5)]+[0.0,0.0])
    return array([coeffs[bisect_left(T_sec,T)-1] for (T_sec,coeffs) in TT]).dot(T_Terms)*ct.gas_constant
def H0s(T):
    T_Terms = array([T**(i+1)/(i+1) for i in range(5)]+[1.0,0.0])
    return array([coeffs[bisect_left(T_sec,T)-1] for (T_sec,coeffs) in TT]).dot(T_Terms)*ct.gas_constant
def S0s(T):
    T_Terms = array([log(T)]+[T**i/i for i in range(1,5)]+[0.0,1.0])
    return array([coeffs[bisect_left(T_sec,T)-1] for (T_sec,coeffs) in TT]).dot(T_Terms)*ct.gas_constant
def G0s(T):
    T_Terms = array([T*(1-log(T))]+[-T**(i+1)/(i*(i+1)) for i in range(1,5)]+[1.0,-T])
    return array([coeffs[bisect_left(T_sec,T)-1] for (T_sec,coeffs) in TT]).dot(T_Terms)*ct.gas_constant

# assert test
Cp0_1900_0 = array([sp.thermo.cp(1900) for sp in gas.species()])
Cp0_1900_1 = Cp0s(1900)
assert np.allclose(Cp0_1900_0,Cp0_1900_1), "Cp0 calculation is wrong!"
H0_1900_0 = array([sp.thermo.h(1900) for sp in gas.species()])
H0_1900_1 = H0s(1900)
assert np.allclose(H0_1900_0,H0_1900_1), "H0 calculation is wrong!"
S0_1900_0 = array([sp.thermo.s(1900) for sp in gas.species()])
S0_1900_1 = S0s(1900)
assert np.allclose(S0_1900_0,S0_1900_1), "S0 calculation is wrong!"
G0_1900_0 = H0_1900_0 - 1900*S0_1900_0
G0_1900_1 = G0s(1900)
assert np.allclose(G0_1900_0,G0_1900_1), "G0 calculation is wrong!"


# In[8]:


## stoichiometric coefficients
## sparse matrix

SR = sp_mat(gas.reactant_stoich_coeffs())
assert SR.shape == (NSP,NR), "Reactant stoichiometric coefficients do not match!"
SP = sp_mat(gas.product_stoich_coeffs())
assert SP.shape == (NSP,NR), "Reactant stoichiometric coefficients do not match!"
SNet = SP - SR

# reaction order is equal to stoichiometric coeffs by default
if all([len(R.orders)==0 for R in gas.reactions()]):
    ## Reaction order is equal to stoichiometric coeffs
    OR = SR
    OP = SP
    ONet = SNet
else:
    raise NotImplementedError


# In[9]:

"""
from matplotlib import pyplot as plt
plt.figure()
plt.spy(SR,marker='.',markersize=1)
# plt.title('sparsity of LHS of stoichiometry')
plt.figure()
plt.spy(SP,marker='.',markersize=1)
# plt.title('sparsity of RHS of stoichiometry')
plt.figure()
plt.spy(SNet,marker='.',markersize=1)
# plt.title('sparsity of RHS-LHS of stoichiometry')
plt.show()
"""

# In[10]:


# log of equilibrium constants

def generate_log_Kc(T:float,mod_irrev:bool=True):
    Inf = float('Inf')
    logKc = ONet.T.dot(log(ct.one_atm/ct.gas_constant/T)-G0s(T)/ct.gas_constant/T)
    # for irreversible reaction, the log(K) = +inf, and log(k_b) = log(k_f) - log(K) = -inf, k_b = 0
    if mod_irrev:
        for r,R in enumerate(reactions):
            if R.reversible == False:
                logKc[r] = Inf 
    return logKc
def generate_log_Kp(T:float,mod_irrev:bool=True):
    Inf = float('Inf')
    logKp = ONet.T.dot(-G0s(T)/ct.gas_constant/T)
    # for irreversible reaction, the log(K) = +inf, and log(k_b) = log(k_f) - log(K) = -inf, k_b = 0
    if mod_irrev:
        for r,R in enumerate(reactions):
            if R.reversible == False:
                logKp[r] = Inf 
    return logKp

## unit test
logKc_0 = log(gas.equilibrium_constants)
logKc_1 = generate_log_Kc(gas.T,False)
assert np.allclose(logKc_0,logKc_1), "Equilbirium constants calculation is wrong!"


# ## Arrhenius formula
# 
# forward reaction rate:
# 
# $$
# k^f_r = A_r T^{\beta_r}\exp(-\frac{T^a_r}{T})  = A_r \exp(-\frac{T^a_r}{T}+{\beta_r}\ln T)  \\
# \mathbf{k^f} = \mathbf{A_r} \times T^{\mathbf{\beta}} \times \exp(-\frac{\mathbf {T^a}}{T}) = \mathbf{A_r} \times \exp(-\frac{\mathbf {T^a}}{T} +\mathbf{\beta}\times \ln T)\\
# \ln k^f_r =\ln A_r + (-\frac{T^a_r}{T}+{\beta_r}\ln T)  \\
# \ln \mathbf{k^f} = \ln \mathbf{A_r} + (-\mathbf {T^a} \times \frac 1 T +\mathbf{\beta}\times \ln T)
# $$
# 

# In[11]:


R0 = reactions[0]
R1 = reactions[NR_E+1]
R2 = reactions[NR_E+NR_FO]


# # Modifier
# 
# log of modifier:

# ## third body efficiency

# In[12]:


NR_TB_Eff = NR_FO+NR_TB+NR_CA

TB_Eff_reactions = ThreeBodyReactions+FalloffReactions+ChemicallyActivatedReactions

name_idx_dict = dict(zip(gas.species_names,range(NSP)))

alpha_default = np.ones(NR_TB+NR_FO+NR_CA)
alpha_extra = np.zeros((NR_TB+NR_FO+NR_CA,NSP))

for i,R in enumerate(TB_Eff_reactions):
    alpha_default[i] = R.default_efficiency
    for sp,eff in R.efficiencies.items():
        alpha_extra[i,name_idx_dict[sp]] = eff - R.default_efficiency
alpha_extra = sp_mat(alpha_extra)

def third_body_concentration(C):
    """
    Return Third body effective concentration
    
    Parameters
    ----------
    C: numpy.array
        C.shape == (NSP,)
    Returns
    -------
    TBC : numpy.array
        Third body effective concentration for ThirdBody, Falloff and ChemicallyActivated Reactions
        TBC.shape == (NR_TB+NR_FO+NR_CA,)
    """
    return (alpha_default*np.sum(C)+alpha_extra.dot(C))


# ## forward base reaction rate coefficients

# In[20]:


def get_logArr_rates(T):
    """
    Return logarithm of Arrhenius rate coefficient of all Arrhenius formula
    
    Parameters
    ----------
    T: float
        Temperature
    Returns
    -------
    logArr_rates : numpy.array
        get all forward base reaction rate coefficient of each Arrehenius formula (include exclude ChebyshevReaction)
        logArr_rates.shape == (NR_E+NR_TB+NR_FO*2+NR_CA*2+NR_PLOG_RATES,)
    """
    T_terms = array([1,-1.0/T,log(T)])
    logArr_rates = rates_const_table.dot(T_terms)
    return logArr_rates

# ## modifier

# In[37]:


## arrangement of TROE parameters is:
## a=0, T***=0,T*=0, T**=0
TROE_Par = array([R.falloff.parameters for R in FalloffReactions_TROE+ChemicallyActivatedReactions_TROE])
for i,t2 in enumerate(TROE_Par[:,3]):
    if t2 == 0:
        TROE_Par[i,3] = float('inf')
## a,b,c,d,e = -1
TROE_Par_OF = TROE_Par[0:NR_FO_TROE]
TROE_Par_CA = TROE_Par[NR_FO_TROE:NR_FO_TROE+NR_CA_TROE]

SRI_Par = array([R.falloff.parameters for R in FalloffReactions_SRI+ChemicallyActivatedReactions_SRI])
SRI_Par_OF = SRI_Par[0:NR_FO_SRI]
SRI_Par_CA = SRI_Par[NR_FO_SRI:NR_FO_SRI+NR_CA_SRI]

def get_logF_TROE(logP,T,p):
    """
    return logF using logP and parameters. vectorized
    """
    from math import log
    if len(p) ==0: return array([])
    log10 = log(10)
    logFcent = np.log((1-p[0])*np.exp(-T/p[1])+p[0]*np.exp(-T/p[2])+np.exp(-p[3]/T))
    ATroe = logP/log10 - 0.67*logFcent/log10 - 0.4
    BTroe = 0.806 - 1.1762*logFcent/log10 - 0.14*logP/log10
    logF_TROE = logFcent/(1+(ATroe/BTroe)**2)
    return logF_TROE

def get_logF_SRI(logP,T,p):
    """
    return logF using logP and parameters. vectorized
    """
    import math 
    from numpy import exp,log
    if len(p) ==0: return array([])
    log10 = math.log(10)
    logF_SRI = log(p[3])+log(T)*p[4] + log(p[0]*exp(-p[1]/T)+exp(-T/p[2]))/(1+(logP/log10)**2)
    return logF_SRI


# ## arrangement and indexing of falloff function
# 
# I used some python and numpy techniques to make it elegant.
# 

# In[38]:


def get_log_modifier(T,C,logArr_rates):
    """
    return reaction rate modifier for each reaction (except Elemental Reaction, ChebyshevReaction and PLog reaction)
    logArr_rates is reused. Numpy's alias (reference to a slice of array) is heavily used.
    modifier.shape == (NR_TB+NR_FO+NR_CA,)
    """
    logX = log(third_body_concentration(C))
    logX_FO = logX[NR_TB:NR_TB+NR_FO]
    logX_CA = logX[NR_TB+NR_FO:NR_TB+NR_FO+NR_CA]

    logArr_rates_FO_high = logArr_rates[NR_E+NR_TB:NR_E+NR_TB+NR_FO]
    logArr_rates_CA_low = logArr_rates[NR_E+NR_TB+NR_FO:NR_E+NR_TB+NR_FO+NR_CA]
    logArr_rates_FO_low = logArr_rates[NR_E+NR_TB+NR_FO+NR_CA:NR_E+NR_TB+2*NR_FO+NR_CA]
    logArr_rates_CA_high = logArr_rates[NR_E+NR_TB+2*NR_FO+NR_CA:NR_E+NR_TB+2*NR_FO+2*NR_CA]

    # calculate logPr
    logPr_FO = logX_FO + logArr_rates_FO_low - logArr_rates_FO_high
    logPr_CA = logX_CA + logArr_rates_CA_low - logArr_rates_CA_high
    
    # calculate Falloff function
    logPr_FO_TROE = logPr_FO[NR_FO_SIMPLE            : NR_FO_SIMPLE+NR_FO_TROE]
    logPr_FO_SRI  = logPr_FO[NR_FO_SIMPLE+NR_FO_TROE : NR_FO_SIMPLE+NR_FO_TROE+NR_FO_SRI]
    logPr_CA_TROE = logPr_CA[NR_CA_SIMPLE            : NR_CA_SIMPLE+NR_CA_TROE]
    logPr_CA_SRI  = logPr_CA[NR_CA_SIMPLE+NR_CA_TROE : NR_CA_SIMPLE+NR_CA_TROE+NR_CA_SRI]

    # reuse logX to store logF
    logF = logX[NR_TB:NR_TB+NR_FO+NR_CA]
    logF_FO = logF[0:NR_FO]
    logF_CA = logF[NR_FO:NR_FO+NR_CA]
    logF_FO_TROE = logF_FO[NR_FO_SIMPLE            : NR_FO_SIMPLE+NR_FO_TROE]
    logF_FO_SRI  = logF_FO[NR_FO_SIMPLE+NR_FO_TROE : NR_FO_SIMPLE+NR_FO_TROE+NR_FO_SRI]
    logF_CA_TROE = logF_CA[NR_CA_SIMPLE            : NR_CA_SIMPLE+NR_CA_TROE]
    logF_CA_SRI  = logF_CA[NR_CA_SIMPLE+NR_CA_TROE : NR_CA_SIMPLE+NR_CA_TROE+NR_CA_SRI]

    # logF_simple = 0 ,Lindemann
    logF_FO[0:NR_FO_SIMPLE] = 0.0
    logF_CA[0:NR_CA_SIMPLE] = 0.0
    
    logF_FO_TROE[:] = get_logF_TROE(logPr_FO_TROE,T,TROE_Par_OF.T)
    logF_CA_TROE[:] = get_logF_TROE(logPr_CA_TROE,T,TROE_Par_CA.T)

    logF_FO_SRI[:]  = get_logF_SRI(logPr_FO_SRI,T,SRI_Par_OF.T)
    logF_CA_SRI[:]  = get_logF_SRI(logPr_CA_SRI,T,SRI_Par_CA.T)

    # calculate modifier
    logF_FO[:] = logF_FO + logPr_FO - np.log(1.0+np.exp(logPr_FO))
    logF_CA[:] = logF_CA - np.log(1.0+np.exp(logPr_CA))

    return logX # return logX as modifier. most modifiers are Third body type.

# ### Plog reaction
# 
# Ref: http://www.cantera.org/docs/sphinx/html/cython/kinetics.html#plogreaction
# 
# - rates: list
#     - (p,Arrhenius): tuple
# $$
# \log k_f(T,p)= \log k_1(T) +(\log k_2(T) - \log k_1(T)) \frac{\log p - \log p_1}{\log p_2 -\log p_1} 
# $$

# In[39]:


# constants
PlogReactionLogPs = [[log(t[0]) for t in R.rates] for R in ChemicallyActivatedReactions]
PlogReactionRatesEnd = np.cumsum(PlogReactionRatesNo)
PlogReactionRatesStart = PlogReactionRatesEnd - PlogReactionRatesNo

def get_Plog_logkf(logP,logArr_rates):
    from numpy import interp
    PlogArr_rates = logArr_rates[NR - NR_PLOG:NR]
    Plog_logkf = np.zeros(NR_PLOG)
    for i,start,end,logPs in enumerate(zip(PlogReactionRatesStart,PlogReactionRatesEnd,PlogReactionLogPs)):
        Plog_logkf[i] = interp(logP, logPs,PlogArr_rates[start,end])
    return Plog_logkf


# ### Chebyshev reaction
# 
# - Pmax,Pmin
# - Tmax,Tmin
# - coeffs: 2D Array
# 
# $$
# \log_{10}{k_f(T,p)} = \sum_i\sum_j\eta_{ij}\phi_i(\tilde T)\phi_j(\tilde p) \\
# \tilde T \equiv \frac{2T^{-1} - T_{\min}^{-1}-T_{\max}^{-1}}{T_{\max}^{-1}-T_{\min}^{-1}} \\
# \tilde p \equiv  \frac{2\log_{10}p - \log_{10}p_\min- \log_{10}p_\max}{ \log_{10}p_\max- \log_{10}p_\min}
# $$
# 
# 

# In[40]:


ChebyshevReactionConsts = [(1/R.Tmax,1/R.Tmin,log(R.Pmax),log(R.Pmin),R.coeffs) for R in ChebyshevReactions]

def get_Cheb_logkf(rT,logP):
    """
    return log of Chebyshev kf
    """
    from numpy.polynomial.chebyshev import chebval2d
    import math
    log10 = math.log(10.0)
    Cheb_logkf = np.zeros(NR_CHEB)
    for i,(rTmax,rTmin,logPmax,logPmin,coeffs) in enumerate(ChebyshevReactionConsts):
        Tbar = (2*rT-rTmin-rTmax)/(rTmax-rTmin)
        Pbar = (2*logP - logPmax -logPmin)/(logPmax - logPmin)
        Cheb_logkf[i] = chebval2d(Tbar,Pbar,coeffs)*log10
    return Cheb_logkf


# ## forward rate coefficient
# 
# $$
# k^f_r = c_rk^f_{r,Arr}
# $$

# In[41]:


def get_logkf(T,P,C):
    """
    get the logkf
    """
    from math import log
    rT = 1.0/T
    logP = log(P)
    logArr_rates = get_logArr_rates(T)
    Major_logkf = logArr_rates[0:NR_E+NR_TB+NR_FO+NR_CA]
    log_modifier = get_log_modifier(T,C,logArr_rates)
    Major_logkf[NR_E:NR_E+NR_TB+NR_FO+NR_CA] += log_modifier
    Plog_logkf = get_Plog_logkf(logP,logArr_rates)
    Cheb_logkf = get_Cheb_logkf(rT,logP)
    return np.r_[Major_logkf,Plog_logkf,Cheb_logkf]



# In[42]:


logkf_0 = log(gas.forward_rate_constants)
logkf_1 = get_logkf(gas.T,gas.P,gas.concentrations)

assert np.allclose(logkf_0,logkf_1),"Reaction forward rate calculation wrong!"


# In[49]:


def get_reaction_type_str(R):
    d = {100:'_Simple',110:'_Troe', 112:'_SRI'}
    type_str = R.__class__.__name__
    if type_str == 'FalloffReaction' or type_str == 'ChemicallyActivatedReaction' :
        type_str += d[R.falloff.falloff_type]
    return type_str
'''
for i,(R,logkf0,logkf1) in enumerate(zip(reactions,logkf_0,logkf_1)):
    print(i,R,get_reaction_type_str(R),"error:",logkf1/logkf0-1)

'''

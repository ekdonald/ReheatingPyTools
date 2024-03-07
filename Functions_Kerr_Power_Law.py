import numpy as np
import pandas as pd
from scipy import interpolate, optimize
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import quad, dblquad, quad_vec, ode, solve_ivp, odeint, romberg, fixed_quad 
from scipy.optimize import toms748, root, curve_fit
from scipy.special import zeta, kn
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.misc import derivative
from numpy import sqrt, log, exp, log10, pi, logspace, linspace, seterr, min, max, append
from numpy import loadtxt, zeros, floor, ceil, unique, sort, cbrt, concatenate, delete, array
from multiprocessing import Pool
import BHProp as bh 
from collections import OrderedDict
olderr = seterr(all='ignore')
import time
import warnings
warnings.filterwarnings('ignore')
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import Functions_phik as phik_funcs
#import reheat_vars as rh_vars


colors = ["#2364aa", "#3da5d9", "#73bfb8", "#fec601", "#ea7317"]
cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)


#----------------------------------------------#
#             USEFUL FUNCTIONS
#----------------------------------------------#

def PlanckMass(t, v, Mi):
    eps = 0.01
    if (eps*Mi > bh.MPL): Mst = eps*Mi
    else: Mst = bh.MPL
    return v[0] - Mst # Function to stop the solver if the BH is equal or smaller than the Planck mass

#----------------#
#  DISTRIBUTION  #
#----------------#

## in mass ##
def fBH_M(M, Mi, sig, alpha):# M in grams
    Mf = Mi * 10**sig
    if(alpha!=0):
        C  = alpha/(Mf**alpha - Mi**alpha)
    else:
        C = 1./np.log(Mf/Mi)
    if((M>=Mi) and (M<=Mi*(10**sig))): 
        return C*M**(alpha-1)
    else:
        return 0

## in spin ##
def fBH_a(ast, astc, sig):
    return (1/(sqrt(2.*pi)*sig))*exp(-0.5*(ast - astc)*(ast - astc)/(sig*sig))

## in mass AND spin ##
def fBH_M_a(M, ast, Mi, sig_M, astc, sig_a, alpha):
    f_M = fBH_M(M, Mi, sig_M, alpha)
    f_a = fBH_a(ast, astc, sig_a)
    return f_M #*f_a

## number distribution ##
def Int_nPBH(ast, M, Mi, sig_M, astc, sig_a, alpha):# M in grams
    return fBH_M_a(M, ast, Mi, sig_M, astc, sig_a, alpha)


## energy distribution ##
def Int_rPBH(ast, M, Mi, sig_M, astc, sig_a, alpha):# M in grams
    return M* fBH_M_a(M, ast, Mi, sig_M, astc, sig_a, alpha)


## number distribution ##
def Int_nPBH_M(M, Mi, sig_M, alpha):# M in grams
    return fBH_M(M, Mi, sig_M, alpha)


## energy distribution ##
def Int_rPBH_M(M, Mi, sig_M, alpha):# M in grams
    return M* fBH_M(M, Mi, sig_M, alpha)


# dM/dt including full grebody factors, for the SM
def eps(M, ast, mDM):
    FSM = bh.fSM(M, ast)      # SM contribution
    FDM = bh.fDM(M, ast, mDM) # DM contribution
    FT  = FSM + FDM           # Total Energy contribution
    return FT

def gam(M, ast, mDM): 
    GSM = bh.gSM(M, ast)      # SM contribution
    GDM = bh.gDM(M, ast, mDM) # DM contribution
    GT  = GSM + GDM           # Total Energy contribution
    return GT # SM contribution

def h(M, ast, mDM): return gam(M, ast, mDM) - 2.*eps(M, ast, mDM) 


def dMdt(M, ast, mDM):
    FSM = bh.fSM(M, ast)      # SM contribution
    FDM = bh.fDM(M, ast, mDM) # DM contribution
    FT  = FSM + FDM           # Total Energy contribution
    return -bh.kappa * FT/(M*M)


def dastdt(M, ast, mDM):
    FSM = bh.fSM(M, ast)      # SM contribution
    FDM = bh.fDM(M, ast, mDM) # DM contribution
    FT  = FSM + FDM           # Total Energy contribution
    GSM = bh.gSM(M, ast)      # SM contribution
    GDM = bh.gDM(M, ast, mDM) # DM contribution
    GT  = GSM + GDM           # Total Energy contribution
    return - ast * bh.kappa * (GT - 2.*FT)/(M*M*M)

# Solving the PBH evolution from initial mass to Planck mass

def PBH_time_ev(Mi, asi, mDM):
    tBE    = []
    MBHBE  = []
    astBE  = []
    taut = -80.
    
    def PlanckMass(t, v, Mi):
        eps = 0.01
        if (eps*Mi > bh.MPL): Mst = eps*Mi
        else: Mst = bh.MPL
        return v[0] - Mst 

    while Mi >= 2.* bh.MPL:
        MPL = lambda t, x:PlanckMass(t, x, Mi)
        MPL.terminal  = True
        MPL.direction = -1.
        tau_sol = solve_ivp(fun=lambda t, y: bh.ItauFO(t, y, mDM), t_span = [-80., 40.], y0 = [Mi, asi], 
                            events=MPL, rtol=1.e-10, atol=1.e-15)
        tau = tau_sol.t[-1] # Log10@PBH lifetime in inverse GeV    
        tBE    = append(tBE,    log10(10.**tau_sol.t[:] + 10.**taut))
        MBHBE  = append(MBHBE,  tau_sol.y[0,:])
        astBE  = append(astBE,  tau_sol.y[1,:])    
        Mi   = tau_sol.y[0,-1]  
        asi  = tau_sol.y[1,-1]    
        taut = log10(10.**tau_sol.t[-1] + 10.**taut)
        
    return [tBE, MBHBE, astBE, taut]

# PBH lifetime

def tau(Mi, asi, mDM):
    #print('Starting tau...')
    taut = -80.
    
    def PlanckMass(t, v, Mi):
        eps = 0.01
        if (eps*Mi > bh.MPL): Mst = eps*Mi
        else: Mst = bh.MPL
        return v[0] - Mst 
    
    while Mi >= 1.05 * bh.MPL:
        MPL = lambda t, x:PlanckMass(t, x, Mi)
        MPL.terminal  = True
        MPL.direction = -1.
        tau_sol = solve_ivp(fun=lambda t, y: bh.ItauFO(t, y, mDM), t_span = [-80., 40.], y0 = [Mi, asi], 
                            events=MPL, rtol=1.e-5, atol=1.e-15)
        tau = tau_sol.t[-1] # Log10@PBH lifetime in inverse GeV 
        Mi   = tau_sol.y[0,-1]  
        asi  = tau_sol.y[1,-1]    
        taut = log10(10.**tau_sol.t[-1] + 10.**taut)                
    tau = taut
        
    return 10.**tau


## tau for generic simu ##
def log_tau_a(i, an, da, mDM):
    ai = an + da*i
    return log10(tau(1., ai, mDM))


###########################################
#
#     Get M(t) and a(t) for any (Mi, ai)
#
###########################################

def fun_aM_K(M_in, a_in, ftau_red, fM_max, fa_max, tsol_max, t):
    if a_in < 0. or a_in > 1.:
        return [bh.MPL, 0.]
    else:
        tau_Mi = 10.**ftau_red(a_in)*M_in**3

        if t <= tau_Mi:
            if(a_in>0):
                x = tsol_max(-log(a_in))
            elif(a_in==0):
                x=1
            t_new = (1-x)*(t/tau_Mi) + x
            M_r = fM_max(x)
            Mt = fM_max(t_new)*M_in/M_r
            a = fa_max(t_new)
        else:
            Mt = 0*bh.MPL
            a  = 0.

        return [Mt, a]


def Int_PBH(grid, t, func_aM_K, ftau_red, fM_max, fa_max, tsol_max, pars):
    #start_tot = time.time()
    Mi, sig_M, ai, sig_a, mDM, alpha = pars
    
    result = 0
    for i in range(len(grid[:,0,0])):
        for j in range(len(grid[0,:,0])):
            
            Mt, at = func_aM_K(grid[i,j,0], grid[i,j,1], ftau_red, fM_max, fa_max, tsol_max, t)
            result += grid[i,j,2] * Mt #* bh.GeV_in_g
    
    #end_tot = time.time()
    #print(end_tot-start_tot)
    return result

def Int_PBH_1D(grid, t, func_aM_K, ftau_red, fM_max, fa_max, tsol_max, pars):
    #start_tot = time.time()
#     Mi, sig_M, ai, sig_a, mDM, alpha = pars
    
    result = 0
    for i in range(len(grid[:,0])):
        
        Mt, at = func_aM_K(grid[i,0], 1e-10, ftau_red, fM_max, fa_max, tsol_max, t)
        result += grid[i,1] * Mt #* bh.GeV_in_g
    
    #end_tot = time.time()
    #print(end_tot-start_tot)
    return result

def Int_PBH_M(t,ti,epsilonMp4, fPBH_M, pars):
    #start_tot = time.time()
    Mi, sig_M, alpha = pars
    
    Mfn = Mi * 10**sig_M
    
    def Mt(Minit, t):
        if(t<Minit**3/(3 * epsilonMp4)):
            
            return Minit*(1-(t)/(Minit**3/(3 * epsilonMp4)))**(1./3.)
        else:
            return 0
        
    def Miev(t):
        return (3*epsilonMp4*(t))**(1./3.)
        
    return romberg(lambda Min: fPBH_M(Min, Mi, sig_M, alpha)*Mt(Min, t), Miev(t), Mfn, tol=1e-25, rtol=1e-25)


def Int_PBH_dM(t,ti,epsilonMp4, fPBH_M, pars):
    #start_tot = time.time()
    Mi, sig_M, alpha = pars
    Mfn = Mi * 10**sig_M
    def Mt(Minit):
        if(t<Minit**3/(3 * epsilonMp4)):
            return Minit*(1-(t)/(Minit**3/(3 * epsilonMp4)))**(1./3.)
        else:
            return 0
        
    Miev= (3*epsilonMp4*(t))**(1./3.)
    if(Miev<Mfn):
        return romberg(lambda Min: -fPBH_M(Min, Mi, sig_M, alpha)*epsilonMp4/(max([Mt(Min),1e-6*bh.MPL])**2), Miev, Mfn, tol=1e-25, rtol=1e-25)
    else:
        return 0


def test(x):
    print('testons')
    return 2*2*2*2*2*2


def Int_Gamm(grid, t, func_aM_K, ftau_red, fM_max, fa_max, tsol_max, pars):
    #start_tot = time.time()
    Mi, sig_M, ai, sig_a, mDM, alpha = pars
    result = 0
    for i in range(len(grid[:,0,0])):
        for j in range(len(grid[0,:,0])):
            
            Mt, at = func_aM_K(grid[i,j,0], grid[i,j,1], ftau_red, fM_max, fa_max, tsol_max, t)
            result += grid[i,j,2] * bh.Gamma_F(Mt, at, mDM) #* bh.GeV_in_g
    
#     end_tot = time.time()
#     print(end_tot-start_tot)
    return result
    

##################################################################################
#                                                                                #
#                Primordial Black Hole + Dark Matter Generation.                 #
#                           Only DM from evaporation                             #
#                      Considering a Mass Distribution f_BH                      #
#                                                                                #
##################################################################################

import numpy as np
#from odeintw import odeintw
import pandas as pd
from scipy import interpolate, optimize
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import quad, quad_vec, ode, solve_ivp, odeint, romberg, dblquad, fixed_quad
from scipy.optimize import root
from scipy.special import zeta, kn, erf
from scipy.interpolate import interp1d, RectBivariateSpline, UnivariateSpline
import multiprocessing as mp
from scipy.misc import derivative
from numpy import sqrt, log, exp, log10, pi, logspace, linspace, seterr, min, max, append
from numpy import loadtxt, zeros, floor, ceil, unique, sort, cbrt, concatenate, delete, array
import BHProp as bh #Schwarzschild and Kerr BHs library
import Functions_Kerr_Power_Law as fun
import Functions_phik as funcs_phik
from collections import OrderedDict
olderr = np.seterr(all='ignore')
import time
import warnings
warnings.filterwarnings('ignore')
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


colors = ["#2364aa", "#3da5d9", "#73bfb8", "#fec601", "#ea7317"]
cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)


def Derivative(Tab_x,Tab_y):
    if(len(Tab_x)!=len(Tab_y)):
        print('Problem of dimension')
        Tab_yprime=False
    else:
        Tab_y_up=Tab_y[1:len(Tab_y)]
        Tab_y_do=Tab_y[0:len(Tab_y)-1]
        Tab_x_up=Tab_x[1:len(Tab_x)]
        Tab_x_do=Tab_x[0:len(Tab_x)-1]
        Tab_yprime=(Tab_y_up-Tab_y_do)/(Tab_x_up-Tab_x_do)
        
    return Tab_yprime

def f_Derivative(Tab_x,Tab_y):
    if(len(Tab_x)!=len(Tab_y)):
        print('Problem of dimension')
        Tab_yprime=False
    else:
        Tab_y_up=Tab_y[1:len(Tab_y)]
        Tab_y_do=Tab_y[0:len(Tab_y)-1]
        Tab_x_up=Tab_x[1:len(Tab_x)]
        Tab_x_do=Tab_x[0:len(Tab_x)-1]
        Tab_yprime=(Tab_y_up-Tab_y_do)/(Tab_x_up-Tab_x_do)
        
    return interp1d((Tab_x_up+Tab_x_do)/2, Tab_yprime)

def Hubble(rho_tot):
    '''
    units are in GeV
    '''
    Mplanck=1.22*(10**19)
    GeVtoSec=6.67*10**(-25)
    
    return sqrt(8*pi/(3*(Mplanck**2))*rho_tot)



def Grid(N_a,N_m,m_min,m_max,a_min, a_max, pars):
    '''
    2D integral (rectangular) 
    '''
    start_tot = time.time()
    Mi, sig_M, ai, sig_a, mDM, alpha = pars
    dm = (m_max-m_min)/N_m
    da = (a_max-a_min)/N_a
    vec_a = (np.linspace(a_min, a_max - da ,num = N_a) + np.linspace(a_min+da, a_max,num = N_a))/2    
    vec_m = (np.linspace(m_min, m_max - dm ,num = N_m) + np.linspace(m_min+dm, m_max,num = N_m))/2
    result = np.zeros(shape=(N_m,N_a,3))
    
    for i in range(N_m):
        for j in range(N_a):
            result[i,j]=[10**vec_m[i],vec_a[j],dm*da*(log(10.) * (10**vec_m[i]/bh.GeV_in_g)) \
						 * fun.fBH_M_a(10**vec_m[i], vec_a[j], Mi, sig_M, ai, sig_a, alpha)]
    end_tot = time.time()
    print(end_tot-start_tot)
    
    return result


def Grid_1D(N_m,m_min,m_max, pars):
    start_tot = time.time()    
    Mi, sig_M, mDM, alpha = pars
    dm = (m_max-m_min)/N_m
    vec_m = (np.linspace(m_min, m_max - dm ,num = N_m) + np.linspace(m_min+dm, m_max,num = N_m))/2    
    result = np.zeros(shape=(N_m,2))
        
    for i in range(N_m):
        result[i]=[10**vec_m[i],dm*(log(10.) * (10**vec_m[i]/bh.GeV_in_g)) * \
				  fun.fBH_M(10**vec_m[i], Mi, sig_M, alpha)]
    end_tot = time.time()
    print('Grid time:',end_tot-start_tot)
    
    return result


#--------------------------------------------------------------------------------#
#    			       Equations before/during evaporation   				     #
#--------------------------------------------------------------------------------#


def FBEqs(a, v, rPBH_f, drPBHdt_f, rRadi, nPBHi, nphi, Gam_fBH):
	'''
	ADDED BY DONALD ON 15.11.2022
	Radiation density from inflaton decay    
	'''
	
	t     = v[0] # Time in GeV^-1
	rPhik = v[1] # Inflaton energy density in GeV^4  
	rRad  = v[2] # Radiation energy density in GeV^4
	Tp    = v[3] # Temperature in GeV
	NDMH  = v[4] # PBH-induced DM number density

	process = funcs_phik.phik_process().process        		  
	k = funcs_phik.phik_process().kvar
	Mpla = funcs_phik.phik_process().Mp               		 
	lambdac = funcs_phik.phik_funcs(process).lambdavar(k)
	
	# Inflaton decay contribution
	fact_phi = 10.**((-6*k/(k+2))*a)
	fact_rad = 10.**(-4*a)
	gammaphi = funcs_phik.phik_funcs(process).gammaphi(k,lambdac)
	lpar = funcs_phik.phik_process().lparameter(k)
	omega_phi = funcs_phik.phik_funcs(process).omegaphi(k)
	Gamma_phi = gammaphi*np.power(fact_phi*rPhik/Mpla**4,lpar)
	
	Amaxin = funcs_phik.Amax_funcs().Amaxin 
	#Amaxin = funcs_phik.phik_funcs(process).Amaxin 
 	
	# contribution of inflaton to the radiation
	rRadphi = Gamma_phi*fact_phi*rPhik*(1.+omega_phi)
	
	#----------------#
	#   Parameters   #
	#----------------#
	# Hubble parameter
	H = sqrt(8*pi*bh.GCF*(nPBHi*rPBH_f(log10(t))*10.**(-3*a) + rPhik*10**(-(6*k/(k+2))*a) + rRad*10.**(-4*a))/3.) 
	
	# Temperature parameter
	Del = 1. + Tp * bh.dgstarSdT(Tp)/(3.*bh.gstarS(Tp))
	
	#----------------------------------------------#
	#    Radiation + PBH + Temperature equations   #
	#----------------------------------------------#
	dtda     = 1./H
	drPhikda = - rRadphi*10**((6*k/(k+2))*a)/H 
	drRadda  = + rRadphi*10**(4*a)/H + (nPBHi*drPBHdt_f(log10(t)))*(10**a/H)  
	dTda     = - (Tp/Del) * (1.0 - (bh.gstarS(Tp)/bh.gstar(Tp))*(0.25*drRadda/(rRad)))
    
	#-----------------------------------------#
	#           Dark Matter Equations         #
	#-----------------------------------------#
	dNDMHda = 0   # PBH-induced contribution w/o contact
	dEqsda = [dtda, drPhikda, drRadda, dTda, dNDMHda]

	return [x * log(10.) for x in dEqsda]
  

#--------------------------------------------------------------------------------#
#    			      	  Equations after evaporation   				         #
#--------------------------------------------------------------------------------#

def FBEqs_aBE(a, v):
    t     = v[0] # Time in GeV^-1
    rPhik = v[1] # Inflaton energy density in GeV^4  
    rRad  = v[2] # Radiation energy density
    Tp    = v[3] # Temperature (radiation)
    NDMH  = v[4] # Thermal DM number density w/o PBH contribution

    #-------- ADDED BY DONALD ON 15.11.2022 -------#
    #    Radiation density from inflaton decay     #
    #----------------------------------------------#
    process = funcs_phik.phik_process().process      
    k = funcs_phik.phik_process().kvar
    Mpla = funcs_phik.phik_process().Mp               
    lambdac = funcs_phik.phik_funcs(process).lambdavar(k)
 	
    # Inflaton decay contribution
    fact_phi = 10.**((-6*k/(k+2))*a)
    fact_rad = 10.**(-4*a)
    gammaphi = funcs_phik.phik_funcs(process).gammaphi(k,lambdac)
    lpar = funcs_phik.phik_process().lparameter(k)
    omega_phi = funcs_phik.phik_funcs(process).omegaphi(k)
    Gamma_phi = gammaphi*np.power(fact_phi*rPhik/Mpla**4,lpar)
    
    Amaxin = funcs_phik.Amax_funcs().Amaxin 
    #Amaxin = funcs_phik.phik_funcs(process).Amaxin  
	
    # contribution of inflaton to the radiation
    rRadphi = Gamma_phi*fact_phi*rPhik*(1.+omega_phi)
    
    #----------------#
    #   Parameters   #
    #----------------#
    H   = sqrt(8*pi*bh.GCF*(rPhik*10**(-(6*k/(k+2))*a) + rRad*10.**(-4*a))/3.)
    Del = 1. + Tp * bh.dgstarSdT(Tp)/(3.*bh.gstarS(Tp))   
    
    #----------------------------------------#
    #    Radiation + Temperature equations   #
    #----------------------------------------#
    dtda     = 1./H
    drPhikda = 0. - rRadphi*10**((6*k/(k+2))*a)/H  
    drRADda = 0. + rRadphi*10**(4*a)/H
    dTda    = - Tp/Del
    		
    #-----------------------------------------#
    #           Dark Matter Equations         #
    #-----------------------------------------#
    dNDMHda = 0.   # PBH-induced contribution w/o contact
    dEqsda = [dtda, drPhikda, drRADda, dTda, dNDMHda]

    return [x * log(10.) for x in dEqsda]



#----------------------------------------------------------------------------------#
#                                   Input parameters                               #
#----------------------------------------------------------------------------------#

class FBEqs_Sol:

    def __init__(self, MPBHi, aPBHi, bPBHi, width_M, width_a, mDM, al, disc_M, disc_a, Ncores):

        self.MPBHi  = MPBHi 		# Log10[M/1g]
        self.aPBHi  = aPBHi 		# a_star
        self.bPBHi  = bPBHi 		# Log10[beta']
        self.width_M  = width_M
        self.width_a  = width_a
        self.mDM    = mDM
        self.al    = al
        self.Ncores = Ncores
        self.disc_M = disc_M
        self.disc_a = disc_a
        
#----------------------------------------------------------------------------------#
#                                Input parameters                                  #
#----------------------------------------------------------------------------------#
    
    def Solt(self):
        
        Mi = 10**(self.MPBHi)    # PBH initial rotation a_star factor
        ai = self.aPBHi
        bi = 10**(self.bPBHi)    # Initial PBH fraction
        Ncores = self.Ncores
        sig_M    = self.width_M  # Width of the mass distributio
        sig_a    = self.width_a  # Width of the mass distribution
        alpha    = self.al

        # Initial Universe temperature
        process = funcs_phik.phik_process().process      
        k = funcs_phik.phik_process().kvar
        Mpla = funcs_phik.phik_process().Mp  
        
        Amaxin = funcs_phik.Amax_funcs().Amaxin              
        #Amaxin = funcs_phik.phik_funcs(process).Amaxin    

        lambdac = funcs_phik.phik_funcs(process).lambdavar(k)
        lpar = funcs_phik.phik_process().lparameter(k)
        Gklvar = funcs_phik.phik_funcs(process).Gkl(k, lambdac)
        rhoRadi = funcs_phik.phik_funcs(process).rhoRad(k, Amaxin)
        rhoend = funcs_phik.phik_funcs(process).rhoendvar(k)
        rhophi_ini = funcs_phik.phik_funcs(process).rhophi(k, Amaxin, rhoend) 

        Ti_0    = (rhoRadi/(106.75*np.pi**2/30.0))**0.25   # Temperature of radiation at BH formation
        Ti_test =  ((45./(16.*106.75*(pi*bh.GCF)**3.))**0.25)*sqrt(bh.gamma*bh.GeV_in_g/Mi) 
        Ti    =  ((45./(16.*106.75*(pi*bh.GCF)**3.))**0.25)*sqrt(bh.gamma*bh.GeV_in_g/Mi) 

        rPBHi = bi*(rhophi_ini + rhoRadi)                           # Initial PBH energy density
        nphi  = (2.*zeta(3)/pi**2)*Ti**3                            # Initial photon number density
        mDM   = 0.                                                  # DM mass in GeV
        ti    = (sqrt(45./(16.*pi**3.*bh.gstar(Ti)*bh.GCF))*Ti**-2) # Initial time, assuming a radiation dom Universe
        NDMHi = 0. 
        TBHi  = bh.TBH(Mi, ai)                                      # Initial BH temperature
        
        print('Ain, rhoRadi, Ti, rPBHi, rhophi_in  beta_approx = ', Amaxin, rhoRadi, Ti, rPBHi, rhophi_ini, rPBHi/rhophi_ini)
        print('Mi, sigma, beta = ', Mi, sig_M, bi)


        Min = max([0*0.1*bh.MPL,Mi])        # Minimal mass for integration
        Mfn = Mi * 10**sig_M                # Maximal mass for integration
        ain = max([1.e-9,ai-4.*sig_a])      # Minimal spin for integration
        afn = ai+4.*sig_a                   # Maximal spin for integration
        disc_M = self.disc_M
        disc_a = self.disc_a
        
        print("Min, Mfn = ", Min, Mfn)
        
        good=0
        error=1e-10

        def integ(M, ast):
            return fun.Int_rPBH(ast,M,Mi, sig_M, ai, sig_a, alpha)

        Int_i = romberg(lambda logM: quad(lambda ast: np.exp(logM)*integ(np.exp(logM), ast), \
						0, 1, epsabs=1e-16, epsrel=1e-16)[0], np.log(Min), np.log(Mfn), tol=1e-25, rtol=1e-25)

        def integ(M, ast):
            return fun.Int_nPBH(ast,M,Mi, sig_M, ai, sig_a, alpha)

        Int_ni = romberg(lambda logM: quad(lambda ast: np.exp(logM)*integ(np.exp(logM), ast), \
						 0, 1, epsabs=1e-16, epsrel=1e-16)[0], np.log(Min), np.log(Mfn), tol=1e-25, rtol=1e-25)

        if(Int_i<0 or Int_ni<0):
            print('##### PROBLEM OF NORMALIZAtION #####')
        print('integral of f_PBH = ', Int_ni)
        print('integral of M*f_PBH = ', Int_i)
        nPBH_i = rPBHi/(Int_i/bh.GeV_in_g)     ## Initial PBH density, adjusted to give rPBHi defined above

        rPBH_Mp = nPBH_i*bh.mPL
        
        
        #******************************************************************#
        #       Solving for maximal spin and M_1 = 1g.                     # 
        #******************************************************************#
        Mbi=1.e10
        asi=1.
        
        start = time.time()

        tBE, MBHBE, astBE, taut = fun.PBH_time_ev(Mbi, asi, mDM)
        

        end = time.time()
        
        print('------------------')
        print('    alpha = ', alpha, '    ')
        print('------------------')

        print(f"Generic Solution : Runtime is {end - start} s")
        
        # Interpolating the results from the solver to get a function of M(t)/Min in general

        fM_max = interp1d(10.**tBE/10.**taut, MBHBE/Mbi)
        fy_max = interp1d(10.**tBE/10.**taut, -log(astBE/asi))
        fa_max = interp1d(10.**tBE/10.**taut, astBE/asi)
        tsol_max = interp1d(-log(astBE/asi), 10.**tBE/10.**taut) # interpolation of y = -log(a*) vs x=t/tau
        ast_vs_M = interp1d(MBHBE/Mbi, -log(astBE/asi))
        
        #print("MBHBE = ", MBHBE)
        #print("tBE = ", tBE)
        
        ###########################################
        #
        #     map the different possible initial spins
        #
        ###########################################
        nts=10

#         a_ar = linspace(0., 1., num = 500, endpoint=True)

        an = 0.
        ax = 1
        da = (ax - an)/nts

        a_ar = [an + da*i for i in range(nts+1)]


        def run():
            print('Ncores=',Ncores)
            # Open the parallel looping
            pool = mp.Pool(Ncores)
            print('--> Calculating tau_PBH on the grid ')

            # discretize the widths

            start = time.time()
            #---------------------
            tau_PBH_red = pool.starmap(fun.log_tau_a, [(i, an, da, mDM) for i in range(nts+1)])
            #---------------------
            print('--> End--')

            # close the parallel looping
            pool.close()

            return tau_PBH_red


        tau_PBH_red = run()
        
        end = time.time()
        
        ### interpolate the result
        ftau_red = interp1d(a_ar, tau_PBH_red)

        print(f"Runtime is {end - start} s")

        
        ###########################################
        #
        #    2D - Integration
        #
        ###########################################
        
        ### Time discretization
        tin = log10(0.1*ti)
        tfn = log10(3000.*10.**ftau_red(0.)*Mfn**3)  #ftau2d(log10(Mfn), 0.99999)[0,0]
 

        Nframe=1550
        
        dt=(tfn-tin)/(Nframe-1)
        temp = array([tin + i*dt for i in range(Nframe)])
        
        ttot=temp.shape[0]

        
        ## initialize table to store the solution
        
        inRad = zeros(ttot)
        inPBH = zeros(ttot)
        iGamm = zeros(ttot)
        iPBHt = zeros(ttot)
        inPhi = zeros(ttot)
        
        #### Compute the initial distribution weights
        
        Grid_test=Grid_1D(disc_M, log10(Min), log10(Mfn),[Mi, sig_M, mDM, alpha])
        
        def run(Nc):
            
            # Open the parallel looping
            pool = mp.Pool(Nc)
            print('--> Start Parallelizing for integration')

            #---------------------
            
            start_t = time.time()
            iPBHt = pool.starmap(fun.Int_PBH_1D, [(Grid_test, 10.**temp[i],fun.fun_aM_K, ftau_red, \
								fM_max, fa_max, tsol_max,[Mi, sig_M, ai, sig_a, mDM, alpha]) for i in range(ttot)])
            end_t = time.time()
            print(f"Runtime is {end_t - start_t} s")
            print('iPBHt done.')
            #---------------------
            print('--> End Parallelizing ----')

            # close the parallel looping
            pool.close()
            return [iGamm, iPBHt]
        
        
        start_tot = time.time()
        iGamm, iPBHt = run(Ncores)
        print(" ")
        print('ti=',ti)
        end_tot = time.time()

        #print(f"Total Runtime is {end_tot - start_tot} s")
        
        ### interpolate
      
        Gam_fBH = interp1d(temp, iGamm, kind='linear')
        rPBH_fBH = interp1d(temp, np.maximum(np.array(iPBHt)- 0*Int_ni*(bh.mPL), np.zeros(len(iPBHt))), kind='linear')
        
        print('--> Running Boltzmann --')
        
        def drPBH_dt(t): 
            j=-1./log(10)/10.**t
            return derivative(rPBH_fBH, t, dx=1e-5) * j
        
        tin = log10(ti)
        tfn = log10(30.*10.**ftau_red(0.)*Mfn**3)
        if bi > 1.e-19*(1.e9/Mi):
            af = root(bh.afin, [40.], args = (rPBHi, rhoRadi, 10.**tfn, 0.), method='lm', tol=1.e-50) # Scale factor 
            aflog10 = af.x[0]  
        else:
            afw = sqrt(1. + 4.*10.**tfn*sqrt(2.*pi*bh.GCF*(rhoRadi)/3.))
            aflog10 = log10(afw)
        print(tin, tfn, aflog10)

        print(" ")


        fact_nphi = np.power(Amaxin, 6*k/(k+2))
        rhophi_in = fact_nphi * rhophi_ini    

        fact_rPBHi = Amaxin**3
        nPBH_i = fact_rPBHi * nPBH_i  

        fact_rRadi = Amaxin**4
        rRadi = fact_rRadi * rhoRadi      

        v0 = [ti, rhophi_in, rRadi, Ti, 0.]
        
        def stopphi(t, y):
            return y[1]*y[0]**(-(6*k/(k+2))) - 1.0e-70
        stopphi.terminal = True
        stopphi.direction = -1

        def stop(t, y): 
            return nPBH_i*rPBH_fBH(log10(y[0])) - 1.e-70
        stop.terminal = True
        stop.direction = -1
        

        start = time.time()
        # solve ODE
        
        solFBE = solve_ivp(lambda t, z: FBEqs(t, z, rPBH_fBH, drPBH_dt, rhoRadi, nPBH_i, nphi, Gam_fBH), 
                          [np.log10(Amaxin), 2.5*aflog10], v0, events=[stop,stopphi], 
                          Method='BDF', rtol=1.e-8, atol=1.e-10, min_step=0.1)

        end = time.time()

        print(f"Runtime of the program is {end - start} s")

        if not solFBE.success: 
            print(solFBE.message)
            print(aflog10, solFBE.t[0], solFBE.t[-1], log10(solFBE.y[0,0]), log10(solFBE.y[0,-1]))
        
        print()
        print('    log(a) after evaporation = ', solFBE.t[-1], '    ')
        print('------------------')
        print('------------------')
        print('    time after evaporation = ', solFBE.y[0,-1], '  versus theoretical = ',10.**ftau_red(0.)*Mfn**3)
        print('------------------')
        print('------------------')
        print('    Temperature after evaporation = ', solFBE.y[3,-1])
        print()
        
        
        #-----------------------------------------#
        #           After BH evaporation          #
        #-----------------------------------------#

        Tfin = 1.e-6 # Final plasma temp in GeV
        aflog10 = solFBE.t[-1]
        azmax = aflog10 + log10(cbrt(bh.gstarS(solFBE.y[3,-1])/bh.gstarS(Tfin))*(solFBE.y[3,-1]/Tfin))
        afmax = max([aflog10, azmax])
        
        print('e-folds remaining:', -aflog10+azmax)

        v0aBE = [solFBE.y[0,-1], solFBE.y[1,-1], solFBE.y[2,-1], solFBE.y[3,-1], solFBE.y[4,-1]]

        # solve ODE        
        solFBE_aBE = solve_ivp(lambda t, z: FBEqs_aBE(t, z), [aflog10, afmax], v0aBE, events=stopphi,
                               method='Radau',max_step=0.1)

        npaf = solFBE_aBE.t.shape[0]
        tFBE=solFBE.y[0,:]
        PBHFBE=nPBH_i*rPBH_fBH(log10(solFBE.y[0,:]))
        a    = concatenate((solFBE.t[:], solFBE_aBE.t[:]), axis=None)
        t    = concatenate((solFBE.y[0,:], solFBE_aBE.y[0,:]), axis=None) 
        Phik  = concatenate((solFBE.y[1,:], solFBE_aBE.y[1,:]), axis=None)         # phi DK
        Rad  = concatenate((solFBE.y[2,:], solFBE_aBE.y[2,:]), axis=None)    
        PBH  = concatenate((nPBH_i*rPBH_fBH(log10(solFBE.y[0,:])), zeros(npaf)),  axis=None)
        TUn  = concatenate((solFBE.y[3,:], solFBE_aBE.y[3,:]), axis=None)
        NDBE = concatenate((solFBE.y[4,:], solFBE_aBE.y[4,:]), axis=None)
        
        ## EFFECTIVE EoS
        Tab_N= np.log(10**a)        
        lnrho_photons=np.log(Rad/(10**(4*a)))
        lnrho_func = interp1d(Tab_N,lnrho_photons)        
        w_photons=-(1+1/3*Derivative(Tab_N,lnrho_photons))
                
        def dlnrho_dN(t): 
            return derivative(lnrho_func, t, dx=0.1)
        
        dlnrho_dN_v = np.vectorize(dlnrho_dN)
        # restrict the domain since the derivative is ill-defined at the boundaries
        imin=0
        imax=-1

        return [a[imin:imax], t[imin:imax], Phik[imin:imax], Rad[imin:imax], PBH[imin:imax], TUn[imin:imax], NDBE[imin:imax]]

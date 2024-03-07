#!/usr/bin/env python3

import Functions_Kerr_Power_Law as Fun
import multiprocessing
import multiprocess as mp
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import BHProp as bh #Schwarzschild and Kerr BHs library
#from Kerr_python_Power_Law import FBEqs_Sol as Kerr_Sol
from Friedmann import FBEqs_Sol as Kerr_Sol
import Functions_phik as phik_functions
#import reheat_vars as rh_vars
import sys


# Read reheating parameters
rhprocess = phik_functions.phik_process().process 
kvar = phik_functions.phik_process().kvar 
mueff = phik_functions.phik_process().mueff 
sigmaeff = phik_functions.phik_process().sigmaeff 

print('------  Number of arguments:', len(sys.argv), 'arguments --------')
#print(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])


mDM     = float(sys.argv[1])
beta    = float(sys.argv[2])

## chose here the initial mass of the black hole
Delta_LogM   = float(sys.argv[4])
alpha   = float(sys.argv[5])
LogMmin = float(sys.argv[3]) 
ac      = 0
sig_a   = 0.05
disc_M  = 500
disc_a  = 200
Ncores  = 8
tag     = sys.argv[6]

print("alpha = ", alpha)
print("LogMmin, Mmin (g) = ", LogMmin, 10**LogMmin)
print('mDM = 10^', mDM,', beta = 10^', beta,', Mmin = 10^', LogMmin,', Delta_Log_M = ', Delta_LogM)
print()
print('--------------------------------------------------------------------------')
print('--- Distribution run ---')


def f_Kerr(alpha_test):
    return Kerr_Sol(LogMmin, ac, beta, Delta_LogM, sig_a, mDM, alpha_test, disc_M, disc_a,Ncores).Solt()


def run():
    # Open the parallel looping
    pool = mp.Pool(1)
    print('--> Start Parallelizing for scanning')

    start = time.time()
    
    # discretize the widths
    alpha_in = np.array([alpha])
    
    results_vec = pool.map(f_Kerr, [s_in for s_in in alpha_in])
    end = time.time()
    print('The run took ', end-start, ' seconds.')
    print('------------------------')
    # close the parallel looping
    pool.close()

    fig, ax = plt.subplots(2, 1, figsize=(7.,10.))
    colors = plt.cm.plasma(np.linspace(0,1,len(alpha_in)))
    
    print('length : ', len(colors))
    print('length : ', len(alpha_in))

    for i in range(len(alpha_in)):
        a, t, Phi, Rad, PBH, TUn, NDBE  = results_vec[i]

        tag_data = '_logbeta_'+str(int(beta))+'_logMc_'+str(int(LogMmin))+'_sigM_'+str(int(Delta_LogM))+'_siga_'+'_alpha_'+str(int(alpha_in[i]))

        print("WRITING OUT DATA FOR k = {}".format(int(kvar)))
        pre_name = 'Results/k='+str(int(kvar))+'/data_scan_'
                
        incols = ["logbeta","logMc","sigma_M","alpha","rhprocess","kvar","yeff","mueff","sigmaeff", "tag"]
        fd = pd.DataFrame(columns=incols)
        listo = [beta, LogMmin, Delta_LogM, -alpha, rhprocess, kvar, yeff, mueff, sigmaeff, tag]
        fd.loc[len(fd)] = listo
        fd.to_csv(pre_name+"inparameters.dat", header=None, index=None, sep=' ', mode='w+')      

        saving_name = pre_name + tag + '_log10a' + tag_data + '.txt'
        np.savetxt(saving_name, a, delimiter=',')

        saving_name = pre_name + tag + '_a4rhorad' + tag_data + '.txt'
        np.savetxt(saving_name, Rad, delimiter=',')

        saving_name = pre_name + tag + '_funkrhophi' + tag_data + '.txt'
        np.savetxt(saving_name, Phi, delimiter=',')

        saving_name = pre_name + tag + '_a3rhoPBH' + tag_data + '.txt'
        np.savetxt(saving_name, PBH, delimiter=',')

        saving_name = pre_name + tag + '_TPBH' + tag_data + '.txt'
        np.savetxt(saving_name, TUn, delimiter=',')


print('-------------------------------------')
print('-   Starting the run')
start = time.time()
print('-------------------------------------')

run()

print('-------------------------------------')
end = time.time()
print('The Full run took ', end-start, ' seconds.')
print('-------------------------------------')




"""
Parameter inference code based on Bayesian methods.

Uses 2D interpolator for 1D PS from 21-cm forest.

Version 12.2.2024

"""
#Import necessary packages
import sys
import glob, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal as scisi
from astropy.convolution import convolve, Box1DKernel
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
import openpyxl
import time
from scipy import interpolate
from scipy.optimize import minimize
import emcee
import corner
import instrumental_features
from numpy import random
import time
start_clock = time.perf_counter()
#Input parameters
z_name = float(sys.argv[1])         #redshift
dvH = float(sys.argv[2])            #used rebinning for line profile convolution in m/s
spec_res = float(sys.argv[3])       #spectral resolution of the telescope in kHz
path_LOS = '../../datasets/21cmFAST_los/los/'
Nobs = int(sys.argv[4])
Nkbins = int(sys.argv[5])

telescope = ['uGMRT','uGMRT','SKA1-low']
tint = [50,500,50]
S147 = 64.2
alphaR = -0.44

Nlos = 1000
n_los = 1000
Nsamples = 10000


min_logfX = -4.
max_logfX = 1.
#min_fX = 0.0001
#max_fX = 10.
min_xHI = 0.01
max_xHI = 0.99

#Find all of the datasets for the interpolation
files = glob.glob(path_LOS+'*.dat')
files_to_remove = glob.glob(path_LOS+'*fXnone*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])

logfX = np.empty(len(files))
xHI_mean = np.empty(len(files))

#Prepare k bins
d_log_k_bins = 0.25
log_k_bins = np.arange(-1.0-d_log_k_bins/2.,3.+d_log_k_bins/2.,d_log_k_bins)
log_k_bins = log_k_bins[2:2+Nkbins+1]

k_bins = np.power(10.,log_k_bins)
k_bins_cent = np.power(10.,log_k_bins+d_log_k_bins/2.)[:-1]
print(k_bins_cent)

done_perc = 0.1
print('Initiate interpolator setup')
PS_sim_ens = np.empty((len(logfX),Nsamples,len(k_bins_cent)))

#Find all of the parameter values in simulated data and read the data for interpolation
for j in range(len(files)):
   
   data = np.fromfile(str(files[j]),dtype=np.float32)
   logfX[j] = data[9]
   xHI_mean[j] = data[11]

   #Read data for signal
   datafile = str('1DPS_dimensionless/1DPS_signal/power_spectrum_signal_21cmFAST_200Mpc_z%.1f_fX%.2f_xHI%.2f_%dkHz_%dLOS.dat' % (z_name,logfX[j],xHI_mean[j],spec_res,Nlos))
   data = np.fromfile(str(datafile),dtype=np.float32)
   Nlos = int(data[0])
   n_kbins = int(data[1])
   k = data[2:2+n_kbins]
   PS_signal = data[2+n_kbins+0*n_kbins*Nlos:2+n_kbins+1*n_kbins*Nlos]
   PS_signal = np.reshape(PS_signal,(Nlos,n_kbins))[:n_los,:]

   #Bin the PS data
   PS_signal_bin = np.empty((n_los,len(k_bins_cent)))

   for i in range(n_los):
      for l in range(len(k_bins_cent)):
         ind = np.where((k>=k_bins[l]) & (k<k_bins[l+1]))[0]
         PS_signal_bin[i,l] = np.mean(PS_signal[i,ind])

   #Take mean for each k bin
   PS_sim_ens[j,:,:] = instrumental_features.multi_obs(PS_signal_bin,Nobs,Nsamples)
   #PS_sim_mean[j,:] = np.mean(PS_sim_ens,axis=0)

   #Track the progress
   done_files = (j+1)/len(files)
   if done_files>=done_perc:
         print('Done %.2f' % done_files)
         done_perc = done_perc+0.1

#array = np.append(xHI_mean,logfX)
#array = np.append(array,PS_sim_mean)
#array.astype('float32').tofile('datasets/interpolators/interpolator_simwithN_200Mpc_z%.1f_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f_Nobs%d_Nsamples%d.dat' % (z_name,telescope,spec_res,tint,S147,alphaR,Nobs,Nsamples),sep='')

inter_fun_PS21 = interpolate.LinearNDInterpolator(np.transpose([xHI_mean,logfX]),PS_sim_ens)

print('Interpolator set up')
stop_clock = time.perf_counter()
time_taken = (stop_clock-start_clock)
print('It took %.3fs of your life' % time_taken)


xHI_grid = np.arange(min_xHI,max_xHI*1.0001,0.01)
logfX_grid = np.arange(min_logfX,max_logfX*1.0001,0.01)

for i in range(len(telescope)):

  #Read the PS of the noise only
  datafile = str('1DPS_dimensionless/1DPS_noise/power_spectrum_noise_21cmFAST_200Mpc_z%.1f_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f_%dLOS.dat' % (z_name,telescope[i],spec_res,tint[i],S147,alphaR,Nlos))
  data = np.fromfile(str(datafile),dtype=np.float32)
  n_kbins = int(data[1])
  k = data[2:2+n_kbins]
  PS_noise = np.reshape(data[2+n_kbins+0*n_kbins*Nlos:2+n_kbins+1*n_kbins*Nlos],(Nlos,n_kbins))

  PS_noise_bin = np.empty((Nlos,len(k_bins_cent)))

  for l in range(Nlos):

    for j in range(len(k_bins_cent)):
      ind = np.where((k>=k_bins[j]) & (k<k_bins[j+1]))[0]
      PS_noise_bin[l,j]  = np.mean(PS_noise[l,ind])

  PS_noise_med = np.mean(PS_noise_bin,axis=0)

  print('Noise PS read for %s, t_int=%dhr' % (telescope[i],tint[i]))

  xHI_lim_68 = np.empty(len(xHI_grid))
  xHI_lim_68[:] = np.nan
  logfX_lim_68 = np.empty(len(xHI_grid))
  logfX_lim_68[:] = np.nan

  xHI_lim_95 = np.empty(len(xHI_grid))
  xHI_lim_95[:] = np.nan
  logfX_lim_95 = np.empty(len(xHI_grid))
  logfX_lim_95[:] = np.nan

  done_perc = 0.1

  for j in range(len(xHI_grid)):
     
     jj = 0

     limit = 0.32

     while True:
               
        PS_signal = inter_fun_PS21(xHI_grid[j],logfX_grid[jj])
        Detections = 0

        for los in range(Nsamples):
           
           if np.all(PS_noise_med<PS_signal[los]):
              Detections += 1  #If the signal is above the noise in all k bins, then it is detected. If this is met, add 1 to Detections

        #print('x_HI=%.4f, logfX=%.4f, Nondection fraction =%.6f' % (xHI_grid[j],logfX_grid[jj],Nondetection/Nsamples))

        if Detections/Nsamples<=limit: #If the <=32% of LOS are detection, then 68% are in non-detection and this is the 68% limit
           xHI_lim_68[j] = xHI_grid[j]
           logfX_lim_68[j] = logfX_grid[jj]
           limit = -1.1

        if Detections/Nsamples<=0.05:  #If the <=5% of LOS are detection, then 95% are in non-detection and this is the 95% limit
           xHI_lim_95[j] = xHI_grid[j]
           logfX_lim_95[j] = logfX_grid[jj]
           break  #If the 95% limit is found, then break the loop

        elif jj==len(logfX_grid)-1:
           break  #If gone through all the values and the 95% limit is not found, then break the loop. The limit will be NaN

        jj += 1

     #print('xHI_lim_68 = %.4f, logfX_lim_68= %.4f' % (xHI_lim_68[j],logfX_lim_68[j]))
     #print('xHI_lim_95 = %.4f, logfX_lim_95= %.4f' % (xHI_lim_95[j],logfX_lim_95[j]))

     #Track the progress
     done_files = (j+1)/len(xHI_grid)
     if done_files>=done_perc:
        print('Done %.2f' % done_files)
        done_perc = done_perc+0.1

  print(xHI_lim_68)
  print(logfX_lim_68)
  print(xHI_lim_95)
  print(logfX_lim_95)
  #Save the limits in files
  array = np.append(xHI_lim_68,logfX_lim_68)
  array = np.append(array,xHI_lim_95)
  array = np.append(array,logfX_lim_95)
  array.astype('float32').tofile('datasets/interpolators/nulldetection2_limits_200Mpc_z%.1f_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f_Nobs%d_Nkbins%d.dat' % (z_name,telescope[i],spec_res,tint[i],S147,alphaR,Nobs,Nkbins),sep='')

  stop_clock = time.perf_counter()
  time_taken = (stop_clock-start_clock)
  print('It took %.3fs of your life' % time_taken)
"""
Generating 21cm forest 1D PS data for signal+noise.

Version 26.04.2024
"""

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

start_clock = time.perf_counter()

#constants

import instrumental_features
import PS1D



#path = 'data/'
path = '../../datasets/21cmFAST_los/'
z_name = float(sys.argv[1])
dvH = float(sys.argv[2])
spec_res = float(sys.argv[3])
n_los = int(sys.argv[4])
telescope = str(sys.argv[5])
S_min_QSO = float(sys.argv[6])
alpha_R = float(sys.argv[7])
N_d = float(sys.argv[8])
t_int = float(sys.argv[9])

#Find all of the datasets for the interpolation
files = glob.glob(path+'los/*.dat')

files_to_remove = glob.glob(path+'los/*fX-10.0*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])
files_to_remove = glob.glob(path+'los/*xHI1.*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])
files_to_remove = glob.glob(path+'los/*xHI0.9*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])
files_to_remove = glob.glob(path+'los/*xHI0.8*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])
files_to_remove = glob.glob(path+'los/*xHI0.7*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])
files_to_remove = glob.glob(path+'los/*xHI0.69*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])
files_to_remove = glob.glob(path+'los/*xHI0.68*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])
files_to_remove = glob.glob(path+'los/*xHI0.67*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])
files_to_remove = glob.glob(path+'los/*xHI0.66*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])
files_to_remove = glob.glob(path+'los/*xHI0.65*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])
files_to_remove = glob.glob(path+'los/*xHI0.64*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])

files = ['../../datasets/21cmFAST_los/los/los_50Mpc_256_n1000_z6.000_fX-0.6_xHI0.24.dat'
        ,'../../datasets/21cmFAST_los/los/los_50Mpc_256_n1000_z6.000_fX-0.6_xHI0.79.dat']



logfX = np.empty(len(files))
xHI_mean = np.empty(len(files))

#Find all of the parameter values in simulated data and read the data for interpolation
for j in range(len(files)):
    
    data = np.fromfile(str(files[j]),dtype=np.float32)
    logfX[j] = data[9]
    xHI_mean[j] = data[11]

Nlos = 1000

datafile = str('%stau_long/los_200Mpc_n%d_z%.3f_dv%d.dat' %(path,Nlos,z_name,dvH))
data  = np.fromfile(str(datafile),dtype=np.float32)
z     = data[0]	#redshift
Nbins = int(data[2])					#Number of pixels/cells/bins in one line-of-sight
Nlos = int(data[3])						#Number of lines-of-sight
x_initial = 4
vel_axis = data[(x_initial+Nbins):(x_initial+2*Nbins)]#Hubble velocity along LoS in km/s
freq_ori = instrumental_features.freq_obs(z,vel_axis*1e5)

Nbins = len(freq_ori)
print('Number of pixels (original): %d' % Nbins)
freq_uni = instrumental_features.uni_freq(freq_ori,np.array([freq_ori]))[0]
freq_smooth = instrumental_features.smooth_fixedbox(freq_uni,freq_uni,spec_res)[0]
bandwidth = (freq_smooth[-1]-freq_smooth[0])/1e6
print('Number of pixels (smoothed): %d' % len(freq_smooth))
print('Bandwidth = %.2fMHz' % bandwidth)
n_kbins = int((len(freq_smooth)/2+1))

for l in range(len(xHI_mean)):

  print('<xHI>=%.2f, log10(fX)=%.2f, %d/%d' % (xHI_mean[l],logfX[l],l+1,len(xHI_mean)))

  datafile = str('%stau_long/tau_200Mpc_n%d_z%.3f_fX%.1f_xHI%.2f_dv%d.dat' %(path,Nlos,z_name,logfX[l],xHI_mean[l],dvH))
  data = np.fromfile(str(datafile),dtype=np.float32)
  tau = np.reshape(data,(Nlos,Nbins))

  tau = tau[:n_los,:]
  signal_ori = instrumental_features.transF(tau)
  freq_uni,signal_uni = instrumental_features.uni_freq(freq_ori,signal_ori)

  PS_signal = np.empty((n_los,n_kbins))

  done_perc = 0.1

  for j in range(n_los):

    freq_smooth,signal_smooth = instrumental_features.smooth_fixedbox(freq_uni,signal_uni[j],spec_res)
    freq_smooth = freq_smooth[:-1]
    signal_smooth = signal_smooth[:-1]
    noise = instrumental_features.add_noise(freq_smooth,telescope,spec_res,S_min_QSO,alpha_R,t_int,N_d)
    k,PS_signal[j,:] = PS1D.get_P(signal_smooth+noise,bandwidth)

    done_LOS = (j+1)/n_los
    if done_LOS>=done_perc:
      print('Done %.2f' % done_LOS)
      done_perc = done_perc+0.1

  array = np.append(n_los,n_kbins)
  array = np.append(array,k)
  array = np.append(array,PS_signal)
  array.astype('float32').tofile('1DPS_dimensionless/1DPS_signalandnoise/power_spectrum_signal_21cmFAST_200Mpc_z%.1f_fX%.2f_xHI%.2f_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f_%dLOS.dat' % (z,logfX[l],xHI_mean[l],telescope,spec_res,t_int,S_min_QSO,alpha_R,n_los),sep='')

  stop_clock = time.perf_counter()
  time_taken = (stop_clock-start_clock)
  print('It took %.3fs to run' % time_taken)

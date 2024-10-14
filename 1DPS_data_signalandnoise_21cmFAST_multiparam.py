"""
Generating 21cm forest 1D PS data for signal+noise.

Version 26.04.2024
"""

import sys
import glob, os
import numpy as np
from astropy.convolution import convolve, Box1DKernel
import time

import instrumental_features
import PS1D

start_clock = time.perf_counter()

path = 'datasets/21cmFAST_los/'
z_name = float(sys.argv[1])	   #redshift
dvH = float(sys.argv[2])         #rebinning width in km/s
spec_res = float(sys.argv[3])	   #spectral resolution of telescope (i.e. frequency channel width) in kHz
n_los = int(sys.argv[4])         #number of lines-of-sight
telescope = str(sys.argv[5])	   #telescope
N_d = int(sys.argv[6])			   #number dishes used by telescope
S_min_QSO = float(sys.argv[7])   #intrinsic flux of QSO at 147Hz in mJy
alpha_R = float(sys.argv[8])	   #radio spectral index of QSO
t_int = float(sys.argv[9])		   #integration time of observation in hours

#Find all of the datasets for the interpolation
files = glob.glob(path+'los/*.dat')
#Remove some if needed
files_to_remove = glob.glob(path+'los/*fXnone*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])

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
z     = data[0]      #redshift
Nbins = int(data[2])	#Number of pixels/cells/bins in one line-of-sight
Nlos = int(data[3])	#Number of lines-of-sight
x_initial = 4
vel_axis = data[(x_initial+Nbins):(x_initial+2*Nbins)]   #Hubble velocity along LoS in km/s
freq_ori = instrumental_features.freq_obs(z,vel_axis*1e5)

#Incorporate spectral resolution of telescope
Nbins = len(freq_ori)
print('Number of pixels (original): %d' % Nbins)
freq_uni = instrumental_features.uni_freq(freq_ori,np.array([freq_ori]))[0]
freq_smooth = instrumental_features.smooth_fixedbox(freq_uni,freq_uni,spec_res)[0]
bandwidth = (freq_smooth[-1]-freq_smooth[0])/1e6
print('Number of pixels (smoothed): %d' % len(freq_smooth))
print('Bandwidth = %.2fMHz' % bandwidth)
n_kbins = int((len(freq_smooth)/2+1))

#Calculate 1D power spectrum for all parameter values
for l in range(len(xHI_mean)):

  print('<xHI>=%.2f, log10(fX)=%.2f, %d/%d' % (xHI_mean[l],logfX[l],l+1,len(xHI_mean)))

  #Load data for optical depth and turn to signal
  datafile = str('%stau_long/tau_200Mpc_n%d_z%.3f_fX%.1f_xHI%.2f_dv%d.dat' %(path,Nlos,z_name,logfX[l],xHI_mean[l],dvH))
  data = np.fromfile(str(datafile),dtype=np.float32)
  tau = np.reshape(data,(Nlos,Nbins))

  tau = tau[:n_los,:]
  signal_ori = instrumental_features.transF(tau)
  freq_uni,signal_uni = instrumental_features.uni_freq(freq_ori,signal_ori) #Interpolate signal to uniform frequency grid

  PS_signal = np.empty((n_los,n_kbins))

  done_perc = 0.1

  for j in range(n_los):

    freq_smooth,signal_smooth = instrumental_features.smooth_fixedbox(freq_uni,signal_uni[j],spec_res)   #Incorporate spectral resolution of telescope
    freq_smooth = freq_smooth[:-1]
    signal_smooth = signal_smooth[:-1]
    noise = instrumental_features.add_noise(freq_smooth,telescope,spec_res,S_min_QSO,alpha_R,t_int,N_d)  #and add noise
    k,PS_signal[j,:] = PS1D.get_P(signal_smooth+noise,bandwidth)  #Calculate 1D power spectrum

    done_LOS = (j+1)/n_los
    if done_LOS>=done_perc:
      print('Done %.2f' % done_LOS)
      done_perc = done_perc+0.1

  #Write data
  array = np.append(n_los,n_kbins)
  array = np.append(array,k)
  array = np.append(array,PS_signal)
  array.astype('float32').tofile('1DPS_dimensionless/1DPS_signalandnoise/power_spectrum_signal_21cmFAST_200Mpc_z%.1f_fX%.2f_xHI%.2f_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f_%dLOS.dat' % (z,logfX[l],xHI_mean[l],telescope,spec_res,t_int,S_min_QSO,alpha_R,n_los),sep='')

  stop_clock = time.perf_counter()
  time_taken = (stop_clock-start_clock)
  print('It took %.3fs to run' % time_taken)

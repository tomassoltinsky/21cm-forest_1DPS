"""
Generating 21cm forest 1D PS data for noise.

Version 19.10.2023
"""

import sys
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

path = '../../datasets/21cmFAST_los/'
z_name = float(sys.argv[1])
dvH = float(sys.argv[2])
telescope = str(sys.argv[3])
spec_res = float(sys.argv[4])
S147 = float(sys.argv[5])
alphaR = float(sys.argv[6])
N_d = float(sys.argv[7])
tint = float(sys.argv[8])
n_los = 1000
fX_name = -2.
mean_xHI = 0.25

datafile = str('%stau_long/los_200Mpc_n%d_z%.3f_dv%d.dat' %(path,1000,z_name,dvH))
data  = np.fromfile(str(datafile),dtype=np.float32)
z     = data[0]	#redshift
Nbins = int(data[2])					#Number of pixels/cells/bins in one line-of-sight
Nlos = int(data[3])						#Number of lines-of-sight
x_initial = 4
vel_axis = data[(x_initial+Nbins):(x_initial+2*Nbins)]#Hubble velocity along LoS in km/s
freq_ori = instrumental_features.freq_obs(z,vel_axis*1e5)

freq_uni = instrumental_features.uni_freq(freq_ori,np.array([freq_ori]))[0]
freq_smooth = instrumental_features.smooth_fixedbox(freq_uni,freq_uni,spec_res)[0]
freq_smooth = freq_smooth[:-1]
bandwidth = (freq_smooth[-1]-freq_smooth[0])/1e6
print('Bandwidth = %.2fMHz' % bandwidth)

n_kbins = int((len(freq_smooth)/2+1))
PS_noise = np.empty((n_los,n_kbins))

done_perc = 0.1

for j in range(n_los):

  noise = instrumental_features.add_noise(freq_smooth,telescope,spec_res,S147,alphaR,tint,N_d)
  k,PS_noise[j,:] = PS1D.get_P(1.+noise,bandwidth)

  done_LOS = (j+1)/n_los
  if done_LOS>=done_perc:
    print('Done %.2f' % done_LOS)
    done_perc = done_perc+0.1

array = np.append(n_los,n_kbins)
array = np.append(array,k)
array = np.append(array,PS_noise)
array.astype('float32').tofile('1DPS_dimensionless/1DPS_noise/power_spectrum_noise_21cmFAST_200Mpc_z%.1f_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f_%dLOS.dat' % (z_name,telescope,spec_res,tint,S147,alphaR,n_los),sep='')

stop_clock = time.perf_counter()
time_taken = (stop_clock-start_clock)
print('This took %.3fs of your life' % time_taken)
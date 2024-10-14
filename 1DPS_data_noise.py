"""
Generating 21cm forest 1D PS data for noise only.

Version 19.10.2023
"""

import sys
import numpy as np
from astropy.convolution import convolve, Box1DKernel
import time

import instrumental_features
import PS1D

start_clock = time.perf_counter()

#Input parameters
path = 'datasets/21cmFAST_los/'

z_name = float(sys.argv[1])		#redshift
dvH = float(sys.argv[2])		  #rebinning width in km/s
spec_res = float(sys.argv[3])	#spectral resolution of telescope (i.e. frequency channel width) in kHz
telescope = str(sys.argv[4])	#telescope
N_d = int(sys.argv[5])			  #number dishes used by telescope
S_min_QSO = float(sys.argv[6])	    #intrinsic flux of QSO at 147Hz in mJy
alpha_R = float(sys.argv[7])	  #radio spectral index of QSO
t_int = float(sys.argv[8])		  #integration time of observation in hours

n_los = 1000
fX_name = -2.
mean_xHI = 0.25

#Load data for velocity-axis and turn to frequency and redshift axis
datafile = str('%stau_long/los_200Mpc_n%d_z%.3f_dv%d.dat' %(path,1000,z_name,dvH))
data  = np.fromfile(str(datafile),dtype=np.float32)
z     = data[0]	#redshift
Nbins = int(data[2])					#Number of pixels/cells/bins in one line-of-sight
Nlos = int(data[3])						#Number of lines-of-sight
x_initial = 4
vel_axis = data[(x_initial+Nbins):(x_initial+2*Nbins)]#Hubble velocity along LoS in km/s
freq_ori = instrumental_features.freq_obs(z,vel_axis*1e5)

#Incorporate spectral resolution of telescope
freq_uni = instrumental_features.uni_freq(freq_ori,np.array([freq_ori]))[0]
freq_smooth = instrumental_features.smooth_fixedbox(freq_uni,freq_uni,spec_res)[0]
freq_smooth = freq_smooth[:-1]
bandwidth = (freq_smooth[-1]-freq_smooth[0])/1e6
print('Bandwidth = %.2fMHz' % bandwidth)

n_kbins = int((len(freq_smooth)/2+1))
PS_noise = np.empty((n_los,n_kbins))

done_perc = 0.1

#Compute the noise for each LOS and calculate 1D power spectrum
for j in range(n_los):

  noise = instrumental_features.add_noise(freq_smooth,telescope,spec_res,S_min_QSO,alpha_R,t_int,N_d)
  k,PS_noise[j,:] = PS1D.get_P(1.+noise,bandwidth)

  done_LOS = (j+1)/n_los
  if done_LOS>=done_perc:
    print('Done %.2f' % done_LOS)
    done_perc = done_perc+0.1

#Write data
array = np.append(n_los,n_kbins)
array = np.append(array,k)
array = np.append(array,PS_noise)
array.astype('float32').tofile('1DPS_dimensionless/1DPS_noise/power_spectrum_noise_21cmFAST_200Mpc_z%.1f_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f_%dLOS.dat' % (z_name,telescope,spec_res,t_int,S_min_QSO,alpha_R,n_los),sep='')

stop_clock = time.perf_counter()
time_taken = (stop_clock-start_clock)
print('This took %.3fs of your life' % time_taken)
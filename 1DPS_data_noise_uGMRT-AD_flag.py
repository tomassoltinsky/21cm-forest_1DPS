"""
Generating 21-cm forest 1D PS data for forward-modelled noise-only sightlines
with spectral channels masked according to the flagging of the uGMRT archival
observations.

The same WLD residual bandpass subtraction and corrupted-channel identification
used for the observational data are applied to the off-source spectra to define
the frequency-channel mask. This mask is then applied to synthetic Gaussian
noise realizations when calculating their 1D power spectra using the periodogram
method for masked data.

The instrumental noise rms in flux-density units is assumed to be frequency
independent and is set by the measured uGMRT archival noise level. The fractional
noise relative to the QSO continuum remains frequency dependent due to the radio
spectral index.

Version 6.6.2026
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
import pywt                                 #for residual bandpass feature mitigation using wavelet transform

start_clock = time.perf_counter()

import instrumental_features
import PS1D


#Input parameters
path = 'datasets/21cmFAST_los/'             #Path to the simulated 21cmFAST LOS data
z_name = float(sys.argv[1])                 #Redshift of the 21cmFAST LOS data
dvH = float(sys.argv[2])                    #Velocity resolution of the LOS data in m/s
telescope = str(sys.argv[3])                #Telescope name
spec_res = float(sys.argv[4])               #Spectral resolution in kHz
S147 = float(sys.argv[5])                   #Intrinsic QSO flux density at 147 MHz in mJy
alphaR = float(sys.argv[6])                 #Radio spectral index of the QSO
rms = float(sys.argv[7])                    #Frequency-independent instrumental noise rms in mJy/beam
WLDlvl = int(sys.argv[8])                   #Level of wavelet decomposition used for residual bandpass subtraction
f_flag = float(sys.argv[9])                 #Threshold for flagging bad channels in units of mean off-source noise rms
n_los = 1000                                #Number of synthetic noise realizations


#Read the off-source spectra from the uGMRT observations
print('Off-source spectra')

datafile = str('obs_data/common_freq.txt')
freq_noise = np.flip(np.loadtxt(datafile))
freq_noise = freq_noise[0:-1]
bandwidth = (freq_noise[-1]-freq_noise[0])                  #Observational bandwidth in MHz
spec_res = (freq_noise[1]-freq_noise[0])*1e3                #Spectral resolution in kHz
print('Bandwidth           = %.2f MHz' % bandwidth)
print('Spectral resolution = %.2f kHz' % spec_res)
print('Number of channels  = %d' % len(freq_noise))
print('Frequency range     = %.2f-%.2f MHz' % (freq_noise[0],freq_noise[-1]))

datafile = str('obs_data/off_source_spectra_common_band_all_night_no_filter.txt')
F_noise = np.flip(np.loadtxt(datafile),axis=1)
F_noise = F_noise[:,0:-1]
Nlos_noise = F_noise.shape[0]                              #Number of off-source spectra

noise_rms = np.std(F_noise, axis=0)                        #Noise rms in individual frequency channels
mean_noise = np.std(F_noise)                               #Mean rms of the off-source spectra
print('RMS noise from off-source spectra = %.2f mJy/bm' % mean_noise)


#Post-process the off-source spectra in the same way as the observational data
print('Post-processing')

#Smoothing with wavelet transform for residual bandpass feature subtraction
def WLD(signal,wavelet_type,level):

  #Parameters
  wavelet = wavelet_type      #Daubechies 8 is good for smooth signals
  detail_keep_level = level   #Level of detail coefficients to keep (lower = smoother fit)

  #Choose wavelet and decomposition level
  max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)

  #Perform wavelet decomposition
  coeffs = pywt.wavedec(signal, wavelet=wavelet, level=max_level)

  #Suppress only fine-scale noise
  #Keep the approximation + low-order details and suppress high-order details
  coeffs_denoised = []
  for i, c in enumerate(coeffs):
      if i == 0 or i <= detail_keep_level:
          coeffs_denoised.append(c)  #Keep approximation + smooth details
      else:
          coeffs_denoised.append(np.zeros_like(c))  #Suppress noise

  #Reconstruct the denoised flux
  bandpassfit = pywt.waverec(coeffs_denoised, wavelet)

  #Ensure same length (sometimes waverec returns slightly longer array)
  bandpassfit = bandpassfit[:len(signal)]

  return bandpassfit


#Subtract residual bandpass features from the off-source spectra
print('WLD --------------------------------------------------')
F_noise_afterWLD = np.empty_like(F_noise)
bandpassfit_noise = np.empty_like(F_noise)
for i in range(Nlos_noise):
  bandpassfit_noise[i,:] = WLD(F_noise[i,:],'db8',level=WLDlvl)
  F_noise_afterWLD[i,:] = F_noise[i,:] - bandpassfit_noise[i,:]

noise_rms = np.std(F_noise_afterWLD, axis=0)                #Noise rms after WLD subtraction
mean_noise = np.std(F_noise_afterWLD[~np.isnan(F_noise_afterWLD)])
print('RMS noise after WLD subtraction = %.2f mJy/bm' % mean_noise)


#Flag corrupted channels based on noise peaks
print('Flagging corrupted channels -------------------------------------')

#Flag noise peaks above the specified threshold in terms of the average noise rms
#20 frequency channels around each detected peak are flagged
ind_flag, peak_props = scisi.find_peaks(noise_rms, height=f_flag*mean_noise, distance=11)
freq_noise_flag = freq_noise[ind_flag]

ind_flagflag = np.empty(0,dtype=int)

#Print the positions of the flagged peaks and remember them
print('Flagged peaks = %d' % len(ind_flag))
for i in range(len(ind_flag)):
  print('  Peak %d: channel = %d, freq = %.3f MHz, noise_rms = %.3f mJy/bm' %
        (i+1, ind_flag[i], freq_noise[ind_flag[i]], noise_rms[ind_flag[i]]))

  ind_flagflag = np.append(ind_flagflag, np.arange(ind_flag[i]-10, ind_flag[i]+10))

#Take each channel only once and ensure that it is within the spectrum
ind_flagflag = np.unique(ind_flagflag)
ind_flagflag = ind_flagflag[(ind_flagflag >= 0) & (ind_flagflag < len(freq_noise))]

#Prepare boolean mask used for the synthetic noise 1D PS calculation
mask = np.ones(len(F_noise_afterWLD[0]), dtype=bool)
mask[ind_flagflag] = False


#Prepare the frequency grid for the synthetic noise realizations
print('Start generating noise 1DPS')

datafile = str('%stau_long/los_200Mpc_n%d_z%.3f_dv%d.dat' %(path,1000,z_name,dvH))
data  = np.fromfile(str(datafile),dtype=np.float32)
z     = data[0]                                            #Redshift
Nbins = int(data[2])                                       #Number of pixels/cells/bins in one line-of-sight
Nlos = int(data[3])                                        #Number of lines-of-sight
x_initial = 4                                              #Index where LOS data arrays start
vel_axis = data[(x_initial+Nbins):(x_initial+2*Nbins)]     #Hubble velocity along LOS in km/s
freq_ori = instrumental_features.freq_obs(z,vel_axis*1e5) #Observed frequency along the simulated LOS

#Determine the dimensions after smoothing to the observational spectral resolution
freq_uni = instrumental_features.uni_freq(freq_ori,np.array([freq_ori]))[0]
freq_smooth = instrumental_features.smooth_fixedbox(freq_uni,freq_uni,spec_res)[0]
freq_smooth = freq_smooth#[:-1]
bandwidth = (freq_smooth[-1]-freq_smooth[0])/1e6
print('Simulated LOS bandwidth = %.2f MHz' % bandwidth)
n_kbins = int((len(freq_smooth)/2+1))                      #Number of Fourier modes
PS_noise = np.empty((n_los,n_kbins))                       #Noise 1D PS for each realization


#Prepare the frequency range corresponding to the uGMRT forest data
bandwidth = 13.47                                          #Bandwidth used for the 1D PS in MHz
z_T = 5.82                                                 #Redshift of the target J352-15
nu_T = 209.002                                             #Observed 21-cm frequency at the target redshift in MHz
freq_range = np.arange(0,bandwidth/1.*1.00001,spec_res*1e-3)
freq_smooth = (nu_T + freq_range)*1e6
T_ind = int(freq_smooth.shape[0])                          #Number of frequency channels used for the 1D PS

bandwidth = (freq_smooth[-1]-freq_smooth[0])/1e6
print('Target redshift              = %.2f' % z_T)
print('Target frequency             = %.2f MHz' % nu_T)
print('Number of frequency channels = %d' % len(freq_smooth))
print('Frequency range              = %.2f-%.2f MHz' % (freq_smooth[0]/1e6,freq_smooth[-1]/1e6))
print('Bandwidth                    = %.2f MHz' % bandwidth)
n_kbins = int((len(freq_smooth)/2+1))                      #Number of Fourier modes
PS_noise = np.empty((n_los,n_kbins))                       #Noise 1D PS for each realization

done_perc = 0.1                                            #Progress-printing interval

#Calculate the QSO continuum and fractional instrumental noise across the band
S_v = S147*np.power(freq_smooth/147.e6,alphaR)             #QSO flux density as a function of frequency in mJy
sigma_noise = rms/S_v                                      #Fractional noise relative to the QSO continuum

print('QSO flux at 147 MHz          = %.2f mJy' % S147)
print('Radio spectral index         = %.2f' % alphaR)
print('Instrumental noise rms       = %.2f mJy/beam' % rms)
print('Number of noise realizations = %d' % n_los)


#Generate the synthetic noise 1D power spectra
for j in range(n_los):

  #Generate Gaussian noise with frequency-independent rms in flux-density units
  noise = np.random.normal(0.,sigma_noise,len(freq_smooth))

  #Apply the observational channel mask and calculate the masked 1D PS
  noise[ind_flagflag] = np.nan
  k,PS_noise[j,:] = PS1D.get_P_masked(1.+noise,mask,bandwidth)

  #Print progress every 10 per cent
  done_LOS = (j+1)/n_los
  if done_LOS>=done_perc:
    print('Done %.2f' % done_LOS)
    done_perc = done_perc+0.1


#Save the synthetic noise 1D PS ensemble
array = np.append(n_los,n_kbins)
array = np.append(array,k)
array = np.append(array,PS_noise)
array.astype('float32').tofile('1DPS_dimensionless/1DPS_noise/power_spectrum_noise_mask_21cmFAST_%.1f-%.1fMHz_z%.1f_WLDlvl%d_fflag%.2f_%s_%.1fkHz_Smin%.1fmJy_alphaR%.2f_rms%.4fmJy_%dLOS.dat' % (freq_smooth[0]/1e6,np.round(freq_smooth[-1]/1e6,0),z_name,WLDlvl,f_flag,telescope,spec_res,S147,alphaR,rms,n_los),sep='')

#Print total execution time
stop_clock = time.perf_counter()
time_taken = (stop_clock-start_clock)
print('This took %.3fs of your life' % time_taken)
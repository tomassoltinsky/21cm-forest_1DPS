"""
Generating 21-cm forest 1D PS data for the signal with spectral channels masked
according to the flagging of the uGMRT archival observations.

The same WLD residual bandpass subtraction and corrupted-channel identification
used for the observational data are applied to the off-source spectra to define
the frequency-channel mask. This mask is then applied to forward-modelled
21cmFAST spectra when calculating their 1D power spectra using the periodogram
method for masked data.

Version 06.06.2026
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
import pywt                                 #for residual bandpass feature mitigation using wavelet transform

start_clock = time.perf_counter()

import instrumental_features
import PS1D


#Input parameters
z_name = float(sys.argv[1])                 #Redshift of the 21cmFAST LOS data
dvH = float(sys.argv[2])                    #Velocity resolution of the LOS data in m/s
spec_res = float(sys.argv[3])               #Spectral resolution in kHz
n_los = int(sys.argv[4])                    #Number of simulated LOS used to calculate the 1D PS
WLDlvl = int(sys.argv[5])                   #Level of wavelet decomposition used for residual bandpass subtraction
f_flag = float(sys.argv[6])                 #Threshold for flagging bad channels in units of mean off-source noise rms


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

#Prepare boolean mask used for the simulated 1D PS calculation
mask = np.ones(len(F_noise_afterWLD[0]), dtype=bool)
mask[ind_flagflag] = False


#Prepare the 21cmFAST signal simulations
print('Start generating signal 1DPS')
path = 'datasets/21cmFAST_los/'                            #Path to the simulated 21cmFAST LOS data

#Find all of the datasets
files = glob.glob(path+'los/*.dat')
files_to_remove = glob.glob(path+'los/*fXnone*.dat')       #Ignore the IGM models with no X-ray heating
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])
files_to_remove = glob.glob(path+'los/*seed*.dat')         #Ignore the IGM models with different random seeds
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])

#Or select which IGM models to use manually
files = ['datasets/21cmFAST_los/los/los_50Mpc_256_n1000_z6.000_fX-3.0_xHI0.80.dat'
        ,'datasets/21cmFAST_los/los/los_50Mpc_256_n1000_z6.000_fX-3.0_xHI0.11.dat']

logfX = np.empty(len(files))                               #log10 of X-ray pre-heating efficiency for each model
xHI_mean = np.empty(len(files))                            #Mean neutral hydrogen fraction for each model
print('Number of files: %d' % len(files))

#Read the IGM parameter values from the selected simulated datasets
for j in range(len(files)):
    
    data = np.fromfile(str(files[j]),dtype=np.float32)
    logfX[j] = data[9]
    xHI_mean[j] = data[11]

Nlos = 20                                                  #Number of LOS in the optical-depth simulation files

#Read the 21cmFAST LOS coordinates
datafile = str('%stau_long/los_200Mpc_n%d_z%.3f_dv%d.dat' %(path,Nlos,z_name,dvH))
data  = np.fromfile(str(datafile),dtype=np.float32)
z     = data[0]                                            #Redshift
Nbins = int(data[2])                                       #Number of pixels/cells/bins in one line-of-sight
Nlos = int(data[3])                                        #Number of lines-of-sight
x_initial = 4                                              #Index where LOS data arrays start
vel_axis = data[(x_initial+Nbins):(x_initial+2*Nbins)]     #Hubble velocity along LOS in km/s
freq_ori = instrumental_features.freq_obs(z,vel_axis*1e5) #Observed frequency along the simulated LOS

#Determine the dimensions after smoothing to the observational spectral resolution
Nbins = len(freq_ori)
print('Number of pixels (original) = %d' % Nbins)
freq_uni = instrumental_features.uni_freq(freq_ori,np.array([freq_ori]))[0]
freq_smooth = instrumental_features.smooth_fixedbox(freq_uni,freq_uni,spec_res)[0]
bandwidth = (freq_smooth[-1]-freq_smooth[0])/1e6
print('Number of pixels (smoothed) = %d' % len(freq_smooth))
print('Simulated LOS bandwidth     = %.2f MHz' % bandwidth)
n_kbins = int((len(freq_smooth)/2+1))                      #Number of Fourier modes

#Prepare the frequency range corresponding to the uGMRT forest data
bandwidth = 13.47                                          #Bandwidth used for the 1D PS in MHz
z_T = 5.82                                                 #Redshift of the target J352-15
nu_T = 209.002                                             #Observed 21-cm frequency at the target redshift in MHz
freq_range = np.arange(0,bandwidth/1.*1.00001,spec_res*1e-3)
freq_smooth = (nu_T + freq_range)*1e6
T_ind = int(freq_smooth.shape[0])                          #Number of frequency channels used for the 1D PS

bandwidth = (freq_smooth[-1]-freq_smooth[0])/1e6
print('Target redshift             = %.2f' % z_T)
print('Target frequency            = %.2f MHz' % nu_T)
print('Number of frequency channels = %d' % len(freq_smooth))
print('Frequency range             = %.2f-%.2f MHz' % (freq_smooth[0]/1e6,freq_smooth[-1]/1e6))
print('Bandwidth                   = %.2f MHz' % bandwidth)
n_kbins = int((len(freq_smooth)/2+1))                      #Number of Fourier modes


#Generate the signal 1D power spectra for each selected IGM model
for l in range(0,len(xHI_mean)):

  print('<xHI>=%.2f, log10(fX)=%.2f, %d/%d' % (xHI_mean[l],logfX[l],l+1,len(xHI_mean)))

  #Skip models for which the output 1D PS file already exists
  outfile = '1DPS_dimensionless/1DPS_signal/power_spectrum_signal_mask_21cmFAST_%.1f-%.1fMHz_z%.1f_WLDlvl%d_fflag%.2f_fX%.2f_xHI%.2f_%.1fkHz_%dLOS.dat' % (np.round(freq_smooth[0]/1e6),np.round(freq_smooth[-1]/1e6),z,WLDlvl,f_flag,logfX[l],xHI_mean[l],spec_res,n_los)
  if os.path.isfile(outfile):
    print('File already exists, skipping.')
    continue

  #Read the 21-cm optical depth for the selected IGM model
  datafile = str('%stau_long/tau_200Mpc_n%d_z%.3f_fX%.1f_xHI%.2f_dv%d.dat' %(path,Nlos,z_name,logfX[l],xHI_mean[l],dvH))
  
  if os.path.isfile(datafile):

   data = np.fromfile(str(datafile),dtype=np.float32)
   tau = np.reshape(data,(Nlos,Nbins))

   #Select the requested number of LOS and convert optical depth to transmitted flux
   tau = tau[:n_los,:]
   signal_ori = instrumental_features.transF(tau)
   freq_uni,signal_uni = instrumental_features.uni_freq(freq_ori,signal_ori)

   PS_signal = np.empty((n_los,n_kbins))                   #Signal 1D PS for each LOS

   done_perc = 0.1                                         #Progress-printing interval

   for j in range(n_los):

      #Smooth the signal to the observational spectral resolution
      freq_smooth0,signal_smooth = instrumental_features.smooth_fixedbox(freq_uni,signal_uni[j],spec_res)
      signal_smooth = signal_smooth[:T_ind]

      #Apply the observational channel mask and calculate the masked 1D PS
      signal_smooth[ind_flagflag] = np.nan
      k,PS_signal[j,:] = PS1D.get_P_masked(signal_smooth,mask,bandwidth)

      #Print progress every 10 per cent
      done_LOS = (j+1)/n_los
      if done_LOS>=done_perc:
         print('Done %.2f' % done_LOS)
         done_perc = done_perc+0.1
         
   #Save the signal 1D PS ensemble
   array = np.append(n_los,n_kbins)
   array = np.append(array,k)
   array = np.append(array,PS_signal)
   array.astype('float32').tofile(outfile,sep='')

  else:
    print("File not found.")
  
  #Print total execution time
  stop_clock = time.perf_counter()
  time_taken = (stop_clock-start_clock)
  print('It took %.3fs to run' % time_taken)
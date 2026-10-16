'''
Code to calibrate he uGMRT observational data once it is in the form of residual flux density spectrum

Includes: - Residual bandpass feature subtraction using wavelet decomposition (WLD)
          - Flagging of bad channels based on relative noise rms
'''
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal as scisi
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
import matplotlib.patches as patches
import openpyxl
from scipy.stats import norm, kstest, ks_2samp
from scipy.stats import anderson_ksamp
from scipy.stats import gaussian_kde

from scipy.signal import savgol_filter      #for residual bandpass feature mitigation using Savitzky–Golay filter
import pywt                                 #for residual bandpass feature mitigation using wavelet transform
from numpy.linalg import svd                #for RFI mitigation using SVD
from scipy.ndimage import binary_dilation   #for RFI mitigation using thresholding
import time

import constants
import instrumental_features
import PS1D

time_start = time.perf_counter()

#Properties of the target source
source = 'PSOJ352-15'
S147 = 110.6
alphaR = -0.89

#Input parameters
WLDlvl = int(sys.argv[1])   #Level of wavelet decomposition. 0 are the coarsest features.
f_flag = float(sys.argv[2]) #Threshold for flagging bad channels in terms of average noise rms
fsize = 24
np.random.seed(42)

#Read the target spectrum data
print('Target spectrum')
datafile = str('obs_data/spectra_common_band_all_night_no_filter.txt')
data = np.loadtxt(datafile)
F_signal = np.flip(data[:,1])[0:-1]
freq_signal = np.flip(data[:,0])[0:-1]

bandwidth = (freq_signal[-1]-freq_signal[0])
spec_res = (freq_signal[1]-freq_signal[0])*1e3
print('Bandwidth           = %.3fMHz' % bandwidth)
print('Spectral resolution = %.3fkHz' % spec_res)
print('Number of channels  = %d' % len(freq_signal))
print('Frequency range     = %.3fMHz to %.3fMHz' % (freq_signal[0],freq_signal[-1]))

#Compute rms in the target sightline
signal_rms = np.std(F_signal)
print('RMS signal = %.2f mJy/bm' % signal_rms)

#Read the off-source (noise-only) spectra data
print('Off-source spectra')
datafile = str('obs_data/common_freq.txt')
freq_noise = np.flip(np.loadtxt(datafile))
freq_noise = freq_noise[0:-1]
bandwidth = (freq_noise[-1]-freq_noise[0])
spec_res = (freq_noise[1]-freq_noise[0])*1e3
print('Bandwidth           = %.2fMHz' % bandwidth)
print('Spectral resolution = %.2fkHz' % spec_res)
print('Number of channels  = %d' % len(freq_noise))
print('Frequency range     = %.2fMHz to %.2fMHz' % (freq_noise[0],freq_noise[-1]))

datafile = str('obs_data/off_source_spectra_common_band_all_night_no_filter.txt')
F_noise = np.flip(np.loadtxt(datafile),axis=1)
F_noise = F_noise[:,0:-1]
Nlos_noise = F_noise.shape[0]

#Compute rms in the off-source sightlines
noise_rms = np.std(F_noise, axis=0)
mean_noise = np.std(F_noise)
print('RMS noise from off-source spectra = %.2f mJy/bm' % mean_noise)



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

  #Suppress only fine-scale noise ===
  #Keep the approximation + low-order details, suppress high-order details. You can tune how many detail levels to keep
  coeffs_denoised = []
  for i, c in enumerate(coeffs):
      if i == 0 or i <= detail_keep_level:
          coeffs_denoised.append(c)  # keep approximation + smooth details
      else:
          coeffs_denoised.append(np.zeros_like(c))  # suppress noise

  #Reconstruct the denoised flux
  bandpassfit = pywt.waverec(coeffs_denoised, wavelet)

  #Ensure same length (sometimes waverec returns slightly longer array)
  bandpassfit = bandpassfit[:len(signal)]

  return bandpassfit


#Perform the WLD
bandpassfit = WLD(F_signal,'db8',level=WLDlvl)  #Extracted broadband residual bandpass feature
F_signal_afterWLD = F_signal - bandpassfit      #and subtracted

print('After WLD --------------------------------------------------')
signal_rms = np.std(F_signal_afterWLD)
print('RMS signal = %.2f mJy/bm' % signal_rms)

F_noise_afterWLD = np.empty_like(F_noise)
bandpassfit_noise = np.empty_like(F_noise)
for i in range(Nlos_noise):
  bandpassfit_noise[i,:] = WLD(F_noise[i,:],'db8',level=WLDlvl) #Extracted broadband residual bandpass feature
  F_noise_afterWLD[i,:] = F_noise[i,:] - bandpassfit_noise[i,:] #and subtracted for each sightline individually

noise_rms = np.std(F_noise_afterWLD, axis=0)
mean_noise = np.std(F_noise_afterWLD[~np.isnan(F_noise_afterWLD)])
print('RMS noise from off-source spectra = %.2f mJy/bm' % mean_noise)


#Plot the residual bandpass feature fitting procedure
fig = plt.figure(figsize=(10.,5.))
gs = gridspec.GridSpec(1,1)

ax4 = plt.subplot(gs[0,0])  

ax4.set_xlim(freq_signal[0],freq_signal[-1])
ax4.set_ylim(-25,25)
ax4.set_xlabel(r'$\nu_{\rm obs}\,[\rm MHz]$', fontsize=fsize)
ax4.set_ylabel(r'Residual $S\,[\rm mJy\,beam^{-1}]$', fontsize=fsize)
ax4.xaxis.set_minor_locator(AutoMinorLocator())
ax4.yaxis.set_minor_locator(AutoMinorLocator())
ax4.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax4.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax4.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=2)

ax4.plot(freq_signal,F_signal,'-',color='black',label='Measured')                         #Residual flux density spectrum before the residual badnpass subtraction
ax4.plot(freq_signal,bandpassfit,'--',color='fuchsia',label='Bandpass fit')               #Residual badnpass fit from WLD
plt.legend(frameon=False,loc='upper left',ncol=2,fontsize=fsize-4)
ax4.plot([freq_signal[0],freq_signal[-1]],[0,0],':',color='darkorange',label='Continuum') #Residual continuum, i.e. zero line

#Upper horizontal axis in terms of redshift
ax4_top = ax4.twiny()
ax4_top.set_xlim(ax4.get_xlim())
z_ticks = np.arange(5.4, 5.8, 0.1)
freq_ticks = 1420.405751 / (z_ticks + 1)
freq_ticks = freq_ticks[(freq_ticks >= freq_signal[0]) & (freq_ticks <= freq_signal[-1])]
z_ticks = 1420.405751 / freq_ticks - 1
z_minor_ticks = np.arange(5.35, 5.8, 0.01)
freq_minor_ticks = 1420.405751 / (z_minor_ticks + 1)
freq_minor_ticks = freq_minor_ticks[(freq_minor_ticks >= freq_signal[0]) & (freq_minor_ticks <= freq_signal[-1])]
freq_minor_ticks = freq_minor_ticks[~np.isclose(freq_minor_ticks[:, None], freq_ticks, atol=1e-6).any(axis=1)]
ax4_top.set_xticks(freq_ticks)
ax4_top.set_xticks(freq_minor_ticks, minor=True)
ax4_top.set_xticklabels(['%.1f' % z for z in z_ticks])
ax4_top.set_xlabel(r'$z$', fontsize=fsize)
ax4_top.tick_params(axis='x', which='major', direction='in', length=20, width=2, labelsize=fsize)
ax4_top.tick_params(axis='x', which='minor', direction='in', length=10, width=2)

plt.tight_layout()
plt.subplots_adjust(hspace=.0)
plt.savefig('plots/Soltinsky_2026/spectrum_postprocess_beforeWLDlvl%d_%s_%d-%dMHz_%.1fkHz.png' % (WLDlvl,source,np.round(freq_signal[0]),np.round(freq_signal[-1]),spec_res))
plt.show()


#Plot off-source sightline spectra after the residual bandpass subtraction
fig = plt.figure(figsize=(10.,5.))
gs = gridspec.GridSpec(1,1)

ax4 = plt.subplot(gs[0,0])  

ax4.set_xlim(freq_noise[0],freq_noise[-1])
ax4.set_ylim(-25,30)
ax4.set_xlabel(r'$\nu_{\rm obs}\,[\rm MHz]$', fontsize=fsize)
ax4.set_ylabel(r'Residual $S\,[\rm mJy\,beam^{-1}]$', fontsize=fsize)
ax4.xaxis.set_minor_locator(AutoMinorLocator())
ax4.yaxis.set_minor_locator(AutoMinorLocator())
ax4.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax4.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax4.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=2)

#Plot all of the sighlines spectra
#To decrease the size for .pdf version one can plot every ith sightline
ith = 100
for i in range(int(Nlos_noise/ith)):
    ax4.plot(freq_noise, F_noise_afterWLD[i*ith], '-', color='gray', alpha=0.3, label='Individual spectra' if i == 0 else None)

#Plot the noise rms per frequency channel (averaged over all sightlines within the channel)
ax4.plot(freq_noise,noise_rms,'-',color='black',label=r'$\sigma_{\rm N}(\nu)$')
#Plot the mean noise rms over all channels
ax4.plot([freq_noise[0],freq_noise[-1]],[mean_noise,mean_noise],'--',color='fuchsia',label=r'$\langle\sigma_{\rm N}(\nu)\rangle=%.2f\,\rm mJy\,beam^{-1}$' % mean_noise)

plt.legend(frameon=False,loc=[0.05,0.8],ncol=3,fontsize=fsize-8,columnspacing=0.75,handletextpad=0.3,labelspacing=0.2,borderaxespad=0.2)
#Residual continuum, i.e. zero line
ax4.plot([freq_noise[0],freq_noise[-1]],[0,0],':',color='darkorange',label='Continuum')

#Plot a mirror image over the zero line
ax4.plot(freq_noise,-noise_rms,'-',color='black',label=r'$\sigma_{\rm N}(\nu)$')
ax4.plot([freq_noise[0],freq_noise[-1]],-np.array([mean_noise,mean_noise]),'--',color='fuchsia',label=r'$\sigma_{\rm N}=%.2f\,\rm mJy\,beam^{-1}$' % mean_noise)

#Upper horizontal axis in terms of redshift
ax4_top = ax4.twiny()
ax4_top.set_xlim(ax4.get_xlim())
z_ticks = np.arange(5.4, 5.8, 0.1)
freq_ticks = 1420.405751 / (z_ticks + 1)
freq_ticks = freq_ticks[(freq_ticks >= freq_noise[0]) & (freq_ticks <= freq_noise[-1])]
z_ticks = 1420.405751 / freq_ticks - 1
z_minor_ticks = np.arange(5.35, 5.8, 0.01)
freq_minor_ticks = 1420.405751 / (z_minor_ticks + 1)
freq_minor_ticks = freq_minor_ticks[(freq_minor_ticks >= freq_noise[0]) & (freq_minor_ticks <= freq_noise[-1])]
freq_minor_ticks = freq_minor_ticks[~np.isclose(freq_minor_ticks[:, None], freq_ticks, atol=1e-6).any(axis=1)]
ax4_top.set_xticks(freq_ticks)
ax4_top.set_xticks(freq_minor_ticks, minor=True)
ax4_top.set_xticklabels(['%.1f' % z for z in z_ticks])
ax4_top.set_xlabel(r'$z$', fontsize=fsize)
ax4_top.tick_params(axis='x', which='major', direction='in', length=20, width=2, labelsize=fsize)
ax4_top.tick_params(axis='x', which='minor', direction='in', length=10, width=2)

plt.tight_layout()
plt.subplots_adjust(hspace=.0)
plt.savefig('plots/Soltinsky_2026/spectrum_postprocess_offsource_WLDlvl%d_%s_%d-%dMHz_%.1fkHz_i%d.png' % (WLDlvl, source, np.round(freq_noise[0]), np.round(freq_noise[-1]), spec_res,ith))
plt.show()



#Plot target sightline spectrum after the residual bandpass subtraction
fig = plt.figure(figsize=(15,7.5))
gs = gridspec.GridSpec(1,1)

ax4 = plt.subplot(gs[0,0])  

ax4.set_xlim(freq_signal[0],freq_signal[-1])
ax4.set_ylim(-25,25)
ax4.set_xlabel(r'$\nu_{\rm obs}\ [\rm MHz]$', fontsize=fsize)
ax4.set_ylabel(r'Residual $S\ [\rm mJy\,beam^{-1}]$', fontsize=fsize)
ax4.xaxis.set_minor_locator(AutoMinorLocator())
ax4.yaxis.set_minor_locator(AutoMinorLocator())
ax4.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax4.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax4.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=2)

ax4.plot(freq_signal,F_signal_afterWLD,'-',color='black',label='Observed')                #target spectrum
ax4.plot([freq_signal[0],freq_signal[-1]],[0,0],':',color='darkorange',label='Continuum') #Continuum, i.e. zero line

#Upper horizontal axis in terms of redshift
ax4_top = ax4.twiny()
ax4_top.set_xlim(ax4.get_xlim())
z_ticks = np.arange(5.4, 5.8, 0.1)
freq_ticks = 1420.405751 / (z_ticks + 1)
freq_ticks = freq_ticks[(freq_ticks >= freq_signal[0]) & (freq_ticks <= freq_signal[-1])]
z_ticks = 1420.405751 / freq_ticks - 1
z_minor_ticks = np.arange(5.35, 5.8, 0.01)
freq_minor_ticks = 1420.405751 / (z_minor_ticks + 1)
freq_minor_ticks = freq_minor_ticks[(freq_minor_ticks >= freq_signal[0]) & (freq_minor_ticks <= freq_signal[-1])]
freq_minor_ticks = freq_minor_ticks[~np.isclose(freq_minor_ticks[:, None], freq_ticks, atol=1e-6).any(axis=1)]
ax4_top.set_xticks(freq_ticks)
ax4_top.set_xticks(freq_minor_ticks, minor=True)
ax4_top.set_xticklabels(['%.1f' % z for z in z_ticks])
ax4_top.set_xlabel(r'$z$', fontsize=fsize)
ax4_top.tick_params(axis='x', which='major', direction='in', length=20, width=2, labelsize=fsize)
ax4_top.tick_params(axis='x', which='minor', direction='in', length=10, width=2)

plt.tight_layout()
plt.subplots_adjust(hspace=.0)
plt.savefig('plots/Soltinsky_2026/spectrum_postprocess_WLDlvl%d_%s_%d-%dMHz_%.1fkHz.png' % (WLDlvl, source, np.round(freq_signal[0]), np.round(freq_signal[-1]), spec_res))
plt.show()



print('Flagging corrupted channels -------------------------------------')

#Flagging corrupted channels based on noise peaks going above threshold in terms of average noise rms
#Peaks defined over 20 frequency channels around channels with f_flag*mean_noise amplitude
ind_flag, peak_props = scisi.find_peaks(noise_rms, height=f_flag*mean_noise, distance=11)
freq_noise_flag = freq_noise[ind_flag]

ind_flagflag = np.empty(0,dtype=int)

#Print the positions of the flagged peaks and remember them
print('Flagged peaks = %d' % len(ind_flag))
for i in range(len(ind_flag)):
  print('  Peak %d: channel = %d, freq = %.3f MHz, noise_rms = %.3f mJy/bm' %
        (i+1, ind_flag[i], freq_noise[ind_flag[i]], noise_rms[ind_flag[i]]))

  ind_flagflag = np.append(ind_flagflag, np.arange(ind_flag[i]-10, ind_flag[i]+10))

#Take each channel only once and needs to be within the spectrum
ind_flagflag = np.unique(ind_flagflag)
ind_flagflag = ind_flagflag[(ind_flagflag >= 0) & (ind_flagflag < len(freq_noise))]

#Compute the effect of flagging on the noise rms
print('Spectrum length before flagging: %d' % len(F_signal_afterWLD))
F_signal_afterWLD_flagged = F_signal_afterWLD.copy()
F_signal_afterWLD_flagged[ind_flagflag] = np.nan
print('Spectrum length after flagging: %d' % len(F_signal_afterWLD_flagged))
signal_rms = np.nanstd(F_signal_afterWLD_flagged)
print('RMS signal = %.2f mJy/bm' % signal_rms)
F_noise_afterWLD_flagged = F_noise_afterWLD.copy()
for i in range(Nlos_noise):
    F_noise_afterWLD_flagged[i,ind_flagflag] = np.nan
noise_rms = np.nanstd(F_noise_afterWLD_flagged, axis=0)
mean_noise = np.std(F_noise_afterWLD_flagged[~np.isnan(F_noise_afterWLD_flagged)])
print('RMS noise from off-source spectra = %.2f mJy/bm' % mean_noise)



#Plot off-source sightline spectra indicating flagged channels. Besides this identical to the rpevious version of the plot.
fig = plt.figure(figsize=(10.,5.))
gs = gridspec.GridSpec(1,1)

ax4 = plt.subplot(gs[0,0])  

ax4.set_xlim(freq_noise[0],freq_noise[-1])
ax4.set_ylim(-25,30)
ax4.set_xlabel(r'$\nu_{\rm obs}\,[\rm MHz]$', fontsize=fsize)
ax4.set_ylabel(r'Residual $S\,[\rm mJy\,beam^{-1}]$', fontsize=fsize)
ax4.xaxis.set_minor_locator(AutoMinorLocator())
ax4.yaxis.set_minor_locator(AutoMinorLocator())
ax4.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax4.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax4.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=2)

for i in range(Nlos_noise):
    ax4.plot(freq_noise, F_noise_afterWLD[i], '-', color='gray', alpha=0.3, label='Individual spectra' if i == 0 else None)

ax4.plot(freq_noise,noise_rms,'-',color='black',label=r'$\sigma_{\rm N}(\nu)$')
ax4.plot([freq_noise[0],freq_noise[-1]],[mean_noise,mean_noise],'--',color='fuchsia',label=r'$\langle\sigma_{\rm N}(\nu)\rangle=%.2f\,\rm mJy\,beam^{-1}$' % mean_noise)

plt.legend(frameon=False,loc=[0.05,0.8],ncol=3,fontsize=fsize-8,columnspacing=0.75,handletextpad=0.3,labelspacing=0.2,borderaxespad=0.2)
ax4.plot([freq_noise[0],freq_noise[-1]],[0,0],':',color='darkorange',label='Continuum')

ax4.plot(freq_noise,-noise_rms,'-',color='black',label=r'$\sigma_{\rm N}(\nu)$')
ax4.plot([freq_noise[0],freq_noise[-1]],-np.array([mean_noise,mean_noise]),'--',color='fuchsia',label=r'$\langle\sigma_{\rm N}(\nu)\rangle=%.2f\,\rm mJy\,beam^{-1}$' % mean_noise)

#Indicate the flagged channels
for i in range(len(freq_noise_flag)):
  ax4.axvspan(freq_noise_flag[i]-10*spec_res/1e3, freq_noise_flag[i]+10*spec_res/1e3,
            color='royalblue', alpha=0.25, label='Flagged channels')

ax4_top = ax4.twiny()
ax4_top.set_xlim(ax4.get_xlim())
z_ticks = np.arange(5.4, 5.8, 0.1)
freq_ticks = 1420.405751 / (z_ticks + 1)
freq_ticks = freq_ticks[(freq_ticks >= freq_noise[0]) & (freq_ticks <= freq_noise[-1])]
z_ticks = 1420.405751 / freq_ticks - 1
z_minor_ticks = np.arange(5.35, 5.8, 0.01)
freq_minor_ticks = 1420.405751 / (z_minor_ticks + 1)
freq_minor_ticks = freq_minor_ticks[(freq_minor_ticks >= freq_noise[0]) & (freq_minor_ticks <= freq_noise[-1])]
freq_minor_ticks = freq_minor_ticks[~np.isclose(freq_minor_ticks[:, None], freq_ticks, atol=1e-6).any(axis=1)]
ax4_top.set_xticks(freq_ticks)
ax4_top.set_xticks(freq_minor_ticks, minor=True)
ax4_top.set_xticklabels(['%.1f' % z for z in z_ticks])
ax4_top.set_xlabel(r'$z$', fontsize=fsize)
ax4_top.tick_params(axis='x', which='major', direction='in', length=20, width=2, labelsize=fsize)
ax4_top.tick_params(axis='x', which='minor', direction='in', length=10, width=2)

plt.tight_layout()
plt.subplots_adjust(hspace=.0)
plt.savefig('plots/Soltinsky_2026/spectrum_postprocess_offsource_WLDlvl%d_fflag%.2f_%s_%d-%dMHz_%.1fkHz.png' % (WLDlvl, f_flag, source, np.round(freq_noise[0]), np.round(freq_noise[-1]), spec_res))
plt.show()



#Plot target sightline spectrum indicating flagged channels. Besides this identical to the rpevious version of the plot.
fig = plt.figure(figsize=(10.,5))
gs = gridspec.GridSpec(1,1)

ax4 = plt.subplot(gs[0,0])  

ax4.set_xlim(freq_signal[0],freq_signal[-1])
ax4.set_ylim(-25,25)
ax4.set_xlabel(r'$\nu_{\rm obs}\ [\rm MHz]$', fontsize=fsize)
ax4.set_ylabel(r'Residual $S\ [\rm mJy\,beam^{-1}]$', fontsize=fsize)
ax4.xaxis.set_minor_locator(AutoMinorLocator())
ax4.yaxis.set_minor_locator(AutoMinorLocator())
ax4.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax4.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax4.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=2)

ax4.plot(freq_signal,F_signal_afterWLD,'-',color='black',label='Observed')
ax4.plot([freq_signal[0],freq_signal[-1]],[0,0],':',color='darkorange',label='Continuum')

#Indicate the flagged channels
for i in range(len(freq_noise_flag)):
  ax4.axvspan(freq_noise_flag[i]-10*spec_res/1e3, freq_noise_flag[i]+10*spec_res/1e3,
            color='royalblue', alpha=0.25, label='Flagged channels')

ax4_top = ax4.twiny()
ax4_top.set_xlim(ax4.get_xlim())
z_ticks = np.arange(5.4, 5.8, 0.1)
freq_ticks = 1420.405751 / (z_ticks + 1)
freq_ticks = freq_ticks[(freq_ticks >= freq_signal[0]) & (freq_ticks <= freq_signal[-1])]
z_ticks = 1420.405751 / freq_ticks - 1
z_minor_ticks = np.arange(5.35, 5.8, 0.01)
freq_minor_ticks = 1420.405751 / (z_minor_ticks + 1)
freq_minor_ticks = freq_minor_ticks[(freq_minor_ticks >= freq_signal[0]) & (freq_minor_ticks <= freq_signal[-1])]
freq_minor_ticks = freq_minor_ticks[~np.isclose(freq_minor_ticks[:, None], freq_ticks, atol=1e-6).any(axis=1)]
ax4_top.set_xticks(freq_ticks)
ax4_top.set_xticks(freq_minor_ticks, minor=True)
ax4_top.set_xticklabels(['%.1f' % z for z in z_ticks])
ax4_top.set_xlabel(r'$z$', fontsize=fsize)
ax4_top.tick_params(axis='x', which='major', direction='in', length=20, width=2, labelsize=fsize)
ax4_top.tick_params(axis='x', which='minor', direction='in', length=10, width=2)

plt.tight_layout()
plt.subplots_adjust(hspace=.0)
plt.savefig('plots/Soltinsky_2026/spectrum_postprocess_flagged_WLDlvl%d_fflag%.2f_%s_%d-%dMHz_%.1fkHz.png' % (WLDlvl, f_flag, source, np.round(freq_signal[0]), np.round(freq_signal[-1]), spec_res))
plt.show()



#Plot target sightline spectrum with flagged channels removed. Besides this identical to the rpevious version of the plot.
fig = plt.figure(figsize=(10.,5))
gs = gridspec.GridSpec(1,1)

ax4 = plt.subplot(gs[0,0])  

ax4.set_xlim(freq_signal[0],freq_signal[-1])
ax4.set_ylim(-15,15)
ax4.set_xlabel(r'$\nu_{\rm obs}\ [\rm MHz]$', fontsize=fsize)
ax4.set_ylabel(r'Residual $S\ [\rm mJy\,beam^{-1}]$', fontsize=fsize)
ax4.xaxis.set_minor_locator(AutoMinorLocator())
ax4.yaxis.set_minor_locator(AutoMinorLocator())
ax4.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax4.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax4.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=2)

ax4.plot(freq_signal,F_signal_afterWLD_flagged,'-',color='black',label='Observed')
ax4.plot([freq_signal[0],freq_signal[-1]],[0,0],':',color='darkorange',label='Continuum')

ax4_top = ax4.twiny()
ax4_top.set_xlim(ax4.get_xlim())
z_ticks = np.arange(5.4, 5.8, 0.1)
freq_ticks = 1420.405751 / (z_ticks + 1)
freq_ticks = freq_ticks[(freq_ticks >= freq_signal[0]) & (freq_ticks <= freq_signal[-1])]
z_ticks = 1420.405751 / freq_ticks - 1
z_minor_ticks = np.arange(5.35, 5.8, 0.01)
freq_minor_ticks = 1420.405751 / (z_minor_ticks + 1)
freq_minor_ticks = freq_minor_ticks[(freq_minor_ticks >= freq_signal[0]) & (freq_minor_ticks <= freq_signal[-1])]
freq_minor_ticks = freq_minor_ticks[~np.isclose(freq_minor_ticks[:, None], freq_ticks, atol=1e-6).any(axis=1)]
ax4_top.set_xticks(freq_ticks)
ax4_top.set_xticks(freq_minor_ticks, minor=True)
ax4_top.set_xticklabels(['%.1f' % z for z in z_ticks])
ax4_top.set_xlabel(r'$z$', fontsize=fsize)
ax4_top.tick_params(axis='x', which='major', direction='in', length=20, width=2, labelsize=fsize)
ax4_top.tick_params(axis='x', which='minor', direction='in', length=10, width=2)

plt.tight_layout()
plt.subplots_adjust(hspace=.0)
plt.savefig('plots/Soltinsky_2026/spectrum_postprocess_WLDlvl%d_fflag%.2f_%s_%d-%dMHz_%.1fkHz.png' % (WLDlvl, f_flag, source, np.round(freq_signal[0]), np.round(freq_signal[-1]), spec_res))
plt.show()



#Test if signal statistically consistent with noise
print('Compare target to off-source spectra')
#Two-sample Kolmogorov-Smirnov test
ks_stat, ks_pvalue = ks_2samp(F_signal_afterWLD_flagged[~np.isnan(F_signal_afterWLD_flagged)], F_noise_afterWLD_flagged[~np.isnan(F_noise_afterWLD_flagged)])
print('KS test: statistic = %.4f, p-value = %.4f' % (ks_stat, ks_pvalue))

#k-sample Anderson-Darling test
res = anderson_ksamp([F_signal_afterWLD_flagged[~np.isnan(F_signal_afterWLD_flagged)], F_noise_afterWLD_flagged[~np.isnan(F_noise_afterWLD_flagged)]])
print('Anderson-Darling test: statistic = %.4f, p-value = %.4f' % (res.statistic, res.significance_level))
print('Anderson-Darling test: p-values = %.4f' % res.pvalue)

#Plot the target sightline residual flux density distribution
fig = plt.figure(figsize=(10., 5.))
gs = gridspec.GridSpec(1, 1)
ax_hist = plt.subplot(gs[0, 0])

#Plot it as a histogram
n, bins, patches_hist = ax_hist.hist(F_signal_afterWLD_flagged[~np.isnan(F_signal_afterWLD_flagged)], bins=100, density=True,
                                     color='black', alpha=0.6, label='Target source')

#Fit Gaussian distribution over the off-source sightlines.
#Force it to be centred at zero given that this is a residual flux density. Note that if this is left to vary freely, mu_fit does not change significantly.
mu_fit = 0.0
_, rms_fit = norm.fit(F_noise_afterWLD_flagged[~np.isnan(F_noise_afterWLD_flagged)], floc=0)
print('Best-fit Gaussian for off-source: mu = %.4f mJy/bm, sigma = %.4f mJy/bm' % (mu_fit, rms_fit))

#Overplot this off-source spectra distribution
x_fit = np.linspace(bins[0], bins[-1], 500)
pdf_fit = norm.pdf(x_fit, mu_fit, rms_fit)
kde = gaussian_kde(np.reshape(F_noise_afterWLD_flagged[~np.isnan(F_noise_afterWLD_flagged)], -1))
ax_hist.plot(x_fit, kde(x_fit), '-', color='fuchsia', linewidth=2,label=r'Off-source')

ax_hist.set_xlim(-13,13)
ax_hist.set_ylim(0,0.16)
ax_hist.set_xlabel(r'Residual $S\,[\rm mJy\,beam^{-1}]$', fontsize=fsize)
ax_hist.set_ylabel(r'PDF', fontsize=fsize)
ax_hist.xaxis.set_minor_locator(AutoMinorLocator())
ax_hist.yaxis.set_minor_locator(AutoMinorLocator())
ax_hist.tick_params(axis='x', which='major', direction='in', bottom=True, top=True,
                    left=True, right=True, length=20, width=2, labelsize=fsize)
ax_hist.tick_params(axis='y', which='major', direction='in', bottom=True, top=True,
                    left=True, right=True, length=20, width=2, labelsize=fsize)
ax_hist.tick_params(axis='both', which='minor', direction='in', bottom=True, top=True,
                    left=True, right=True, length=10, width=2)
ax_hist.legend(frameon=False, loc='upper left', fontsize=fsize - 4)

ax_hist.text(0.05, 0.7, r'KS test: $p=%.3f$' % ks_pvalue, transform=ax_hist.transAxes, fontsize=fsize - 4, verticalalignment='top')
ax_hist.text(0.05, 0.6, r'AD test: $p>%.2f$' % res.pvalue, transform=ax_hist.transAxes, fontsize=fsize - 4, verticalalignment='top')

plt.tight_layout()
plt.savefig('plots/Soltinsky_2026/signal_distribution_postprocess_target-offsource_WLDlvl%d_fflag%.2f_mu0_%s_%d-%dMHz_%.1fkHz.png' % (WLDlvl, f_flag, source, np.round(freq_signal[0]), np.round(freq_signal[-1]), spec_res))
plt.show()



print('Power spectrum ---------------------------------------------')
fsize=26
print('Target')
#Assuming single power-law radio continuum spectrum
S_v = S147*np.power(freq_signal/147.,alphaR)

#Sidelobe contamination cutoff following Thyagarajan 2020, ApJ, 899, 16
z_T = 5.82            #redshift of the target J352-15
#Cosmological parameters taken from Planck 2013 results (Planck Collaboration et al. 2014, A&A , 571, A16). Consistent with 21cmFAST simulations generated here.
omega_M = 0.308       #matter energy density
omega_Lambda = 0.692  #vacuum density
omega_k = 1.0 - omega_M - omega_Lambda
h = 0.678             #Hubble parameter
Ez = np.sqrt(omega_M*(1+z_T)**3+omega_k*(1+z_T)**2+omega_Lambda)
nu_T = (constants.c/constants.lambda_0)/(1+z_T)/1e6 #Frequency loction of the target in MHz
print("nu_T = %.3f MHz" % nu_T)

#uGMRT from GMRT Technical Manual 2024
theta_P = 120.*(185/nu_T)*np.pi/180/60               #radians
theta_S = 17.3*(185/nu_T)*np.pi/180/3600             #radians

print("theta_P = %.2erad = %.5fdeg" % (theta_P, theta_P*180/np.pi))
print("theta_S = %.2erad = %.5fdeg" % (theta_S, theta_S*180/np.pi))
k_SL_Mpc = 2*np.pi*100*Ez*1e5/(constants.c*(1+z_T))*np.sin(theta_P/2)/theta_S  #hcMpc^-1
MPCtoMHz = constants.c*(1+z_T)**2/(constants.c/constants.lambda_0)/(100*h*Ez*1e5)*h/(2*np.pi)*1e6
k_SL = k_SL_Mpc*MPCtoMHz     #MHz^-1
print("E(z) = %.3f" % Ez)
print("k_SL = %.3f hMpc^-1 = %.3f MHz^-1" % (k_SL_Mpc, k_SL))



print('Before WLD --------------')

print('Target')
mask = np.ones(len(F_signal), dtype=bool) #No mask applied
k_signal,PS_signal_all = PS1D.get_P_masked(1+F_signal/S_v,mask,bandwidth)
print('%d k bins from %.3fMHz^-1 to %.3fMHz^-1' % (len(k_signal),k_signal[1],k_signal[-1]))

#Prepare k bins
d_log_k_bins = 0.25
log_k_bins = np.arange(-0.5,2.75+d_log_k_bins/2.,d_log_k_bins)
k_bins = np.power(10.,log_k_bins)
k_bins_cent = np.power(10.,log_k_bins+d_log_k_bins/2.)[:-1]
print('%d k bin centres from %.3fMHz^-1 to %.3fMHz^-1' % (len(k_bins_cent),k_bins_cent[0],k_bins_cent[-1]))

#Bin the power spectrum in k
PS_signal = np.empty(len(k_bins_cent))
for j in range(len(k_bins_cent)):
  ind = np.where((k_signal>=k_bins[j]) & (k_signal<k_bins[j+1]))[0]
  #print('Bin %d: k = %.3f-%.3f MHz^-1, N_modes = %d' % (j+1,k_bins[j],k_bins[j+1],len(ind)))
  PS_signal[j] = np.mean(PS_signal_all[ind])

print('Off-source')
PS_noise_bin = np.empty((Nlos_noise,len(k_bins_cent)))

#Loop over each sightline for off-source spectra, i.e. noise
done_perc = 0.2
for j in range(Nlos_noise):

  k,PS_noise = PS1D.get_P_masked(1.+F_noise[j]/S_v,mask,bandwidth)

  for i in range(len(k_bins_cent)):
    ind = np.where((k>=k_bins[i]) & (k<k_bins[i+1]))[0]
    PS_noise_bin[j,i]  = np.mean(PS_noise[ind])

  #Track the progress
  done_LOS = (j+1)/Nlos_noise
  if done_LOS>=done_perc:
    print('Done %.2f' % done_LOS)
    done_perc = done_perc+0.2

#Plot the power spectra
fig = plt.figure(figsize=(10.,5))
gs = gridspec.GridSpec(1,1)

ax0 = plt.subplot(gs[0,0])

#For legend
ax0.plot([10000,20000],[1e-3,2e-3],'-',color='black',linewidth=2,label=r'Target source')
ax0.plot([10000,20000],[1e-3,2e-3],'-',color='fuchsia',label=r'Off-source')
ax0.legend(frameon=False,loc='lower right',fontsize=fsize-2,ncol=1)

for j in range(Nlos_noise):
  ax0.plot(k_bins_cent,PS_noise_bin[j,:],'-',color='fuchsia',alpha=0.1,label=r'Off-source' if j == 0 else None)
ax0.plot(k_bins_cent,PS_signal,'-',color='black',linewidth=2,label=r'Target source')

ax0.set_xlim(k_bins_cent[0],k_bins_cent[-1])
ax0.set_ylim(2e-10,5e-2)
ax0.set_xscale('log')
ax0.set_yscale('log')
ax0.set_xlabel(r'$\kappa_{\nu} \,\rm [MHz^{-1}]$', fontsize=fsize)
ax0.set_ylabel(r'$\kappa_{\nu}P_{21}$', fontsize=fsize)
ax0.tick_params(axis='x',which='major',direction='in',bottom=True,top=False,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax0.tick_params(axis='y',which='major',direction='in',bottom=True,top=False,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax0.tick_params(axis='both',which='minor',direction='in',bottom=True,top=False,left=True,right=True
		,length=10,width=2)

#Indicate sidelobe-induced mode-mixing cutoff
ax0.axvline(x=k_SL,color='gray',linestyle=':')
ax0.text(k_SL*0.75,1e-8,r'$\kappa_{\nu,\mathrm{SL}}$',rotation=90,fontsize=fsize-2)

#Upper horizontal axis in terms of cosmological k in h/cMpc
secax = ax0.secondary_xaxis('top',functions=(lambda tau: tau / MPCtoMHz,lambda k: k * MPCtoMHz))
secax.set_xscale('log')
secax.set_xlabel(r"$k_{\parallel}\ [h\,\mathrm{cMpc}^{-1}]$", fontsize=fsize)
secax.tick_params(axis='x',which='major',direction='in',length=20,width=2,labelsize=fsize)
secax.tick_params(axis='x',which='minor',direction='in',length=10,width=2)

plt.tight_layout()
plt.subplots_adjust(hspace=.0)
plt.savefig('plots/Soltinsky_2026/kP21_beforeWLD_%s_%d-%dMHz_%.1fkHz_small_kappa.png' % (source, np.round(freq_signal[0]), np.round(freq_signal[-1]), spec_res))
plt.show()



print('After WLD --------------')
print('Target')
mask = np.ones(len(F_signal_afterWLD), dtype=bool)  #No mask applied

k_signal,PS_signal_all = PS1D.get_P_masked(1+F_signal_afterWLD/S_v,mask,bandwidth)

#Bin the power spectrum in k
PS_signal = np.empty(len(k_bins_cent))
for j in range(len(k_bins_cent)):
  ind = np.where((k_signal>=k_bins[j]) & (k_signal<k_bins[j+1]))[0]
  #print('Bin %d: k = %.3f-%.3f MHz^-1, N_modes = %d' % (j+1,k_bins[j],k_bins[j+1],len(ind)))
  PS_signal[j] = np.mean(PS_signal_all[ind])

print('Off-source')
PS_noise_bin = np.empty((Nlos_noise,len(k_bins_cent)))

#Loop over each sightline for off-source spectra, i.e. noise
done_perc = 0.2
for j in range(Nlos_noise):

  k,PS_noise = PS1D.get_P_masked(1.+F_noise_afterWLD[j]/S_v,mask,bandwidth)

  for i in range(len(k_bins_cent)):
    ind = np.where((k>=k_bins[i]) & (k<k_bins[i+1]))[0]
    PS_noise_bin[j,i]  = np.mean(PS_noise[ind])

  #Track the progress
  done_LOS = (j+1)/Nlos_noise
  if done_LOS>=done_perc:
    print('Done %.2f' % done_LOS)
    done_perc = done_perc+0.2

#Plot the power spectra
fig = plt.figure(figsize=(10.,5))
gs = gridspec.GridSpec(1,1)

ax0 = plt.subplot(gs[0,0])

#For the legend
ax0.plot([10000,20000],[1e-3,2e-3],'-',color='black',linewidth=2,label=r'Target source')
ax0.plot([10000,20000],[1e-3,2e-3],'-',color='fuchsia',label=r'Off-source')
ax0.legend(frameon=False,loc='lower right',fontsize=fsize,ncol=1)

for j in range(Nlos_noise):
  ax0.plot(k_bins_cent,PS_noise_bin[j,:],'-',color='fuchsia',alpha=0.1,label=r'Off-source' if j == 0 else None)
ax0.plot(k_bins_cent,PS_signal,'-',color='black',linewidth=2,label=r'Target source')

ax0.set_xlim(k_bins_cent[0],k_bins_cent[-1])
ax0.set_ylim(2e-10,5e-2)
ax0.set_xscale('log')
ax0.set_yscale('log')
ax0.set_xlabel(r'$\kappa_{\nu} \,\rm [MHz^{-1}]$', fontsize=fsize)
ax0.set_ylabel(r'$\kappa_{\nu}P_{21}$', fontsize=fsize)
ax0.tick_params(axis='x',which='major',direction='in',bottom=True,top=False,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax0.tick_params(axis='y',which='major',direction='in',bottom=True,top=False,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax0.tick_params(axis='both',which='minor',direction='in',bottom=True,top=False,left=True,right=True
		,length=10,width=2)

#Indicate sidelobe-induced mode-mixing cutoff
ax0.axvline(x=k_SL,color='gray',linestyle=':')
ax0.text(k_SL*0.75,5e-4,r'$\kappa_{\nu,\mathrm{SL}}$',rotation=90,fontsize=fsize-2)

#Upper horizontal axis in terms of cosmological k in h/cMpc
secax = ax0.secondary_xaxis('top',functions=(lambda tau: tau / MPCtoMHz,lambda k: k * MPCtoMHz))
secax.set_xscale('log')
secax.set_xlabel(r"$k_{\parallel}\ [h\,\mathrm{cMpc}^{-1}]$", fontsize=fsize)
secax.tick_params(axis='x',which='major',direction='in',length=20,width=2,labelsize=fsize)
secax.tick_params(axis='x',which='minor',direction='in',length=10,width=2)

plt.tight_layout()
plt.subplots_adjust(hspace=.0)
plt.savefig('plots/Soltinsky_2026/kP21_postprocess_mask_WLDlvl%d_fflag%.2f_%s_%d-%dMHz_%.1fkHz_small_kappa.png' % (5.0, f_flag, source, np.round(freq_signal[0]), np.round(freq_signal[-1]), spec_res))
plt.show()



print('After flagging --------------')
print('Target')
mask = np.ones(len(F_signal_afterWLD_flagged), dtype=bool)
mask[ind_flagflag] = False  #Mask bad channels

k_signal,PS_signal_all = PS1D.get_P_masked(1+F_signal_afterWLD_flagged/S_v,mask,bandwidth)

#Bin the power spectrum in k
PS_signal = np.empty(len(k_bins_cent))
for j in range(len(k_bins_cent)):
  ind = np.where((k_signal>=k_bins[j]) & (k_signal<k_bins[j+1]))[0]
  #print('Bin %d: k = %.3f-%.3f MHz^-1, N_modes = %d' % (j+1,k_bins[j],k_bins[j+1],len(ind)))
  PS_signal[j] = np.mean(PS_signal_all[ind])

print('Off-source')
PS_noise_bin = np.empty((Nlos_noise,len(k_bins_cent)))

#Loop over each sightline for off-source spectra, i.e. noise
done_perc = 0.2
for j in range(Nlos_noise):

  k,PS_noise = PS1D.get_P_masked(1.+F_noise_afterWLD_flagged[j]/S_v,mask,bandwidth)

  for i in range(len(k_bins_cent)):
    ind = np.where((k>=k_bins[i]) & (k<k_bins[i+1]))[0]
    PS_noise_bin[j,i]  = np.mean(PS_noise[ind])

  #Track the progress
  done_LOS = (j+1)/Nlos_noise
  if done_LOS>=done_perc:
    print('Done %.2f' % done_LOS)
    done_perc = done_perc+0.2

#Plot the power spectra
fig = plt.figure(figsize=(10.,5))
gs = gridspec.GridSpec(1,1)

ax0 = plt.subplot(gs[0,0])

#For the legend
ax0.plot([10000,20000],[1e-3,2e-3],'-',color='black',linewidth=2,label=r'Target source')
ax0.plot([10000,20000],[1e-3,2e-3],'-',color='fuchsia',label=r'Off-source')
ax0.legend(frameon=False,loc='lower right',fontsize=fsize,ncol=1)

for j in range(Nlos_noise):
  ax0.plot(k_bins_cent,PS_noise_bin[j,:],'-',color='fuchsia',alpha=0.1,label=r'Off-source' if j == 0 else None)
ax0.plot(k_bins_cent,PS_signal,'-',color='black',linewidth=2,label=r'Target source')

ax0.set_xlim(k_bins_cent[0],k_bins_cent[-1])
ax0.set_ylim(2e-10,5e-2)
ax0.set_xscale('log')
ax0.set_yscale('log')
ax0.set_xlabel(r'$\kappa_{\nu} \,\rm [MHz^{-1}]$', fontsize=fsize)
ax0.set_ylabel(r'$\kappa_{\nu}P_{21}$', fontsize=fsize)
ax0.tick_params(axis='x',which='major',direction='in',bottom=True,top=False,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax0.tick_params(axis='y',which='major',direction='in',bottom=True,top=False,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax0.tick_params(axis='both',which='minor',direction='in',bottom=True,top=False,left=True,right=True
		,length=10,width=2)

#Indicate sidelobe-induced mode-mixing cutoff
ax0.axvline(x=k_SL,color='gray',linestyle=':')
ax0.text(k_SL*0.75,5e-4,r'$\kappa_{\nu,\mathrm{SL}}$',rotation=90,fontsize=fsize-2)

#Upper horizontal axis in terms of cosmological k in h/cMpc
secax = ax0.secondary_xaxis('top',functions=(lambda tau: tau / MPCtoMHz,lambda k: k * MPCtoMHz))
secax.set_xscale('log')
secax.set_xlabel(r"$k_{\parallel}\ [h\,\mathrm{cMpc}^{-1}]$", fontsize=fsize)
secax.tick_params(axis='x',which='major',direction='in',length=20,width=2,labelsize=fsize)
secax.tick_params(axis='x',which='minor',direction='in',length=10,width=2)

plt.tight_layout()
plt.subplots_adjust(hspace=.0)
plt.savefig('plots/Soltinsky_2026/kP21_postprocess_mask_WLDlvl%d_fflag%.2f_%s_%d-%dMHz_%.1fkHz_small_kappa.png' % (WLDlvl, f_flag, source, np.round(freq_signal[0]), np.round(freq_signal[-1]), spec_res))
plt.show()


#Save the postprocessed/calibrated flux density and power spectra
npy_table = {
  'frequency': freq_signal.astype(np.float32),
  'F_signal': F_signal_afterWLD_flagged.astype(np.float32),
  'F_noise': F_noise_afterWLD_flagged.astype(np.float32)
}
npy_fname = 'obs_data/postprocessed_mask_obs_Flux_%d-%dMHz_%s_%.1fkHz_WLDlvl%d_fflag%.2f.npy' % (np.round(freq_signal[0]), np.round(freq_signal[-1]), source, spec_res, WLDlvl, f_flag)
np.save(npy_fname, npy_table, allow_pickle=True)

npy_table = {
  'k_bin': k_bins_cent.astype(np.float32),
  'PS_signal': PS_signal.astype(np.float32),
  'PS_noise': PS_noise_bin.astype(np.float32)
}
npy_fname = 'obs_data/postprocessed_mask_obs_PS_%d-%dMHz_%s_%.1fkHz_WLDlvl%d_fflag%.2f.npy' % (np.round(freq_signal[0]), np.round(freq_signal[-1]), source, spec_res, WLDlvl, f_flag)
np.save(npy_fname, npy_table, allow_pickle=True)

time_end = time.perf_counter()
print('Took %.2f seconds of your life' % (time_end - time_start))
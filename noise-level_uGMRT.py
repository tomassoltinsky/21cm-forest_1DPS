import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
import pywt                                 #for residual bandpass feature mitigation using wavelet transform
import scipy.signal as scisi

#Calibration parameters used in our analysis in Šoltinský et al. 2026
f_flag = 1.5  #Threshold for flagging bad channels in terms of average noise rms
WLDlvl = 0    #Level of wavelet decomposition. 0 are the coarsest features.

tint = np.array([2.77,3.10,6.00,2.75,2.88])               #On-source integration time per observing night
exc_frac = np.array([0.4008,0.3568,0.2732,0.1482,0.5244]) #Fraction of data excissed due to systematics mitigation

tint_eff_single = tint*(1-exc_frac) #On-source effective integration time per observing night
print(tint_eff_single)
print('tint_total     = %.3fhr' % np.sum(tint))
print('tint_eff_total = %.3fhr' % np.sum(tint_eff_single))
exc_frac_total = 1-np.sum(tint_eff_single)/np.sum(tint)
print('exc_frac_total = %.4f' % exc_frac_total)

tint_eff = np.empty(len(tint_eff_single))
for i in range(len(tint_eff_single)):
    tint_eff[i] = np.sum(tint_eff_single[:i+1]) #Cumulative on-source effective integration time per observing night



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

noise_rms = np.std(F_noise, axis=0)
mean_noise = np.std(F_noise)
print('RMS noise from off-source spectra = %.2f mJy/bm' % mean_noise)


#Perform the same calibration steps as in obs_data_postprocess_bandpassfilter.py but for multiple datasets obtained by adding a single observations night at a time
def RMS(x, axis=None):
    return np.sqrt(np.nanmean(x**2, axis=axis))

print('WLD bandpass subtraction')
# Smoothing with wavelet transform

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

F_noise_afterWLD = np.empty_like(F_noise)
bandpassfit_noise = np.empty_like(F_noise)
for i in range(Nlos_noise):
  bandpassfit_noise[i,:] = WLD(F_noise[i,:],'db8',level=WLDlvl)
  F_noise_afterWLD[i,:] = F_noise[i,:] - bandpassfit_noise[i,:]

noise_rms = RMS(F_noise_afterWLD, axis=0)
mean_noise = RMS(F_noise_afterWLD[~np.isnan(F_noise_afterWLD)])
print('RMS noise from off-source spectra = %.2f mJy/bm' % mean_noise)
mean_noise_all = mean_noise

print('Flagging corrupted channels -------------------------------------')

#Flagging corrupted channels based on noise peaks
#Note that the same channels will be flagged for all datasets based on the all 5 nights data

ind_flag, peak_props = scisi.find_peaks(noise_rms, height=f_flag*mean_noise, distance=11)
freq_noise_flag = freq_noise[ind_flag]

ind_flagflag = np.empty(0,dtype=int)

print('Flagged peaks = %d' % len(ind_flag))
for i in range(len(ind_flag)):
  print('  Peak %d: channel = %d, freq = %.3f MHz, noise_rms = %.3f mJy/bm' %
        (i+1, ind_flag[i], freq_noise[ind_flag[i]], noise_rms[ind_flag[i]]))

  ind_flagflag = np.append(ind_flagflag, np.arange(ind_flag[i]-10, ind_flag[i]+10))

ind_flagflag = np.unique(ind_flagflag)
ind_flagflag = ind_flagflag[(ind_flagflag >= 0) & (ind_flagflag < len(freq_noise))]



sigma_N = np.empty(5)

for i in range(1,6):

  print('%d nights --------------------------------------------------' % i)

  datafile = str('obs_data/off_source_spectra_%dnight.npy' % i)
  F_noise = np.load(datafile)
  F_noise = F_noise[:,0:-1]
  Nlos_noise = F_noise.shape[0]

  noise_rms = np.nanstd(F_noise, axis=0)
  mean_noise = np.std(F_noise[~np.isnan(F_noise)])
  print('RMS noise from off-source spectra = %.2f mJy/bm' % mean_noise)

  print('WLD')
  F_noise_afterWLD = np.empty_like(F_noise)
  bandpassfit_noise = np.empty_like(F_noise)
  for j in range(Nlos_noise):
    bandpassfit_noise[j,:] = WLD(F_noise[j,:],'db8',level=WLDlvl)
    F_noise_afterWLD[j,:] = F_noise[j,:] - bandpassfit_noise[j,:]

  noise_rms = np.nanstd(F_noise_afterWLD, axis=0)
  mean_noise = np.std(F_noise_afterWLD[~np.isnan(F_noise_afterWLD)])
  print('RMS noise from off-source spectra = %.2f mJy/bm' % mean_noise)
  
  print('Flagging')
  for j in range(Nlos_noise):
    F_noise_afterWLD[j,ind_flagflag] = np.nan
  noise_rms = np.nanstd(F_noise_afterWLD, axis=0)
  mean_noise = np.std(F_noise_afterWLD[~np.isnan(F_noise_afterWLD)])
  print('RMS noise from off-source spectra = %.2f mJy/bm' % mean_noise)
  
  sigma_N[i-1] = mean_noise


sigma_N[-1] = mean_noise_all

tint_eff_fit = np.linspace(1.,12.5,100)

#Fit the sigma_N vs t_eff relationship as a single power-law
log_tint_eff = np.log(tint_eff)
log_sigma_N = np.log(sigma_N)
coeffs, cov = np.polyfit(log_tint_eff,log_sigma_N,1,cov=True)
coeffs_std = np.sqrt(np.diag(cov))
print(cov)
print(coeffs_std)
poly = np.poly1d(coeffs)
print('Fitted power-law:  sigma_N = %.3f * (tint_eff)^%.3f' % (np.exp(coeffs[1]),coeffs[0]))
sigma_N_fit = np.exp(coeffs[1])*np.power(tint_eff_fit,coeffs[0])

#Fit the data with radiometer scaling of (tint_eff)^-0.5
coeffs_fixed = np.polyfit(log_tint_eff, log_sigma_N + 0.5 * log_tint_eff, 0)
poly_fixed = np.poly1d(coeffs_fixed)
print('Fitted radiometer: sigma_N = %.3f * (tint_eff)^-0.5' % (np.exp(coeffs_fixed[0])))
sigma_N_fit_radiometer = np.exp(coeffs_fixed[0])*np.power(tint_eff_fit,-0.5)  #Theoretical radiometer scaling fit

#Instead one can compute the theoretical amplitude using the GMRT exposure time calculator - https://www.gmrt.ncra.tifr.res.in/gtac/etc/etc/etc.html
sigma_N_RM = 2.14
t_eff_RM = 5.5
sigma_N_fit_radiometer = sigma_N_RM*np.power(tint_eff_fit/t_eff_RM,-0.5)

#Plot the measured sigma_N and fitted empirical scaling with t_eff
fsize = 20
fig = plt.figure(figsize=(10.,5.))
gs = gridspec.GridSpec(1,1)

ax0 = plt.subplot(gs[0,0])  

ax0.set_ylim(0.,10.5)
ax0.set_xlim(0,13)

ax0.scatter(tint_eff,sigma_N,color='royalblue',label='Measured')  #Measured sigma_N datapoints
ax0.plot(tint_eff_fit,sigma_N_fit_radiometer,'--',color='fuchsia',label=r'$\sigma_N \propto t_{\mathrm{eff}}^{-0.5}$ (radiometer)')
ax0.plot(tint_eff_fit,sigma_N_fit,'-',color='darkorange',label=r'$\sigma_N \propto t_{\mathrm{eff}}^{%.2f}$ (empirical)' % (coeffs[0]))
plt.legend(frameon=False,loc='upper right',fontsize=fsize)

ax0.set_xlabel(r'$t_{\rm eff}\,[\rm hr]$', fontsize=fsize)
ax0.set_ylabel(r'$\sigma_{\rm N}\,[\rm mJy\,beam^{-1}\,per\,channel]$', fontsize=fsize)
ax0.xaxis.set_minor_locator(AutoMinorLocator())
ax0.yaxis.set_minor_locator(AutoMinorLocator())
ax0.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax0.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax0.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=2)

plt.tight_layout()
plt.subplots_adjust(left=0.11, hspace=0, wspace=0)
plt.savefig('plots/Soltinsky_2026/noise-level_vs_tinteff_ETC.png')
plt.show()


#Compare to theoretical radiometer equation expectation
ratio_obs_vs_RM = sigma_N/(sigma_N_RM*np.power(tint_eff/t_eff_RM,-0.5))
for i in range(len(ratio_obs_vs_RM)):
    print('t_eff: %.2f hr, Ratio obs vs radiometer: %.2f' % (tint_eff[i], ratio_obs_vs_RM[i]))


#Compute forecasted sensitivity based on our empirical scaling
RFI_frac = 0.4  #Assuming round value based on our analysis above
tint_archival = np.sum(tint_eff_single)
print('Archival effective integration time (after RFI excision) = %.2f hr' % tint_archival)
tint = np.array([100])
tint_forecasts = tint_archival + tint*(1.-RFI_frac)
print('Forecasted effective integration times (after RFI excision) in hr = ', tint_forecasts)
sigma_N_forecasts = np.exp(coeffs[1])*np.power(tint_forecasts,coeffs[0])
for i in range(len(tint)):
    print('Forecasted noise level for t_int=%.1fhr, t_eff=%.2fhr, t_eff+t_arch=%.2fhr is %.3f mJy beam^{-1} per channel' % (tint[i],tint[i]*(1.-RFI_frac),tint_forecasts[i],sigma_N_forecasts[i]))
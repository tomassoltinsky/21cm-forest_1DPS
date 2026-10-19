"""
Creating the 21-cm forest 1D power spectrum plot comparing the post-processed
uGMRT archival measurement of PSO J352-15 with forward-modelled 21cmFAST signal
and synthetic instrumental-noise power spectra.

The simulated signal and noise power spectra include the same spectral-channel
mask as the uGMRT archival observations. Two frequency-independent instrumental
noise rms values are considered to illustrate the measured archival sensitivity
and a lower-noise observational scenario.

Version 06.06.2026
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
from scipy import interpolate

import constants
import PS1D


#Input parameters
z_name = float(sys.argv[1])                 #Redshift of the 21cmFAST LOS data
dvH = float(sys.argv[2])                    #Velocity resolution of the LOS data in m/s
spec_res = float(sys.argv[3])               #Spectral resolution in kHz
fX_name = float(sys.argv[4])                #log10 of X-ray pre-heating efficiency
xHI_mean = float(sys.argv[5])               #Mean neutral hydrogen fraction
WLDlvl = int(sys.argv[6])                   #Level of wavelet decomposition used for residual bandpass subtraction
f_flag = float(sys.argv[7])                 #Threshold for flagging bad channels in units of mean off-source noise rms

#Simulation and observational parameters
n_los_signal = 20                           #Number of simulated signal LOS
n_los_noise = 1000                          #Number of synthetic noise realizations
rms = [3.62,1.65]                           #Frequency-independent instrumental noise rms values in mJy/beam
telescope = ['uGMRT','uGMRT']               #Telescope corresponding to each noise realization ensemble

S_min_QSO = 110.6                           #Intrinsic QSO flux density at 147 MHz in mJy
alpha_R = -0.89                             #Radio spectral index of the QSO
target = 'PSO J352−15'                      #Target source


#Read the forward-modelled signal 1D PS ensemble
datafile = str('1DPS_dimensionless/1DPS_signal/power_spectrum_signal_mask_21cmFAST_209.0-222.0MHz_z%.1f_WLDlvl%d_fflag%.2f_fX%.2f_xHI%.2f_%.1fkHz_%dLOS.dat' % (z_name,WLDlvl,f_flag,fX_name,xHI_mean,spec_res,n_los_signal))
data = np.fromfile(str(datafile),dtype=np.float32)
Nlos = int(data[0])                          #Number of signal LOS stored in the file
n_kbins = int(data[1])                      #Number of unbinned k modes
k = data[2:2+n_kbins]                       #Frequency wavenumbers in MHz^-1
PS_signal = data[2+n_kbins+0*n_kbins*Nlos:2+n_kbins+1*n_kbins*Nlos]
PS_signal = np.reshape(PS_signal,(Nlos,n_kbins))[:n_los_signal,:]


#Prepare logarithmic k bins
d_log_k_bins = 0.25                         #Logarithmic k-bin width
log_k_bins = np.arange(-0.5,2.75+d_log_k_bins/2.,d_log_k_bins)
k_bins = np.power(10.,log_k_bins)           #Edges of logarithmic k bins
k_bins_cent = np.power(10.,log_k_bins+d_log_k_bins/2.)[:-1] #Centers of logarithmic k bins

print('Number of logarithmic k bins = %d' % len(k_bins_cent))
print('k range = %.3f-%.3f MHz^-1' % (k_bins[0],k_bins[-1]))

PS_signal_bin = np.empty((n_los_signal,len(k_bins_cent))) #Binned signal PS for individual LOS

print('%d modes from %.3f MHz^-1 to %.3f MHz^-1' % (len(k),k[1],k[-1]))

#Bin the signal 1D PS for each LOS
for i in range(n_los_signal):
  for j in range(len(k_bins_cent)):
    ind = np.where((k>=k_bins[j]) & (k<k_bins[j+1]))[0]
    PS_signal_bin[i,j] = np.mean(PS_signal[i,ind])

#Calculate the mean and LOS scatter of the signal 1D PS
PS_signal_med = np.mean(PS_signal_bin,axis=0)              #Mean signal 1D PS
PS_signal_16 = np.percentile(PS_signal_bin,16,axis=0)      #16th percentile
PS_signal_84 = np.percentile(PS_signal_bin,84,axis=0)      #84th percentile


#Calculate the sidelobe-induced mode-mixing cutoff following Thyagarajan 2020, ApJ, 899, 16
z_T = 5.82                                                 #Redshift of the target J352-15
omega_M = 0.308                                            #Matter energy density
omega_Lambda = 0.692                                       #Vacuum energy density
omega_k = 1.0 - omega_M - omega_Lambda                     #Curvature energy density
h = 0.678                                                  #Reduced Hubble parameter
Ez = np.sqrt(omega_M*(1+z_T)**3+omega_k*(1+z_T)**2+omega_Lambda)
nu_T = (constants.c/constants.lambda_0)/(1+z_T)/1e6        #Observed 21-cm frequency at target redshift in MHz

print('Target redshift  = %.2f' % z_T)
print('Target frequency = %.2f MHz' % nu_T)
print('E(z)             = %.3f' % Ez)


#MWA
print('MWA')
theta_P = 1.44/4.4                                         #Primary beam angular scale in radians
theta_S = constants.c/(nu_T*1e6)/3e5                       #Synthesized beam using Rayleigh approximation for 3 km baseline
k_SL_Mpc = 2*np.pi*100*Ez*1e5/(constants.c*(1+z_T))*np.sin(theta_P/2)/theta_S  #Sidelobe cutoff in h cMpc^-1
MPCtoMHz = constants.c*(1+z_T)**2/(constants.c/constants.lambda_0)/(100*h*Ez*1e5)*h/(2*np.pi)*1e6
k_SL = k_SL_Mpc*MPCtoMHz                                   #Sidelobe cutoff in MHz^-1
print('Primary beam      = %.2f deg' % (theta_P*180/np.pi))
print('Synthesized beam  = %.2f arcsec' % (theta_S*180/np.pi*3600))
print('Sidelobe cutoff   = %.3f h/cMpc = %.3f MHz^-1' % (k_SL_Mpc,k_SL))


#LOFAR from van Haarlem et al. 2013 Table B1 and B2 selecting NL remote and Bmax=100km
print('LOFAR')
w_low  = (nu_T-200)/(210-200)
w_high = (210-nu_T)/(210-200)
theta_P = (w_low*2.13+w_high*2.03)*np.pi/180               #Primary beam angular scale in radians
theta_S = (w_low*2.48+w_high*2.36)*np.pi/180/3600          #Synthesized beam angular scale in radians
k_SL_Mpc = 2*np.pi*100*Ez*1e5/(constants.c*(1+z_T))*np.sin(theta_P/2)/theta_S
MPCtoMHz = constants.c*(1+z_T)**2/(constants.c/constants.lambda_0)/(100*h*Ez*1e5)*h/(2*np.pi)*1e6
k_SL = k_SL_Mpc*MPCtoMHz
print('Primary beam      = %.2f deg' % (theta_P*180/np.pi))
print('Synthesized beam  = %.2f arcsec' % (theta_S*180/np.pi*3600))
print('Sidelobe cutoff   = %.3f h/cMpc = %.3f MHz^-1' % (k_SL_Mpc,k_SL))


#uGMRT from GMRT Technical Manual 2024
print('uGMRT')
theta_P = 120.*(185/nu_T)*np.pi/180/60                     #Primary beam angular scale in radians
theta_S = 17.3*(185/nu_T)*np.pi/180/3600                   #Synthesized beam angular scale in radians
k_SL_Mpc = 2*np.pi*100*Ez*1e5/(constants.c*(1+z_T))*np.sin(theta_P/2)/theta_S
MPCtoMHz = constants.c*(1+z_T)**2/(constants.c/constants.lambda_0)/(100*h*Ez*1e5)*h/(2*np.pi)*1e6
k_SL = k_SL_Mpc*MPCtoMHz
print('Primary beam      = %.2f deg' % (theta_P*180/np.pi))
print('Synthesized beam  = %.2f arcsec' % (theta_S*180/np.pi*3600))
print('Sidelobe cutoff   = %.3f h/cMpc = %.3f MHz^-1' % (k_SL_Mpc,k_SL))


#Read the measured target residual flux density spectrum
print('Power spectrum - Measured signal')
datafile = str('obs_data/spectra_common_band_all_night_no_filter.txt')
data = np.loadtxt(datafile)
signal_smooth = np.flip(data[:,1])
freq_smooth = np.flip(data[:,0])
signal_smooth = signal_smooth[0:-1]
freq_smooth = freq_smooth[0:-1]

bandwidth = (freq_smooth[-1]-freq_smooth[0])               #Observational bandwidth in MHz
spec_res = (freq_smooth[1]-freq_smooth[0])*1e3             #Spectral resolution in kHz
print('Bandwidth           = %.2f MHz' % bandwidth)
print('Spectral resolution = %.2f kHz' % spec_res)
print('Number of channels  = %d' % len(freq_smooth))
print('Frequency range     = %.2f-%.2f MHz' % (freq_smooth[0],freq_smooth[-1]))

#Calculate the QSO continuum across the observational band
S_v = S_min_QSO*np.power(freq_smooth/147.,alpha_R)

#Calculate and bin the measured target 1D PS before post-processing
k_signal_obs,PS_signal_all_obs = PS1D.get_P(1+signal_smooth/S_v,bandwidth)

PS_signal_obs = np.empty(len(k_bins_cent))

for j in range(len(k_bins_cent)):
  ind = np.where((k_signal_obs>=k_bins[j]) & (k_signal_obs<k_bins[j+1]))[0]
  PS_signal_obs[j] = np.mean(PS_signal_all_obs[ind])


#Read the off-source spectra
print('Off-source spectra')

datafile = str('obs_data/common_freq.txt')
freq_noise = np.flip(np.loadtxt(datafile))
freq_noise = freq_noise[0:-1]
bandwidth = (freq_noise[-1]-freq_noise[0])
spec_res = (freq_noise[1]-freq_noise[0])*1e3
print('Bandwidth           = %.2f MHz' % bandwidth)
print('Spectral resolution = %.2f kHz' % spec_res)
print('Number of channels  = %d' % len(freq_noise))
print('Frequency range     = %.2f-%.2f MHz' % (freq_noise[0],freq_noise[-1]))

datafile = str('obs_data/off_source_spectra_common_band_all_night_no_filter.txt')
F_noise = np.flip(np.loadtxt(datafile),axis=1)
F_noise = F_noise[:,0:-1]
Nlos_noise = F_noise.shape[0]                              #Number of off-source spectra
noise_rms = np.std(F_noise, axis=0)                        #Noise rms in individual frequency channels
mean_noise = np.mean(noise_rms)                            #Mean noise rms across the band
print('RMS noise from off-source spectra = %.2f mJy/beam' % mean_noise)


#Calculate the observed off-source noise 1D PS
print('Power spectrum - Noise')

S_v = S_min_QSO*np.power(freq_noise/147.,alpha_R)          #QSO continuum across the observational band

PS_noise_obs_bin = np.empty((Nlos_noise,len(k_bins_cent))) #Binned PS for individual off-source spectra
PS_noise_obs_med = np.empty(len(k_bins_cent))

for j in range(Nlos_noise):

  k,PS_noise_obs = PS1D.get_P(1.+F_noise[j]/S_v,bandwidth)

  for i in range(len(k_bins_cent)):
    ind = np.where((k>=k_bins[i]) & (k<k_bins[i+1]))[0]
    PS_noise_obs_bin[j,i] = np.mean(PS_noise_obs[ind])

PS_noise_obs_med = np.mean(PS_noise_obs_bin,axis=0)         #Mean observed off-source noise 1D PS
print('Observed mean noise PS:')
print(PS_noise_obs_med)


#Read and bin the forward-modelled synthetic noise 1D PS ensembles
PS_noise_bin = np.empty((len(rms),Nlos_noise,len(k_bins_cent)))
PS_noise_med = np.empty((len(rms),len(k_bins_cent)))

for i in range(len(rms)):

  datafile = str('1DPS_dimensionless/1DPS_noise/power_spectrum_noise_mask_21cmFAST_209.0-222.0MHz_z%.1f_WLDlvl%d_fflag%.2f_%s_%.1fkHz_Smin%.1fmJy_alphaR%.2f_rms%.4fmJy_%dLOS.dat' % (z_name,WLDlvl,f_flag,telescope[i],spec_res,S_min_QSO,alpha_R,rms[i],n_los_noise))
  data = np.fromfile(str(datafile),dtype=np.float32)
  n_kbins = int(data[1])                                  #Number of unbinned k modes
  k = data[2:2+n_kbins]                                   #Frequency wavenumbers in MHz^-1
  PS_noise = np.reshape(data[2+n_kbins+0*n_kbins*Nlos_noise:2+n_kbins+1*n_kbins*Nlos_noise],(Nlos_noise,n_kbins))

  #Bin each synthetic noise realization
  for l in range(Nlos_noise):

    for j in range(len(k_bins_cent)):
      ind = np.where((k>=k_bins[j]) & (k<k_bins[j+1]))[0]
      PS_noise_bin[i,l,j] = np.mean(PS_noise[l,ind])

  PS_noise_med[i,:] = np.mean(PS_noise_bin[i],axis=0)      #Mean synthetic noise 1D PS

  print('For RMS=%.2f mJy/beam: PS range = %.3e-%.3e' %
        (rms[i],np.nanmin(PS_noise_med[i,:]),np.nanmax(PS_noise_med[i,:])))


#Read the post-processed measured target and off-source 1D power spectra
datafile = 'obs_data/postprocessed_mask_obs_PS_%d-%dMHz_%s_%.1fkHz_WLDlvl%d_fflag%.2f.npy' % (209,222,'PSOJ352-15',spec_res,0,1.5)
data = np.load(datafile, allow_pickle=True).item()
k_signal_obs = data['k_bin']                              #Binned frequency wavenumbers
PS_signal_obs = data['PS_signal']                         #Post-processed measured target 1D PS
PS_noise_bin_obs = data['PS_noise']                       #Post-processed off-source 1D PS ensemble


#Plot the observed and forward-modelled 1D power spectra
fsize = 16
fig = plt.figure(figsize=(10.,5.))
gs = gridspec.GridSpec(1,1)

k_min = 5e-1
k_max = 6e2

ax0 = plt.subplot(gs[0,0])

#Observed target 1D PS
ax0.plot(k_signal_obs,PS_signal_obs,'-',color='black',label=r'Signal (observed)')

#Print representative signal and noise PS values
print('Observed signal PS = ',PS_signal_obs[0],PS_signal_obs[3],PS_signal_obs[-1])
print('Simulated noise PS = ',PS_noise_med[0][0],PS_noise_med[0][3],PS_noise_med[0][-1])
print('Simulated signal PS = ',PS_signal_med[0],PS_signal_med[3],PS_signal_med[-1])


#Plot simulated signal and noise 1D PS
colours = ['fuchsia','royalblue','forestgreen']

ax0.plot([-1,-2],[-1,-2],'-',color='darkorange',label=r'Signal only')
#ax0.plot([-1,-2],[-1,-2],'--',color='fuchsia',label=r'Noise only')

for j in range(len(rms)):
  ax0.plot(k_bins_cent,PS_noise_med[j],'--',color=colours[j],
           label=r'Noise only (simulated, $\sigma_{\mathrm{N}}=%.2f\mathrm{mJy\,beam^{-1}}$)' % (rms[j]))

ax0.legend(frameon=False,loc='lower right',fontsize=fsize,ncol=1)

#ax0.text(k_bins_cent[5],1.6*PS_noise_med[0][5],r'$\sigma_{\mathrm{N}}=%.1f\mathrm{mJy\,beam^{-1}}$' % (rms[0]),rotation=12.5,fontsize=fsize)
#ax0.text(k_bins_cent[5],0.3*PS_noise_med[1][5],r'$\sigma_{\mathrm{N}}=%.1f\mathrm{mJy\,beam^{-1}}$' % (rms[1]),rotation=12.5,fontsize=fsize)

ax0.plot(k_bins_cent,PS_signal_med,'-',color='darkorange',label=r'Signal only (simulated)')
ax0.fill_between(k_bins_cent,PS_signal_16,PS_signal_84,alpha=0.25,color='darkorange')

#Indicate sidelobe-induced mode-mixing cutoff
ax0.axvline(x=k_SL,color='gray',linestyle=':')
ax0.text(k_SL*0.8,7e-5,r'$\kappa_{\nu,\mathrm{SL}}$',rotation=90,fontsize=fsize)

ax0.set_xlim(k_bins_cent[0],k_bins_cent[-1])
ax0.set_ylim(1e-7,1e-2)
ax0.set_xscale('log')
ax0.set_yscale('log')
ax0.set_yticks(np.power(10.,np.arange(-6,-1,2)))
ax0.set_xlabel(r'$\kappa_{\nu} \,\rm [MHz^{-1}]$', fontsize=fsize)
ax0.set_ylabel(r'$\kappa_{\nu} P_{21}$', fontsize=fsize)
ax0.tick_params(axis='x',which='major',direction='in',bottom=True,top=False,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax0.tick_params(axis='y',which='major',direction='in',bottom=True,top=False,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax0.tick_params(axis='both',which='minor',direction='in',bottom=True,top=False,left=True,right=True
		,length=10,width=2)

#Upper horizontal axis in terms of cosmological wavenumber
secax = ax0.secondary_xaxis('top',
                            functions=(lambda tau: tau / MPCtoMHz,
                                      lambda k:   k * MPCtoMHz))
secax.set_xscale('log')
secax.set_xlabel(r"$k_{\parallel}\ [h\,\mathrm{cMpc}^{-1}]$", fontsize=fsize)
secax.tick_params(axis='x',which='major',direction='in',length=20,width=2,labelsize=fsize)
secax.tick_params(axis='x',which='minor',direction='in',length=10,width=2)

plt.tight_layout()
plt.subplots_adjust(hspace=.0)
plt.savefig('plots/Soltinsky_2026/kP21_uGMRT-AD_209-222MHz_z%.1f_WLDlvl%d_fflag%.2f_fX%.2f_xHI%.2f_%dkHz_withobs_kappa.png' % (z_name,WLDlvl,f_flag,fX_name,xHI_mean,spec_res))
plt.show()
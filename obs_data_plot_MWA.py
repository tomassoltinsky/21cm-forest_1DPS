'''
Code to plot calibrated/measured power spectrum of the signal and noise from the MWA observations.

Version 24.6.2026
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
from scipy.stats import gaussian_kde

from scipy.signal import savgol_filter      #for residual bandpass feature mitigation using Savitzky–Golay filter
import pywt                                 #for residual bandpass feature mitigation using wavelet transform
from numpy.linalg import svd                #for RFI mitigation using SVD
from scipy.ndimage import binary_dilation   #for RFI mitigation using thresholding
import time

import constants
import instrumental_features

time_start = time.perf_counter()

#Properties of the target source
source = 'PSOJ352-15'
S147 = 110.6
alphaR = -0.89
spec_res = 40 #Spectral resolution of the data in kHz
fsize = 24


#Read the MWA data
datafile = str('obs_data/MWA_Full_k_Pmeas_Pnoise.txt')
data = np.loadtxt(datafile)
k_signal = data[:,0][0:-1]    #k bins in MHz^-1
P_measured = data[:,1][0:-1]  #Measured signal power spectrum
P_noiseth = data[:,2][0:-1]   #Thermal noise power spectrum

#Prepare k bins
d_log_k_bins = 0.25
log_k_bins = np.arange(-0.75,2.25+d_log_k_bins/2.,d_log_k_bins)
k_bins = np.power(10.,log_k_bins)
k_bins_cent = np.power(10.,log_k_bins+d_log_k_bins/2.)[:-1]

#Bin the power spectra in the same way as the uGMRT data
PS_measured = np.empty(len(k_bins_cent))
PS_noiseth = np.empty(len(k_bins_cent))

for j in range(len(k_bins_cent)):
  ind = np.where((k_signal>=k_bins[j]) & (k_signal<k_bins[j+1]))[0]
  #print('Bin %d: k = %.3f-%.3f MHz^-1, N_modes = %d' % (j+1,k_bins[j],k_bins[j+1],len(ind)))
  PS_measured[j] = np.mean(P_measured[ind])
  PS_noiseth[j] = np.mean(P_noiseth[ind])

print('Bin centers:')
print(k_bins_cent)
print('Signal power:')
print(PS_measured)
print('Noise power:')
print(PS_noiseth)


#Plot the power sepctra
fig = plt.figure(figsize=(10.,5))
gs = gridspec.GridSpec(1,1)

ax0 = plt.subplot(gs[0,0])

ax0.plot([10000,20000],[1e-3,2e-3],'-',color='black',linewidth=2,label=r'Signal')
ax0.plot([10000,20000],[1e-3,2e-3],'-',color='fuchsia',label=r'Noise')
ax0.legend(frameon=False,loc='lower right',fontsize=fsize-2,ncol=2)

ax0.plot(k_bins_cent,PS_measured,'-',color='black',linewidth=2,label=r'Signal')
ax0.plot(k_bins_cent,PS_noiseth,'-',color='fuchsia',linewidth=2,label=r'Noise')

ax0.set_xlim(k_bins_cent[0],k_bins_cent[-1])
ax0.set_ylim(1e-6,1e0)
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
print("E(z) = %.3f" % Ez)

#MWA details
theta_P = 1.44/4.4
theta_S = constants.c/(nu_T*1e6)/3e5    #Using Rayleigh approximation assuming 3km baseline
k_SL_Mpc = 2*np.pi*100*Ez*1e5/(constants.c*(1+z_T))*np.sin(theta_P/2)/theta_S  #hcMpc^-1
MPCtoMHz = constants.c*(1+z_T)**2/(constants.c/constants.lambda_0)/(100*h*Ez*1e5)*h/(2*np.pi)*1e6
k_SL = k_SL_Mpc*MPCtoMHz     #MHz^-1
print("theta_P = %.2erad = %.2fdeg" % (theta_P, theta_P*180/np.pi))
print("theta_S = %.2erad = %.2farcsec" % (theta_S, theta_S*180/np.pi*3600))
print("k_SL = %.3f hMpc^-1 = %.3f MHz^-1" % (k_SL_Mpc, k_SL))

#Indicate sidelobe-induced mode-mixing cutoff
ax0.axvline(x=k_SL,color='gray',linestyle=':')
ax0.text(k_SL*0.75,2e-6,r'$\kappa_{\nu,\mathrm{SL}}$',rotation=90,fontsize=fsize-2)


#Upper horizontal axis in terms of cosmological k in h/cMpc
secax = ax0.secondary_xaxis('top',
                            functions=(lambda tau: tau / MPCtoMHz,
                                      lambda k:   k * MPCtoMHz))
secax.set_xscale('log')
secax.set_xlabel(r"$k_{\parallel}\ [h\,\mathrm{cMpc}^{-1}]$", fontsize=fsize)
secax.tick_params(axis='x',which='major',direction='in',length=20,width=2,labelsize=fsize)
secax.tick_params(axis='x',which='minor',direction='in',length=10,width=2)

plt.tight_layout()
plt.subplots_adjust(hspace=.0)
plt.savefig('plots/Soltinsky_2026/kP21_postprocess_MWA_%s_%d-%dMHz_%.1fkHz_small_kappa.png' % (source, 201, 232, spec_res))
plt.show()
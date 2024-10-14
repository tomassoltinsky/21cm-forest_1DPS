"""
Creating the power spectrum plot.

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

fsize = 24

#Input parameters
z_name = float(sys.argv[1])   #redshift
dvH = float(sys.argv[2])      #rebinning width in km/s
spec_res = float(sys.argv[3]) #spectral resolution of telescope (i.e. frequency channel width) in kHz
fX_name = float(sys.argv[4])  #log10(f_X)
xHI_mean = float(sys.argv[5]) #mean x_HI
n_los = 1000

#Observational setup parameters
telescope = ['uGMRT','SKA1-low']  #telescope
t_int = [500,50]                  #integration time in hours
S_min_QSO = [64.2,64.2]           #intrinsic flux of QSO at 147Hz in mJy
alpha_R = [-0.44,-0.44]           #radio spectral index of QSO


#Load 21-cm forest 1D PS data
datafile = str('1DPS_dimensionless/1DPS_signal/power_spectrum_signal_21cmFAST_200Mpc_z%.1f_fX%.2f_xHI%.2f_%dkHz_%dLOS.dat' % (z_name,fX_name,xHI_mean,spec_res,n_los))
data = np.fromfile(str(datafile),dtype=np.float32)
Nlos = int(data[0])
n_kbins = int(data[1])
k = data[2:2+n_kbins]
PS_signal = data[2+n_kbins+0*n_kbins*Nlos:2+n_kbins+1*n_kbins*Nlos]
PS_signal = np.reshape(PS_signal,(Nlos,n_kbins))[:n_los,:]

#Generate k bins
d_log_k_bins = 0.25
log_k_bins = np.arange(-1.0-d_log_k_bins/2.,3.+d_log_k_bins/2.,d_log_k_bins)
log_k_bins = log_k_bins[1:]
k_bins = np.power(10.,log_k_bins)
k_bins_cent = np.power(10.,log_k_bins+d_log_k_bins/2.)[:-1]
print(k_bins_cent)
print('%d bins from %.3fMHz^-1 to %.3fMHz^-1' % (len(k),k[1],k[-1]))

#Bin the PS data
PS_signal_bin = np.empty((n_los,len(k_bins_cent)))

for j in range(len(k_bins_cent)):
  ind = np.where((k>=k_bins[j]) & (k<k_bins[j+1]))[0]
  print('Number of elements in %.3fMHz^-1<k<%.3fMHz^-1: %d' % (k_bins[j],k_bins[j+1],len(ind)))

for i in range(n_los):
  for j in range(len(k_bins_cent)):
    ind = np.where((k>=k_bins[j]) & (k<k_bins[j+1]))[0]
    PS_signal_bin[i,j] = np.mean(PS_signal[i,ind])

#Take median and 68% scatter for each k bin
PS_signal_med = np.mean(PS_signal_bin,axis=0)
PS_signal_16 = np.percentile(PS_signal_bin,16,axis=0)
PS_signal_84 = np.percentile(PS_signal_bin,84,axis=0)


#Do the same for the noise for all observational setups
Nlos_noise = 1000
PS_noise_bin = np.empty((len(t_int),Nlos_noise,len(k_bins_cent)))
PS_noise_med = np.empty((len(t_int),len(k_bins_cent)))
PS_noise_16 = np.empty((len(t_int),len(k_bins_cent)))
PS_noise_84 = np.empty((len(t_int),len(k_bins_cent)))

for i in range(len(t_int)):

  datafile = str('1DPS_dimensionless/1DPS_noise/power_spectrum_noise_21cmFAST_200Mpc_z%.1f_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f_%dLOS.dat' % (z_name,telescope[i],spec_res,t_int[i],S_min_QSO[i],alpha_R[i],Nlos))
  data = np.fromfile(str(datafile),dtype=np.float32)
  n_kbins = int(data[1])
  k = data[2:2+n_kbins]
  PS_noise = np.reshape(data[2+n_kbins+0*n_kbins*Nlos_noise:2+n_kbins+1*n_kbins*Nlos_noise],(Nlos_noise,n_kbins))

  for l in range(Nlos_noise):

    for j in range(len(k_bins_cent)):
      ind = np.where((k>=k_bins[j]) & (k<k_bins[j+1]))[0]
      PS_noise_bin[i,l,j]  = np.mean(PS_noise[l,ind])

  PS_noise_med[i,:] = np.mean(PS_noise_bin[i],axis=0)
  PS_noise_16[i,:]  = np.percentile(PS_noise_bin[i],16,axis=0)
  PS_noise_84[i,:]  = np.percentile(PS_noise_bin[i],84,axis=0)


#Plot the 1D PS
fig = plt.figure(figsize=(10.,5.))
gs = gridspec.GridSpec(1,1)

ax0 = plt.subplot(gs[0,0])

ax0.plot([100,200],[1,2],'-',color='darkorange',label=r'Signal')
ax0.plot([100,200],[1,2],'--',color='fuchsia',label=r'Noise')
ax0.legend(frameon=False,loc='upper left',fontsize=fsize-3,ncol=2)

ax0.plot(k_bins_cent,PS_signal_med,'-',color='darkorange',label=r'Signal')
ax0.fill_between(k_bins_cent,PS_signal_16,PS_signal_84,alpha=0.25,color='darkorange')
for j in range(0,len(S_min_QSO)):
  ax0.plot(k_bins_cent,PS_noise_med[j],'--',color='fuchsia',label=r'Signal')
  ax0.text(k_bins_cent[3],0.575*PS_noise_med[j,3],r'$\mathrm{%s,}\, t_{\mathrm{int}}=%d\mathrm{hr}$' % (telescope[j],t_int[j]),rotation=21,fontsize=fsize-3)

ax0.set_ylim(6e-9,5e-5)
ax0.set_xscale('log')
ax0.set_yscale('log')
ax0.set_xlabel(r'$k \,\rm [MHz^{-1}]$', fontsize=fsize)
ax0.set_ylabel(r'$kP_{21}$', fontsize=fsize)
ax0.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax0.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax0.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=2)

plt.tight_layout()
plt.subplots_adjust(hspace=.0)
plt.savefig('plots/kP21_21cmFAST_multinoise_200Mpc_z%.1f_fX%.2f_xHI%.2f_%dkHz.png' % (z_name,fX_name,xHI_mean,spec_res))
plt.show()

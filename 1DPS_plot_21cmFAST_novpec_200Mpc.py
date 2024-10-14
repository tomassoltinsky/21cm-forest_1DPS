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
import openpyxl
import time

import instrumental_features
import PS1D

fsize = 24

#Input parameters
z_name = float(sys.argv[1])   #redshift
dvH = float(sys.argv[2])      #rebinning width in km/s
logfX = float(sys.argv[3])    #log10(f_X)
xHI_mean = float(sys.argv[4]) #<x_HI>
path = '../../datasets/21cmFAST_los/'
n_los = 1000
Nlos = 1000

#Generate k bins
d_log_k_bins = 0.25
log_k_bins = np.arange(-1.0-d_log_k_bins/2.,10.+d_log_k_bins/2.,d_log_k_bins)
k_bins = np.power(10.,log_k_bins)
k_bins_cent = np.power(10.,log_k_bins+d_log_k_bins/2.)[:-1]
PS_signal_bin = np.empty((n_los,len(k_bins_cent)))

#Load data for x-axis and turn to frequency and redshift axis
datafile = str('%stau_long/los_200Mpc_n%d_z%.3f_dv%d.dat' %(path,Nlos,z_name,dvH))
data  = np.fromfile(str(datafile),dtype=np.float32)
z     = data[0]	#redshift
Nbins = int(data[2])					#Number of pixels/cells/bins in one line-of-sight
Nlos = int(data[3])						#Number of lines-of-sight
x_initial = 4
vel_axis = data[(x_initial+Nbins):(x_initial+2*Nbins)]#Hubble velocity along LoS in km/s
freq_ori = instrumental_features.freq_obs(z,vel_axis*1e5)
bandwidth = (freq_ori[-1]-freq_ori[0])/1e6
print('Bandwidth = %.2fMHz' % bandwidth)

#Load optical depth data
datafile = str('%stau_long/tau_200Mpc_n%d_z%.3f_fX%.1f_xHI%.2f_dv%d.dat' %(path,Nlos,z_name,logfX,xHI_mean,dvH))
data = np.fromfile(str(datafile),dtype=np.float32)
tau = np.reshape(data,(Nlos,-1))

tau = tau[:n_los,:]
signal_ori = instrumental_features.transF(tau)
freq_uni,signal_uni = instrumental_features.uni_freq(freq_ori,signal_ori)
freq_uni = freq_uni[:-1]
signal_uni = signal_uni[:,:-1]
n_kbins = int((len(freq_uni)/2+1))

PS_signal = np.empty((n_los,n_kbins))

#Compute the 1D power spectrum
for j in range(n_los):
  k,PS_signal[j,:] = PS1D.get_P(signal_uni[j],bandwidth)

#Bin the PS data
for i in range(n_los):
  for j in range(len(k_bins_cent)):
    ind = np.where((k>=k_bins[j]) & (k<k_bins[j+1]))[0]
    PS_signal_bin[i,j] = np.mean(PS_signal[i,ind])

#Take median and 68% scatter for each k bin
PS_signal_med = np.median(PS_signal_bin,axis=0)
PS_signal_16 = np.percentile(PS_signal_bin,16,axis=0)
PS_signal_84 = np.percentile(PS_signal_bin,84,axis=0)

print('%d bins from %.3fMHz^-1 to %.3fMHz^-1' % (len(k),k[1],k[-1]))



#Do the same for the case of no peculiar velocity
datafile = str('%stau_long/tau_200Mpc_n%d_z%.3f_fX%.1f_xHI%.2f_dv%d_novpec.dat' %(path,Nlos,z_name,logfX,xHI_mean,dvH))
data = np.fromfile(str(datafile),dtype=np.float32)
tau = np.reshape(data,(Nlos,-1))

tau = tau[:n_los,:]
signal_ori = instrumental_features.transF(tau)
freq_uni,signal_uni = instrumental_features.uni_freq(freq_ori,signal_ori)
freq_uni = freq_uni[:-1]
signal_uni = signal_uni[:,:-1]

PS_signal = np.empty((n_los,n_kbins))

for j in range(n_los):
  k,PS_signal[j,:] = PS1D.get_P(signal_uni[j],bandwidth)

PS_signal_bin = np.empty((n_los,len(k_bins_cent)))

for i in range(n_los):
  for j in range(len(k_bins_cent)):
    ind = np.where((k>=k_bins[j]) & (k<k_bins[j+1]))[0]
    PS_signal_bin[i,j] = np.mean(PS_signal[i,ind])

PS_signal_novpec_med = np.median(PS_signal_bin,axis=0)
PS_signal_novpec_16 = np.percentile(PS_signal_bin,16,axis=0)
PS_signal_novpec_84 = np.percentile(PS_signal_bin,84,axis=0)


#Plot the 1D power spectrum
fig = plt.figure(figsize=(10.,5.))
gs = gridspec.GridSpec(1,1)

ax0 = plt.subplot(gs[0,0])

ax0.plot([1,2],[1,2],'-',color='darkorange',label=r'With $v_{\rm pec}$')
ax0.plot([1,2],[1,2],'-',color='royalblue',label=r'Without $v_{\rm pec}$')
ax0.legend(frameon=False,loc='lower left',fontsize=fsize-3,ncol=1)

ax0.plot(k_bins_cent,PS_signal_med,'-',color='darkorange',label=r'Signal')
ax0.fill_between(k_bins_cent,PS_signal_16,PS_signal_84,alpha=0.25,color='darkorange')

ax0.plot(k_bins_cent,PS_signal_novpec_med,'-',color='royalblue',label=r'Signal')
ax0.fill_between(k_bins_cent,PS_signal_novpec_16,PS_signal_novpec_84,alpha=0.25,color='royalblue')

spec_res = 8
k_max = np.pi/(spec_res/1e3)
print(k_max)
ax0.plot([k_max,k_max],[1e-30,1e0],'--',c='fuchsia')
ax0.text(k_max*1.1,1e-19,r'$\Delta\nu=%d\,\rm kHz$ limit' % spec_res,rotation='vertical',fontsize=fsize)

ax0.set_ylim(1e-9,1e-5)
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
plt.subplots_adjust(bottom=.2)
plt.savefig('plots/power_spectrum_21cmFAST_dimles_novpec_200Mpc_z%.1f_fX%.2f_xHI%.2f.png' % (z_name,logfX,xHI_mean))
plt.show()
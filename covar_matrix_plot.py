"""
Parameter inference code based on Bayesian methods.

Uses 2D interpolator for 1D PS from 21-cm forest.

Likelihood calculated from the covariance matrix.

Version 6.5.2024

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
from scipy import interpolate
from scipy.optimize import minimize
import emcee
import corner
from numpy import random
random.seed(0)
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
import numpy.linalg
import instrumental_features

#Input parameters
z_name = float(sys.argv[1])         #redshift
dvH = float(sys.argv[2])            #used rebinning for line profile convolution in m/s
spec_res = float(sys.argv[3])       #spectral resolution of the telescope in kHz
xHI_mean_mock = float(sys.argv[4])  #mock HI fraction
logfX_mock = float(sys.argv[5])     #mock logfX
path_LOS = '../../datasets/21cmFAST_los/los/'
telescope = str(sys.argv[6])
S147 = float(sys.argv[7])           #intrinsic flux density of background source at 147MHz in mJy
alphaR = float(sys.argv[8])         #radio spectrum power-law index of background source
tint = float(sys.argv[9])           #intergration time for the observation in h
Nobs = int(sys.argv[10])            #number of observed sources or LOS

Nlos = 1000
n_los = 1000
Nsamples = 10000

min_logfX = -4.
max_logfX = 1.
min_xHI = 0.01
max_xHI = 0.6

fsize = 20

#Prepare k bins
d_log_k_bins = 0.25
log_k_bins = np.arange(-1.0-d_log_k_bins/2.,3.+d_log_k_bins/2.,d_log_k_bins)
log_k_bins = log_k_bins[2:-1]

k_bins = np.power(10.,log_k_bins)
k_bins_cent = np.power(10.,log_k_bins+d_log_k_bins/2.)[:-1]
print(k_bins_cent)


#Read the signal only data for which we want to estimate parameters
datafile = str('1DPS_dimensionless/1DPS_signal/power_spectrum_signal_21cmFAST_200Mpc_z%.1f_fX%.2f_xHI%.2f_%dkHz_%dLOS.dat' % (z_name,logfX_mock,xHI_mean_mock,spec_res,Nlos))
data = np.fromfile(str(datafile),dtype=np.float32)
Nlos = int(data[0])
n_kbins = int(data[1])
k = data[2:2+n_kbins]
PS_sv = data[2+n_kbins+0*n_kbins*Nlos:2+n_kbins+1*n_kbins*Nlos]
PS_sv = np.reshape(PS_sv,(Nlos,n_kbins))[:n_los,:]

#Bin the PS data
PS_sv_bin = np.empty((n_los,len(k_bins_cent)))

for i in range(n_los):
  for l in range(len(k_bins_cent)):
    ind = np.where((k>=k_bins[l]) & (k<k_bins[l+1]))[0]
    PS_sv_bin[i,l] = np.mean(PS_sv[i,ind])

#Get mean for each k bin assuming observation of multiple LOS
PS_sv_ens = instrumental_features.multi_obs(PS_sv_bin,Nobs,Nsamples)
PS_sv_mean = np.mean(PS_sv_ens,axis=0)

random.seed(0)

LOS = random.randint(0,Nlos-1,size=Nobs)
signal_multiobs = PS_sv_bin[LOS][:]
signal_multiobs_mean_sv = np.mean(signal_multiobs,axis=0)

#Compute covariance matrix for sample variance only
Mcovar_sv = np.cov(np.transpose(PS_sv_ens))
print('SV')
print(Mcovar_sv)
print(np.amin(np.abs(Mcovar_sv)),np.amax(np.abs(Mcovar_sv)))


#Read data for noise only
datafile = str('1DPS_dimensionless/1DPS_noise/power_spectrum_noise_21cmFAST_200Mpc_z%.1f_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f_%dLOS.dat' % (z_name,telescope,spec_res,tint,S147,alphaR,Nlos))
data = np.fromfile(str(datafile),dtype=np.float32)
n_kbins = int(data[1])
k = data[2:2+n_kbins]
PS_noise = data[2+n_kbins+0*n_kbins*Nlos:2+n_kbins+1*n_kbins*Nlos]
PS_noise = np.reshape(PS_noise,(Nlos,n_kbins))

#Bin the PS data
PS_noise_bin = np.empty((Nlos,len(k_bins_cent)))

for i in range(Nlos):
  for l in range(len(k_bins_cent)):
    ind = np.where((k>=k_bins[l]) & (k<k_bins[l+1]))[0]
    PS_noise_bin[i,l] = np.mean(PS_noise[i,ind])

#Get ensemble of observations of multiple LOS
#sig_PS_noise = np.std(PS_noise_bin,axis=0)
PS_noise_ens = instrumental_features.multi_obs(PS_noise_bin,Nobs,Nsamples)

random.seed(0)

LOS = random.randint(0,Nlos-1,size=Nobs)
signal_multiobs = PS_noise_bin[LOS][:]
signal_multiobs_mean_noise= np.mean(signal_multiobs,axis=0)

#Compute covariance matrix for noise only
Mcovar_noise = np.cov(np.transpose(PS_noise_ens))
print('Noise')
print(Mcovar_noise)
print(np.amin(np.abs(Mcovar_noise)),np.amax(np.abs(Mcovar_noise)))


#Read the mock (signal+noise) data for which we want to estimate parameters
datafile = str('1DPS_dimensionless/1DPS_signalandnoise/power_spectrum_signal_21cmFAST_200Mpc_z%.1f_fX%.2f_xHI%.2f_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f_%dLOS.dat' % (z_name,logfX_mock,xHI_mean_mock,telescope,spec_res,tint,S147,alphaR,Nlos))
data = np.fromfile(str(datafile),dtype=np.float32)
Nlos = int(data[0])
n_kbins = int(data[1])
k = data[2:2+n_kbins]
PS_nsv = data[2+n_kbins+0*n_kbins*Nlos:2+n_kbins+1*n_kbins*Nlos]
PS_nsv = np.reshape(PS_nsv,(Nlos,n_kbins))[:n_los,:]

#Bin the PS data
PS_nsv_bin = np.empty((n_los,len(k_bins_cent)))

for i in range(n_los):
  for l in range(len(k_bins_cent)):
    ind = np.where((k>=k_bins[l]) & (k<k_bins[l+1]))[0]
    PS_nsv_bin[i,l] = np.mean(PS_nsv[i,ind])

#Get mean for each k bin assuming observation of multiple LOS
PS_nsv_ens = instrumental_features.multi_obs(PS_nsv_bin,Nobs,Nsamples)
print('PS_nsv_ens.shape=', PS_nsv_ens.shape)
PS_nsv_mean = np.mean(PS_nsv_ens,axis=0)

random.seed(0)

LOS = random.randint(0,Nlos-1,size=Nobs)
signal_multiobs = PS_nsv_bin[LOS][:]
signal_multiobs_mean_mock = np.mean(signal_multiobs,axis=0)

Mcovar_mock = np.cov(np.transpose(PS_nsv_ens))
print('Mock')
print(Mcovar_mock)
print(np.amin(np.abs(Mcovar_mock)),np.amax(np.abs(Mcovar_mock)))

#Plotting the covariance and correlation matrices
'''
fig = plt.figure(figsize=(7.5,7.85))
gs = gridspec.GridSpec(1,3,width_ratios=[7.5,0.1,0.25])

ax0= plt.subplot(gs[0,0])
im=ax0.imshow(Mcorr,origin='lower',interpolation='none',cmap=plt.cm.inferno
          ,extent=[log_k_bins[0],log_k_bins[-1],log_k_bins[0],log_k_bins[-1]]
          ,aspect='auto',vmin=0,vmax=1)

ax0.set_xlabel(r'$\mathrm{log}_{10}(k/\rm MHz^{-1})$',fontsize=fsize)
ax0.set_ylabel(r'$\mathrm{log}_{10}(k/\rm MHz^{-1})$',fontsize=fsize)
ax0.xaxis.set_minor_locator(AutoMinorLocator())
ax0.yaxis.set_minor_locator(AutoMinorLocator())
ax0.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
     ,length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='y',which='minor',direction='in',bottom=True,top=True,left=True,right=True
     ,length=5,width=1)
ax0.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
     ,length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='x',which='minor',direction='in',bottom=True,top=True,left=True,right=True
     ,length=5,width=1)

axc= plt.subplot(gs[0,2])
cbar=fig.colorbar(im,pad=0.02,cax=axc)
#cbar.set_label(r'$\tau_{21}$',size=fsize)
cbar.ax.tick_params(labelsize=fsize)
fig.gca()

#ax0.set_title(r"$\langle x_{\rm HI}\rangle=%.2f,\, \log_{10}(f_{\mathrm{X}})=%.1f,\, %s,\, t_{\rm int}=%d\,\rm hr,\, S_{\rm 147}=%.1f\,\mathrm{mJy},\, \alpha_{\rm R}=%.2f$" % (xHI_mean_mock,logfX_mock,telescope,tint,S147,alphaR),fontsize=fsize)
ax0.set_title(r"$\langle x_{\rm HI}\rangle=%.2f,\, \log_{10}(f_{\mathrm{X}})=%.1f,\, %s,\, t_{\rm int}=%d\,\mathrm{hr},\, N_{\rm obs}=%d$" % (xHI_mean[j],logfX[j],telescope,tint,Nobs),fontsize=fsize)

plt.tight_layout()
plt.subplots_adjust(wspace=.0)
#plt.savefig('covariance_matrix/correlation_dimless_matrix_z%.1f_fX%.2f_xHI%.2f_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f_median%dLOS.png' % (z_name,logfX[j],xHI_mean[j],telescope,spec_res,tint,S147,alphaR,Nobs))
#plt.show()
plt.close()
'''


fig = plt.figure(figsize=(5.95,5.))
gs = gridspec.GridSpec(1,3,width_ratios=[5.,0.1,0.25])
#gs = fig.add_subplot(111, aspect='equal')

ax0= plt.subplot(gs[0,0])
im=ax0.imshow(Mcovar_noise,origin='lower',interpolation='none',cmap=plt.cm.bwr
          ,extent=[log_k_bins[0],log_k_bins[-1],log_k_bins[0],log_k_bins[-1]]
          #,aspect='auto',norm=LogNorm(vmin=3e-18,vmax=6e-13))
          ,aspect='auto',norm=SymLogNorm(linthresh=1e-19,vmin=-6e-13,vmax=6e-13))

ax0.set_xlabel(r'$\mathrm{log}_{10}(k/\rm MHz^{-1})$',fontsize=fsize)
ax0.set_ylabel(r'$\mathrm{log}_{10}(k/\rm MHz^{-1})$',fontsize=fsize)
ax0.set_xticks(np.arange(0.,2.5,1.))
ax0.set_yticks(np.arange(0.,2.5,1.))
ax0.xaxis.set_minor_locator(AutoMinorLocator())
ax0.yaxis.set_minor_locator(AutoMinorLocator())
ax0.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
     ,length=20,width=2,labelsize=fsize)
ax0.tick_params(axis='y',which='minor',direction='in',bottom=True,top=True,left=True,right=True
     ,length=10,width=2)
ax0.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
     ,length=20,width=2,labelsize=fsize)
ax0.tick_params(axis='x',which='minor',direction='in',bottom=True,top=True,left=True,right=True
     ,length=10,width=2)

axc= plt.subplot(gs[0,2])
cbar=fig.colorbar(im,cax=axc)
#cbar.set_label(r'$\tau_{21}$',size=fsize)
cbar.ax.tick_params(labelsize=fsize)
fig.gca()

#ax0.set_title(r"$\langle x_{\rm HI}\rangle=%.2f,\, \log_{10}(f_{\mathrm{X}})=%.1f,\, %s,\, t_{\rm int}=%d\,\rm hr,\, S_{\rm 147}=%.1f\,\mathrm{mJy},\, \alpha_{\rm R}=%.2f$" % (xHI_mean_mock,logfX_mock,telescope,tint,S147,alphaR),fontsize=fsize)
#ax0.set_title(r"$%s,\, t_{\rm int}=%d\,\mathrm{hr},\, N_{\rm obs}=%d$" % (telescope,tint,Nobs),fontsize=fsize)
ax0.set_title('Noise',fontsize=fsize)

plt.tight_layout()
plt.subplots_adjust(wspace=.0)
plt.savefig('covariance_matrix/covariance_matrix_noise_dk%.2f_%dkbins_z%.1f_fX%.2f_xHI%.2f_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f_%dLOS.pdf' % (d_log_k_bins,len(k_bins_cent),z_name,logfX_mock,xHI_mean_mock,telescope,spec_res,tint,S147,alphaR,Nobs))
plt.show()
plt.close()


fig = plt.figure(figsize=(5.95,5.))
gs = gridspec.GridSpec(1,3,width_ratios=[5.,0.1,0.25])

ax0= plt.subplot(gs[0,0])
im=ax0.imshow(Mcovar_sv,origin='lower',interpolation='none',cmap=plt.cm.bwr
          ,extent=[log_k_bins[0],log_k_bins[-1],log_k_bins[0],log_k_bins[-1]]
          #,aspect='auto',norm=LogNorm(vmin=3e-18,vmax=6e-13))
          ,aspect='auto',norm=SymLogNorm(linthresh=1e-16,vmin=-6e-13,vmax=6e-13))

ax0.set_xlabel(r'$\mathrm{log}_{10}(k/\rm MHz^{-1})$',fontsize=fsize)
ax0.set_ylabel(r'$\mathrm{log}_{10}(k/\rm MHz^{-1})$',fontsize=fsize)
ax0.set_xticks(np.arange(0.,2.5,1.))
ax0.set_yticks(np.arange(0.,2.5,1.))
ax0.xaxis.set_minor_locator(AutoMinorLocator())
ax0.yaxis.set_minor_locator(AutoMinorLocator())
ax0.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
     ,length=20,width=2,labelsize=fsize)
ax0.tick_params(axis='y',which='minor',direction='in',bottom=True,top=True,left=True,right=True
     ,length=10,width=2)
ax0.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
     ,length=20,width=2,labelsize=fsize)
ax0.tick_params(axis='x',which='minor',direction='in',bottom=True,top=True,left=True,right=True
     ,length=10,width=2)

axc= plt.subplot(gs[0,2])
cbar=fig.colorbar(im,pad=0.02,cax=axc)
#cbar.set_label(r'$\tau_{21}$',size=fsize)
cbar.ax.tick_params(labelsize=fsize)
fig.gca()

#ax0.set_title(r"$\langle x_{\rm HI}\rangle=%.2f,\, \log_{10}(f_{\mathrm{X}})=%.1f,\, %s,\, t_{\rm int}=%d\,\rm hr,\, S_{\rm 147}=%.1f\,\mathrm{mJy},\, \alpha_{\rm R}=%.2f$" % (xHI_mean_mock,logfX_mock,telescope,tint,S147,alphaR),fontsize=fsize)
#ax0.set_title(r"$\langle x_{\rm HI}\rangle=%.2f,\, \log_{10}(f_{\mathrm{X}})=%.1f,\, N_{\rm obs}=%d$" % (xHI_mean_mock,logfX_mock,Nobs),fontsize=fsize)
ax0.set_title('Sample variance',fontsize=fsize)

plt.tight_layout()
plt.subplots_adjust(wspace=.0)
plt.savefig('covariance_matrix/covariance_matrix_SV_dk%.2f_%dkbins_z%.1f_fX%.2f_xHI%.2f_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f_%dLOS.pdf' % (d_log_k_bins,len(k_bins_cent),z_name,logfX_mock,xHI_mean_mock,telescope,spec_res,tint,S147,alphaR,Nobs))
plt.show()
plt.close()


fig = plt.figure(figsize=(5.95,5.))
gs = gridspec.GridSpec(1,3,width_ratios=[5.,0.1,0.25])

ax0= plt.subplot(gs[0,0])
im=ax0.imshow(Mcovar_mock,origin='lower',interpolation='none',cmap=plt.cm.bwr
          ,extent=[log_k_bins[0],log_k_bins[-1],log_k_bins[0],log_k_bins[-1]]
          #,aspect='auto',norm=LogNorm(vmin=3e-18,vmax=6e-13))
          ,aspect='auto',norm=SymLogNorm(linthresh=1e-16,vmin=-6e-13,vmax=6e-13))

ax0.set_xlabel(r'$\mathrm{log}_{10}(k/\rm MHz^{-1})$',fontsize=fsize)
ax0.set_ylabel(r'$\mathrm{log}_{10}(k/\rm MHz^{-1})$',fontsize=fsize)
ax0.set_xticks(np.arange(0.,2.5,1.))
ax0.set_yticks(np.arange(0.,2.5,1.))
ax0.xaxis.set_minor_locator(AutoMinorLocator())
ax0.yaxis.set_minor_locator(AutoMinorLocator())
ax0.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
     ,length=20,width=2,labelsize=fsize)
ax0.tick_params(axis='y',which='minor',direction='in',bottom=True,top=True,left=True,right=True
     ,length=10,width=2)
ax0.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
     ,length=20,width=2,labelsize=fsize)
ax0.tick_params(axis='x',which='minor',direction='in',bottom=True,top=True,left=True,right=True
     ,length=10,width=2)

axc= plt.subplot(gs[0,2])
cbar=fig.colorbar(im,pad=0.02,cax=axc)
#cbar.set_label(r'$\tau_{21}$',size=fsize)
cbar.ax.tick_params(labelsize=fsize)
fig.gca()

#ax0.set_title(r"$\langle x_{\rm HI}\rangle=%.2f,\, \log_{10}(f_{\mathrm{X}})=%.1f,\, %s,\, t_{\rm int}=%d\,\rm hr,\, S_{\rm 147}=%.1f\,\mathrm{mJy},\, \alpha_{\rm R}=%.2f$" % (xHI_mean_mock,logfX_mock,telescope,tint,S147,alphaR),fontsize=fsize)
#ax0.set_title(r"$\langle x_{\rm HI}\rangle=%.2f,\, \log_{10}(f_{\mathrm{X}})=%.1f,\, %s,\, t_{\rm int}=%d\,\mathrm{hr},\, N_{\rm obs}=%d$" % (xHI_mean_mock,logfX_mock,telescope,tint,Nobs),fontsize=fsize)
ax0.set_title('Noise + Sample variance',fontsize=fsize)

plt.tight_layout()
plt.subplots_adjust(wspace=.0)
plt.savefig('covariance_matrix/covariance_matrix_SVandN_dk%.2f_%dkbins_z%.1f_fX%.2f_xHI%.2f_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f_%dLOS.pdf' % (d_log_k_bins,len(k_bins_cent),z_name,logfX_mock,xHI_mean_mock,telescope,spec_res,tint,S147,alphaR,Nobs))
plt.show()
plt.close()
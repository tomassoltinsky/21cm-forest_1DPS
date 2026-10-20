"""
Covariance matrix calculation and plotting for the 21-cm forest 1D power spectrum
corresponding to the uGMRT archival observations of PSO J352-15.

The covariance matrices are calculated separately for the simulated 21-cm forest
signal (sample variance), the measured off-source noise, and the combined
signal+noise data. The simulated signal ensemble is interpolated to the selected
xHI and THI values using the precomputed parameter-space dataset.

Version 25.4.2026
"""

#Import necessary packages
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
from matplotlib.colors import SymLogNorm
from scipy import interpolate


#Input parameters
z_name = float(sys.argv[1])         #Redshift of the 21cmFAST LOS data
spec_res = float(sys.argv[2])       #Spectral resolution of the telescope in kHz
xHI_mock = float(sys.argv[3])       #Mean neutral hydrogen fraction of the selected IGM model
logTHI_mock = float(sys.argv[4])    #log10 of the neutral IGM temperature in K
WLDlvl = int(sys.argv[5])           #Level of wavelet decomposition used for residual bandpass subtraction
f_flag = float(sys.argv[6])         #Threshold for flagging bad channels in units of mean off-source noise rms

telescope = 'uGMRT'                 #Telescope name
seed = 370                          #Random seed of the simulated signal dataset


#Frequency range of the flux spectrum
freq_low = 209.                     #Lower frequency limit in MHz
freq_high = 222.                    #Upper frequency limit in MHz

Nsamples = 1000                     #Number of signal PS samples for each IGM model
Ntargets = 1                        #Number of observed target sightlines

fsize = 20                          #Plot font size


#Prepare logarithmic k bins
d_log_k_bins = 0.25                 #Logarithmic k-bin width
log_k_bins = np.arange(0.,2.751+d_log_k_bins/2.,d_log_k_bins)

k_bins = np.power(10.,log_k_bins)   #Edges of logarithmic k bins
k_bins_cent = np.power(10.,log_k_bins+d_log_k_bins/2.)[:-1] #Centers of logarithmic k bins
Nkbins = len(k_bins_cent)           #Number of k bins

print('Number of k bins = %d' % Nkbins)
print('k range          = %.2f-%.2f MHz^-1' % (k_bins[0],k_bins[-1]))


#Load the simulated signal 1D PS parameter-space dataset
datafile = str('datasets/interpolators/individualPS_mask_signal-xHIlogfXTHI_%.1f-%.1fMHz_z%.1f_WLDlvl%d_fflag%.2f_%.1fkHz_Nobs%d_Nkbins%d_Nsamples%d_seed%d.dat' % (freq_low,freq_high,z_name,WLDlvl,f_flag,spec_res,Ntargets,Nkbins,Nsamples,seed))
data = np.fromfile(str(datafile),dtype=np.float32)

Npoints = int(data[0])                                              #Number of IGM models in the parameter space
xHI_mean = data[1+0*Npoints:1+1*Npoints]                            #Mean neutral hydrogen fraction
logT_HI = np.log10(data[1+2*Npoints:1+3*Npoints])                   #log10 of neutral IGM temperature in K

#Read and interpolate the individual-LOS signal 1D PS ensemble
PS_sim_ens = np.reshape(data[1+3*Npoints:],(Npoints,Nsamples,-1))
inter_fun_PS21_sv = interpolate.LinearNDInterpolator(np.transpose([xHI_mean,logT_HI]),PS_sim_ens)
PS_sv_bin = inter_fun_PS21_sv(xHI_mock,logTHI_mock)                  #Signal PS ensemble at selected IGM parameters


#Load the measured off-source noise 1D PS ensemble
datafile = 'obs_data/postprocessed_mask_obs_PS_%d-%dMHz_%s_%.1fkHz_WLDlvl%d_fflag%.2f.npy' % (freq_low,freq_high,'PSOJ352-15',spec_res,WLDlvl,f_flag)
data = np.load(datafile, allow_pickle=True).item()

PS_noise_bin = data['PS_noise']                                     #Measured off-source noise PS ensemble

print('Noise PS dimensions before k selection = ',PS_noise_bin.shape)

#Remove the first two k bins to match the k range used for the covariance analysis
PS_noise_bin = PS_noise_bin[:,2:]

print('Noise PS dimensions after k selection  = ',PS_noise_bin.shape)


#Construct the combined signal+noise PS ensemble
PS_nsv_bin = PS_sv_bin + PS_noise_bin


#Compute covariance matrix for sample variance only
Mcovar_sv = np.cov(np.transpose(PS_sv_bin))

print('Covariance matrix - sample variance')
print('  Minimum absolute value = %.3e' % np.amin(np.abs(Mcovar_sv)))
print('  Minimum value          = %.3e' % np.amin(Mcovar_sv))
print('  Maximum value          = %.3e' % np.amax(Mcovar_sv))
print('  Diagonal               = ',np.diag(Mcovar_sv))


#Compute covariance matrix for measured noise only
Mcovar_noise = np.cov(np.transpose(PS_noise_bin))

print('Covariance matrix - noise')
print('  Minimum absolute value = %.3e' % np.amin(np.abs(Mcovar_noise)))
print('  Minimum value          = %.3e' % np.amin(Mcovar_noise))
print('  Maximum value          = %.3e' % np.amax(Mcovar_noise))
print('  Diagonal               = ',np.diag(Mcovar_noise))


#Compute covariance matrix for signal+noise
Mcovar_nsv = np.cov(np.transpose(PS_nsv_bin))

print('Covariance matrix - signal+noise')
print('  Minimum absolute value = %.3e' % np.amin(np.abs(Mcovar_nsv)))
print('  Minimum value          = %.3e' % np.amin(Mcovar_nsv))
print('  Maximum value          = %.3e' % np.amax(Mcovar_nsv))
print('  Diagonal               = ',np.diag(Mcovar_nsv))


#Compare the noise and sample-variance contributions to the diagonal covariance
print('Noise/sample-variance ratio:')
print(np.diag(Mcovar_noise)/np.diag(Mcovar_sv))


#Plot the covariance matrix for sample variance
fig = plt.figure(figsize=(5.95,5.))
gs = gridspec.GridSpec(1,3,width_ratios=[5.,0.1,0.25])

ax0= plt.subplot(gs[0,0])
im=ax0.imshow(Mcovar_sv,origin='lower',interpolation='none',cmap=plt.cm.bwr
          ,extent=[log_k_bins[0],log_k_bins[-1],log_k_bins[0],log_k_bins[-1]]
          ,aspect='auto',norm=SymLogNorm(linthresh=1e-12,vmin=-2e-7,vmax=2e-7))

ax0.set_xlabel(r'$\mathrm{log}_{10}(\kappa_{\nu}/\rm MHz^{-1})$',fontsize=fsize)
ax0.set_ylabel(r'$\mathrm{log}_{10}(\kappa_{\nu}/\rm MHz^{-1})$',fontsize=fsize)
ax0.set_xticks(np.arange(log_k_bins[0],log_k_bins[-1],1.))
ax0.set_yticks(np.arange(log_k_bins[0],log_k_bins[-1],1.))
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
cbar.ax.tick_params(labelsize=fsize)

ax0.set_title('Sample variance',fontsize=fsize)

plt.tight_layout()
plt.subplots_adjust(wspace=.0)
plt.savefig('plots/Soltinsky_2026/covariance_matrix_obs_PSOJ352−15_SV_%.1f-%.1fMHz_%.1fkHz_%s_dk%.2f_%dkbins_xHI%.2f_logTHI%.2f_WLDlvl%d_fflag%.2f_kappa.png' % (freq_low,freq_high,spec_res,telescope,d_log_k_bins,Nkbins,xHI_mock,logTHI_mock,WLDlvl,f_flag))
plt.show()
plt.close()


#Plot the covariance matrix for measured noise
fig = plt.figure(figsize=(5.95,5.))
gs = gridspec.GridSpec(1,3,width_ratios=[5.,0.1,0.25])

ax0= plt.subplot(gs[0,0])
im=ax0.imshow(Mcovar_noise,origin='lower',interpolation='none',cmap=plt.cm.bwr
          ,extent=[log_k_bins[0],log_k_bins[-1],log_k_bins[0],log_k_bins[-1]]
          ,aspect='auto',norm=SymLogNorm(linthresh=1e-12,vmin=-2e-7,vmax=2e-7))

ax0.set_xlabel(r'$\mathrm{log}_{10}(\kappa_{\nu}/\rm MHz^{-1})$',fontsize=fsize)
ax0.set_ylabel(r'$\mathrm{log}_{10}(\kappa_{\nu}/\rm MHz^{-1})$',fontsize=fsize)
ax0.set_xticks(np.arange(log_k_bins[0],log_k_bins[-1],1.))
ax0.set_yticks(np.arange(log_k_bins[0],log_k_bins[-1],1.))
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
cbar.ax.tick_params(labelsize=fsize)

ax0.set_title('Noise',fontsize=fsize)

plt.tight_layout()
plt.subplots_adjust(wspace=.0)
plt.savefig('plots/Soltinsky_2026/covariance_matrix_obs_PSOJ352−15_noise_%.1f-%.1fMHz_%.1fkHz_%s_dk%.2f_%dkbins_xHI%.2f_logTHI%.2f_WLDlvl%d_fflag%.2f_kappa.png' % (freq_low,freq_high,spec_res,telescope,d_log_k_bins,Nkbins,xHI_mock,logTHI_mock,WLDlvl,f_flag))
plt.show()
plt.close()


#Plot the covariance matrix for signal+noise
fig = plt.figure(figsize=(5.95,5.))
gs = gridspec.GridSpec(1,3,width_ratios=[5.,0.1,0.25])

ax0= plt.subplot(gs[0,0])
im=ax0.imshow(Mcovar_nsv,origin='lower',interpolation='none',cmap=plt.cm.bwr
          ,extent=[log_k_bins[0],log_k_bins[-1],log_k_bins[0],log_k_bins[-1]]
          ,aspect='auto',norm=SymLogNorm(linthresh=1e-12,vmin=-2e-7,vmax=2e-7))

ax0.set_xlabel(r'$\mathrm{log}_{10}(\kappa_{\nu}/\rm MHz^{-1})$',fontsize=fsize)
ax0.set_ylabel(r'$\mathrm{log}_{10}(\kappa_{\nu}/\rm MHz^{-1})$',fontsize=fsize)
ax0.set_xticks(np.arange(log_k_bins[0],log_k_bins[-1],1.))
ax0.set_yticks(np.arange(log_k_bins[0],log_k_bins[-1],1.))
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
cbar.ax.tick_params(labelsize=fsize)

ax0.set_title('Noise + Sample variance',fontsize=fsize)

plt.tight_layout()
plt.subplots_adjust(wspace=.0)
plt.savefig('plots/Soltinsky_2026/covariance_matrix_obs_PSOJ352−15_SVandN_%.1f-%.1fMHz_%.1fkHz_%s_dk%.2f_%dkbins_xHI%.2f_logTHI%.2f_WLDlvl%d_fflag%.2f_kappa.png' % (freq_low,freq_high,spec_res,telescope,d_log_k_bins,Nkbins,xHI_mock,logTHI_mock,WLDlvl,f_flag))
plt.show()
plt.close()
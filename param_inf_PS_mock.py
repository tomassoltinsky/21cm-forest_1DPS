'''
Plot the 21-cm forest 1D PS from the mock observation, MCMC inferrence and MCMC posterior draws.

Version 16.04.2024
'''
#Import necessary packages
import sys
import glob, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal as scisi
from astropy.convolution import convolve, Box1DKernel
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
from scipy import interpolate
from scipy.optimize import minimize
import emcee
import corner
import instrumental_features
from numpy import random
import time

#Input parameters
z_name = float(sys.argv[1])         #redshift
dvH = float(sys.argv[2])            #used rebinning for line profile convolution in m/s
spec_res = float(sys.argv[3])       #spectral resolution of the telescope in kHz
xHI_mean_mock = float(sys.argv[4])  #mock HI fraction
logfX_mock = float(sys.argv[5])     #mock logfX
path_LOS = '../../datasets/21cmFAST_los/los/'
telescope = str(sys.argv[6])        #telescope
S147 = float(sys.argv[7])           #intrinsic flux density of background source at 147MHz in mJy
alphaR = float(sys.argv[8])         #radio spectrum power-law index of background source
tint = float(sys.argv[9])           #intergration time for the observation in h
Nobs = int(sys.argv[10])            #number of LOS to be observed
Nkbins = int(sys.argv[11])          #number of k bins

path = 'MCMC_samples'
Nsteps = 100000
Ndraws = 20
n_los = 1000
Nlos = 1000
Nsamples = 10000



#Data for mock observation

#Prepare k bins
d_log_k_bins = 0.25
log_k_bins = np.arange(-1.0-d_log_k_bins/2.,3.+d_log_k_bins/2.,d_log_k_bins)
log_k_bins = log_k_bins[2:2+Nkbins+1]

k_bins = np.power(10.,log_k_bins)
k_bins_cent = np.power(10.,log_k_bins+d_log_k_bins/2.)[:-1]
print(k_bins_cent)

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
PS_signal_mock_std = np.std(PS_nsv_ens,axis=0)

random.seed(0)

LOS = random.randint(0,Nlos-1,size=Nobs)
signal_multiobs = PS_nsv_bin[LOS][:]
PS_signal_mock = np.mean(signal_multiobs,axis=0)

print(PS_signal_mock+PS_signal_mock_std)
print(PS_signal_mock)
print(PS_signal_mock-PS_signal_mock_std)

print('Mock data prepared')



#Find all of the datasets for the interpolation
files = glob.glob(path_LOS+'*.dat')

#Remove files if needed
files_to_remove = glob.glob(path_LOS+'*fXnone*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])

logfX = np.empty(len(files))
xHI_mean = np.empty(len(files))
PS_sim_mean = np.empty((len(files),len(k_bins_cent)))

done_perc = 0.1
#Find all of the parameter values in simulated data and read the data for interpolation
for j in range(len(files)):
   
   data = np.fromfile(str(files[j]),dtype=np.float32)
   logfX[j] = data[9]
   xHI_mean[j] = data[11]

   #Read data for signal
   datafile = str('1DPS_dimensionless/1DPS_signal/power_spectrum_signal_21cmFAST_200Mpc_z%.1f_fX%.2f_xHI%.2f_%dkHz_%dLOS.dat' % (z_name,logfX[j],xHI_mean[j],spec_res,Nlos))
   data = np.fromfile(str(datafile),dtype=np.float32)
   Nlos = int(data[0])
   n_kbins = int(data[1])
   k = data[2:2+n_kbins]
   PS_signal = data[2+n_kbins+0*n_kbins*Nlos:2+n_kbins+1*n_kbins*Nlos]
   PS_signal = np.reshape(PS_signal,(Nlos,n_kbins))[:n_los,:]

   #Bin the PS data
   PS_signal_bin = np.empty((n_los,len(k_bins_cent)))

   for i in range(n_los):
      for l in range(len(k_bins_cent)):
         ind = np.where((k>=k_bins[l]) & (k<k_bins[l+1]))[0]
         PS_signal_bin[i,l] = np.mean(PS_signal[i,ind])

   #Take mean for each k bin
   PS_sim_ens = instrumental_features.multi_obs(PS_signal_bin,Nobs,Nsamples)
   PS_sim_mean[j,:] = np.mean(PS_sim_ens,axis=0)

   #Track the progress
   done_files = (j+1)/len(files)
   if done_files>=done_perc:
         print('Done %.2f' % done_files)
         done_perc = done_perc+0.1

#Set up N-dimensional linear interpolator for calculating P21 for any parameter values within the range given in the prior function
inter_fun_PS21 = interpolate.LinearNDInterpolator(np.transpose([xHI_mean,logfX]),PS_sim_mean)

print('Interpolator prepared')



#Data for inferred and posterior draws PS

data = np.load('%s/flatsamp_200Mpc_xHI%.2f_fX%.1f_%s_%dkHz_Smin%.1fmJy_alphaR%.2f_t%dh_dk%.2f_%dkbins_%dsteps.npy' % (path,xHI_mean_mock,logfX_mock,telescope,spec_res,S147,alphaR,tint,d_log_k_bins,Nkbins,Nsteps))
xHI_inf = data[0,0]
logfX_inf = data[0,1]
print(xHI_inf,logfX_inf)

ind_draws = np.random.randint(1,data.shape[0],Ndraws)
data = data[ind_draws][:]
xHI_draws = data[:,0]
logfX_draws = data[:,1]
print(xHI_draws)
print(logfX_draws)

print('Inferred and posterior draws PS prepared')



fsize = 20
fig = plt.figure(figsize=(10.,5.))
gs = gridspec.GridSpec(1,1)
ax0 = plt.subplot(gs[0,0])

#Plot mock observation data
ax0.errorbar(k_bins_cent,PS_signal_mock,yerr=PS_signal_mock_std,fmt=' ',marker='o',capsize=5,color='darkorange',label='Mock data')
ax0.plot([1e-20,1e-20],[1e-20,2e-20],'-',linewidth=2,color='fuchsia',label='Inferred')
ax0.plot([1e-20,1e-20],[1e-20,2e-20],'-',color='royalblue',alpha=0.5,label='Posterior draws')
plt.legend(frameon=False,loc='lower right',fontsize=fsize-3)

#Plot posterior draws PS
for i in range(len(xHI_draws)):
   ax0.plot(k_bins_cent,inter_fun_PS21(xHI_draws[i],logfX_draws[i]),'-',color='royalblue',alpha=0.5,label='Posterior draws')

#Plot inferred PS
ax0.plot(k_bins_cent,inter_fun_PS21(xHI_inf,logfX_inf),'-',linewidth=2,color='fuchsia',label='Inferred')

ax0.set_xlim(0.25,np.ceil(k_bins_cent[-1]))
ax0.set_ylim(1e-7,3e-6)
ax0.set_xscale('log')
ax0.set_yscale('log')
ax0.set_xlabel(r'$k \,\rm [MHz^{-1}]$', fontsize=fsize)
ax0.set_ylabel(r'$kP_{21}\,\rm [MHz]$', fontsize=fsize)
ax0.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=20/3*2,width=2/3*2,labelsize=fsize)
ax0.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=20/3*2,width=2/3*2,labelsize=fsize)
ax0.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=10/3*2,width=2/3*2)

plt.tight_layout()
plt.subplots_adjust(hspace=2.0)
plt.savefig('1DPS_plots/power_spectrum_mockandinf_200Mpc_z%.1f_xHI%.2f_fX%.1f_%s_%dkHz_Smin%.1fmJy_alphaR%.2f_t%dh_%dkbins_mockPS.pdf' % (z_name,xHI_mean_mock,logfX_mock,telescope,spec_res,S147,alphaR,tint,Nkbins))
plt.close()


#Plot corner plot for the inferred and posterior draws PS
fsize = 14
fig, axes = plt.subplots(2,sharex=True, figsize=(5.,5.))
min_logfX = -4.
max_logfX = 1.
labels = [r"$\langle x_{\rm HI}\rangle$",r"$\mathrm{log}_{10}(f_{\mathrm{X}})$"]

flat_samples = np.load('MCMC_samples/flatsamp_200Mpc_xHI%.2f_fX%.1f_%s_%dkHz_Smin%.1fmJy_alphaR%.2f_t%dh_dk%.2f_%dkbins_%dsteps.npy' % (xHI_mean_mock,logfX_mock,telescope,spec_res,S147,alphaR,tint,d_log_k_bins,Nkbins,Nsteps))
sett=dict(fontsize=fsize)

fig = corner.corner(flat_samples,range=[[0.,1.],[min_logfX,max_logfX]],color='royalblue',levels=[1-np.exp(-0.5),1-np.exp(-2.)],smooth=False,labels=labels,label_kwargs=sett
                                 ,show_titles=True,title_fmt='.2f',title_kwargs=sett,truths=[xHI_inf,logfX_inf],truth_color='fuchsia')
corner.overplot_points(fig=fig,xs=[[np.nan,np.nan],[np.nan,np.nan],[xHI_mean_mock,logfX_mock],[np.nan,np.nan]],marker='x',markersize=10,markeredgewidth=3,color='darkorange')

for ax in fig.get_axes():
      ax.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
            ,length=11.25,width=1.125,labelsize=fsize)
      ax.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
            ,length=11.25,width=1.125,labelsize=fsize)
      ax.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True
            ,length=5.625,width=1.125)

plt.savefig('1DPS_plots/power_spectrum_mockandinf_200Mpc_z%.1f_xHI%.2f_fX%.1f_%s_%dkHz_Smin%.1fmJy_alphaR%.2f_t%dh_%dkbins_corner.pdf' % (z_name,xHI_mean_mock,logfX_mock,telescope,spec_res,S147,alphaR,tint,Nkbins))
plt.show()
plt.close()
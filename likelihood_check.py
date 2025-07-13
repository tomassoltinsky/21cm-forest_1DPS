"""
Code to check covariance used.

Version 18.6.2025

"""
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
import openpyxl
import time
from scipy import interpolate
from scipy.optimize import minimize
import emcee
import corner
import instrumental_features
from numpy import random
random.seed(5)
np.random.seed(5)
import time
import csv
start_clock = time.perf_counter()
#Input parameters
z_name = float(sys.argv[1])         #redshift
dvH = float(sys.argv[2])            #used rebinning for line profile convolution in m/s
spec_res = float(sys.argv[3])       #spectral resolution of the telescope in kHz
path_LOS = 'datasets/21cmFAST_los/los/'
telescope = str(sys.argv[4])
S147 = float(sys.argv[5])           #intrinsic flux density of background source at 147MHz in mJy
alphaR = float(sys.argv[6])         #radio spectrum power-law index of background source
tint = float(sys.argv[7])           #intergration time for the observation in h
Nobs = int(sys.argv[8])
xHI_mean_mock = float(sys.argv[9])
logfX_mock    = float(sys.argv[10])
k_ind = int(sys.argv[11])

Nlos = 1000
n_los = 1000
Nsamples = 10000
Nkbins = 6

min_logfX = -4.
max_logfX = 1.
#min_fX = 0.0001
#max_fX = 10.
min_xHI = 0.01
max_xHI = 0.99



#Prepare k bins
d_log_k_bins = 0.25
log_k_bins = np.arange(-1.0-d_log_k_bins/2.,3.+d_log_k_bins/2.,d_log_k_bins)
log_k_bins = log_k_bins[2:2+Nkbins+1]

k_bins = np.power(10.,log_k_bins)
k_bins_cent = np.power(10.,log_k_bins+d_log_k_bins/2.)[:-1]
#print(k_bins_cent)
print('k_ind = %.2fMHz^-1' % k_bins_cent[k_ind])



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
PS_nsv_mean = np.mean(PS_nsv_ens,axis=0)
#PS_nsv_sigma = np.std(PS_nsv_ens,axis=0)
Mcovar_nsv = np.cov(np.transpose(PS_nsv_ens))

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

PS_noise_mean = np.mean(PS_noise_bin,axis=0)

PS_nsub = PS_nsv_bin-PS_noise_mean
PS_nsub_ens = instrumental_features.multi_obs(PS_nsub,Nobs,Nsamples)



#Choose which combination of Mcovar and y to be used
#Mcovar = Mcovar_mock
Mcovar = Mcovar_nsv
#PS_mock_mean = signal_multiobs_mean_mock

#print('Mcovar.shape=', Mcovar.shape)
#print('PS_mock_mean.shape=', PS_mock_mean.shape)
#print(Mcovar)
print('Covariance matrix and mock data prepared')

stop_clock = time.perf_counter()
time_taken = (stop_clock-start_clock)
print('It took %.3fs of your life' % time_taken)



'''
done_perc = 0.1
print('Initiate interpolator setup')
PS_sim_mean = np.empty((len(logfX),len(k_bins_cent)))

#Find all of the parameter values in simulated data and read the data for interpolation
for j in range(len(files)):
    
    data = np.fromfile(str(files[j]),dtype=np.float32)
    logfX[j] = data[9]
    xHI_mean[j] = data[11]
    #print('f_X=%.2f, <x_HI,box>=%.8f' % (logfX[j],xHI_mean[j]))

    #Read data for signal
    datafile = str('1DPS_dimensionless/1DPS_signal/power_spectrum_signal_21cmFAST_200Mpc_z%.1f_fX%.2f_xHI%.2f_%dkHz_%dLOS.dat' % (z_name,logfX[j],xHI_mean[j],spec_res,Nlos))
    #datafile = str('1DPS_dimensionless/1DPS_signalandnoise/power_spectrum_signal_21cmFAST_50Mpc_z%.1f_fX%.2f_xHI%.2f_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f_%dLOS.dat' % (z_name,logfX[j],xHI_mean[j],telescope,spec_res,tint,S147,alphaR,Nlos))
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

array = np.append(xHI_mean,logfX)
array = np.append(array,PS_sim_mean)
array.astype('float32').tofile('1DPS_dimensionless/kP21_sim_200Mpc_z%.1f_%dkHz_Nobs%d_Nsamples%d_kbins%d.dat' % (z_name,spec_res,Nobs,Nsamples,len(k_bins_cent)),sep='')
'''


N_parcom = 534
datafile = str('1DPS_dimensionless/kP21_sim_200Mpc_z%.1f_%dkHz_Nobs%d_Nsamples%d_kbins%d.dat' % (z_name,spec_res,Nobs,Nsamples,len(k_bins_cent)))
data = np.fromfile(str(datafile),dtype=np.float32)
xHI_mean = data[0*N_parcom:1*N_parcom]
logfX    = data[1*N_parcom:2*N_parcom]
PS_sim_mean = np.reshape(data[2*N_parcom:],(N_parcom,len(k_bins_cent)))

#Set up N-dimensional linear interpolator for calculating P21 for any parameter values within the range given in the prior function
#inter_fun_likelihood = interpolate.LinearNDInterpolator(np.transpose([xHI_mean,logfX]),Likelihood)
inter_fun_PS21 = interpolate.LinearNDInterpolator(np.transpose([xHI_mean,logfX]),PS_sim_mean)

print('Interpolator set up')
stop_clock = time.perf_counter()
time_taken = (stop_clock-start_clock)
print('It took %.3fs of your life' % time_taken)


'''
#Define parameters to be estimated
#The below steps are based on tutarial for emcee package (Foreman-Mackey et al. 2013) at https://emcee.readthedocs.io/en/stable/tutorials/line/
par_spc=np.array([xHI_mean,logfX])
labels = [r"$\langle x_{\rm HI}\rangle$",r"$\mathrm{log}_{10}(f_{\mathrm{X}})$"]

#Define prior distribution as uniform within the range of our simulated data
logfX_cut_min = -0.6
slope_cut = (0.81-0.99)/(max_logfX-logfX_cut_min)
def log_prior(theta):
    xHI_mean1, logfX1 = theta
    #if min_xHI <= xHI_mean1 <= max_xHI and min_fX <= fX1 <= max_fX:
    if min_xHI <= xHI_mean1 <= max_xHI and min_logfX <= logfX1 < logfX_cut_min:
      return 0
    elif logfX_cut_min <= logfX1 <= max_logfX and min_xHI <= xHI_mean1 <= 0.99+slope_cut*(logfX1-logfX_cut_min):
      return 0
    return -np.inf

#Define (negative) likelihood function based on covariance matrix
def log_likelihood(theta,signal_mock_mean,covar_mat):
  xHI_mean1, logfX1 = theta 

  # Add a small regularization term to the covariance matrix
  #regularization = 1e-6 * np.eye(covar_mat.shape[0])
  #covar_mat += regularization

  d_mat = signal_mock_mean-inter_fun_PS21(xHI_mean1, logfX1)
  d_mat_T = np.transpose(d_mat)
  covar_mat_inv = np.linalg.inv(covar_mat)
  covar_mat_det = np.linalg.det(covar_mat)

  #print(d_mat)
  #print(d_mat_T)
  #rint(covar_mat)
  #rint(covar_mat_det)
  #print(covar_mat_inv)
  #print(np.linalg.multi_dot([covar_mat,covar_mat_inv]))
  #print(np.linalg.multi_dot([d_mat_T,covar_mat_inv,d_mat]))
  #print(np.log(covar_mat_det))

  return 0.5*np.linalg.multi_dot([d_mat_T,covar_mat_inv,d_mat])+np.log(covar_mat_det)

#Define posterior function based on Bayes therom. Note that no normalization is assumed.
def log_posterior(theta,signal_mock_mean,covar_mat):
    LP = log_prior(theta)
    if not np.isfinite(LP):
        return -np.inf
    return LP-log_likelihood(theta,signal_mock_mean,covar_mat)
'''


#Define Gaussian likelihood function based on a single covariance matrix and d element corresponding to the k of interest
def likelihood_atk(theta,signal_mock_mean,covar_mat,k_i):
  xHI_mean1, logfX1 = theta 

  d_mat_sim = signal_mock_mean-inter_fun_PS21(xHI_mean1, logfX1)[k_i]
  d_mat_nsv = signal_mock_mean-PS_nsv_mean[k_i]

  sigma_i = covar_mat[k_i,k_i]
  print('sigma_i=%.2e' % sigma_i)

  return 1/np.sqrt(2*np.pi*sigma_i)*np.exp(-d_mat_sim**2/(2*sigma_i)), 1/np.sqrt(2*np.pi*sigma_i)*np.exp(-d_mat_nsv**2/(2*sigma_i))



#Prepare bins for histogram
PS_bins = np.arange(0,2.e-5,1e-7)
PS_bins_cent = PS_bins[:-1]+(PS_bins[1]-PS_bins[0])/2.

#Calculate assumed Gaussian likelihood
likelihood = likelihood_atk([xHI_mean_mock,logfX_mock],PS_bins_cent,Mcovar,k_ind)

#And plot
fsize = 16
fig = plt.figure(figsize=(5.,5.))
gs = gridspec.GridSpec(1,1)

ax = plt.subplot(gs[0,0])  

PS_ens_atk_nsv = np.empty(Nsamples)
PS_ens_atk_nsub = np.empty(Nsamples)
for i in range(Nsamples):
   PS_ens_atk_nsv[i] = PS_nsv_ens[i][k_ind]
   PS_ens_atk_nsub[i] = PS_nsub_ens[i][k_ind]

ax.hist(PS_ens_atk_nsv,bins=PS_bins,density=True,color='royalblue',alpha=0.25,label=r'Sampled $\langle P_{21}^{\rm S+N}\rangle_{10}$')
ax.hist(PS_ens_atk_nsub,bins=PS_bins,density=True,color='darkorange',alpha=0.25,label=r'Sampled $\langle P_{21}^{\rm N_{sub}}\rangle_{10}$')
ax.plot(PS_bins_cent,likelihood[0],'-',color='fuchsia',label=r'$\mathcal{L}_{\rm B}(P_{21}|\mathbf{\theta})$')
ax.legend(frameon=False,loc=[0.1,0.65],fontsize=fsize)
ax.set_xlim(0,1.5e-5)
#ax.set_xlim(0,.7e-5)
ax.set_ylim(0.,5.5e5)
ax.set_xlabel(r'$P_{21}(k=%.2f\,\rm MHz^{-1})$' % k_bins_cent[k_ind], fontsize=fsize)
ax.set_ylabel(r'$PDF$', fontsize=fsize)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=2)

plt.tight_layout()
plt.subplots_adjust(hspace=.0)
plt.savefig('likelihood_check/likelihood_check_z%.1f_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f_Nobs%d_k%.2f.png' % (z_name,telescope,spec_res,tint,S147,alphaR,Nobs,k_bins_cent[k_ind]),dpi=300)
plt.show()
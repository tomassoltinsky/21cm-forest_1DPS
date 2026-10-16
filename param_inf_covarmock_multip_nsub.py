"""
Parameter inference code based on Bayesian methods for multiple mock IGM models.

Uses 2D interpolator for 1D PS from 21-cm forest and performs inference on noise-subtracted mock data.

Version 17.2.2025

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
import time
start_clock = time.perf_counter()

#Input parameters
z_name = float(sys.argv[1])         #redshift
dvH = float(sys.argv[2])            #used rebinning for line profile convolution in m/s
spec_res = float(sys.argv[3])       #spectral resolution of the telescope in kHz
path_LOS = '../../datasets/21cmFAST_los/los/'
telescope = str(sys.argv[4])        #telescope for mock observation
S147 = float(sys.argv[5])           #intrinsic flux density of background source at 147MHz in mJy
alphaR = float(sys.argv[6])         #radio spectrum power-law index of background source
tint = float(sys.argv[7])           #intergration time for the observation in h
Nobs = int(sys.argv[8])             #number of LOS for mock observations

#Select mock IGM models for parameter inference
xHI_mean_mock = [0.11,0.80,0.52,0.11,0.80]
logfX_mock    = [-1.0,-1.0,-2.0,-3.0,-3.0]

#Set number of LOS, samples and k bins
Nlos = 1000
n_los = 1000
Nsamples = 10000
Nkbins = 6

#Set parameter ranges for the prior
min_logfX = -4.
max_logfX = 1.
#min_fX = 0.0001
#max_fX = 10.
min_xHI = 0.01
max_xHI = 0.99

#Find all of the datasets for the interpolation
files = glob.glob(path_LOS+'*.dat')

#Remove files without X-ray preheating
files_to_remove = glob.glob(path_LOS+'*fXnone*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])

#Or one can select specific models
'''
files = ['../../datasets/21cmFAST_los/los/los_50Mpc_256_n1000_z6.000_fX-3.0_xHI0.11.dat'
        ,'../../datasets/21cmFAST_los/los/los_50Mpc_256_n1000_z6.000_fX-3.0_xHI0.61.dat']
'''


#Loop over IGM models
for param_i in range(len(logfX_mock)):

  print('Mock: xHI=%.2f, fX=%.1f' % (xHI_mean_mock[param_i],logfX_mock[param_i]))

  logfX = np.empty(len(files))
  xHI_mean = np.empty(len(files))

  #Prepare k bins
  d_log_k_bins = 0.25
  log_k_bins = np.arange(-1.0-d_log_k_bins/2.,3.+d_log_k_bins/2.,d_log_k_bins)
  log_k_bins = log_k_bins[2:2+Nkbins+1]

  k_bins = np.power(10.,log_k_bins)
  k_bins_cent = np.power(10.,log_k_bins+d_log_k_bins/2.)[:-1]
  print(k_bins_cent)

  '''
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

  #Select a reproducible set of LOS for the mock observation
  random.seed(0)

  LOS = random.randint(0,Nlos-1,size=Nobs)
  signal_multiobs = PS_sv_bin[LOS][:]
  signal_multiobs_mean_sv = np.mean(signal_multiobs,axis=0)

  #Compute covariance matrix for sample variance only
  Mcovar_sv = np.cov(np.transpose(PS_sv_ens))


  '''
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
  print(PS_noise_mean.shape)
  '''
  #Get ensemble of observations of multiple LOS
  #sig_PS_noise = np.std(PS_noise_bin,axis=0)
  PS_noise_ens = instrumental_features.multi_obs(PS_noise_bin,Nobs,Nsamples)

  random.seed(0)

  LOS = random.randint(0,Nlos-1,size=Nobs)
  signal_multiobs = PS_noise_bin[LOS][:]
  signal_multiobs_mean_noise= np.mean(signal_multiobs,axis=0)

  #Compute covariance matrix for noise only
  Mcovar_noise = np.cov(np.transpose(PS_noise_ens))
  '''


  #Read the mock (signal+noise) data for which we want to estimate parameters
  datafile = str('1DPS_dimensionless/1DPS_signalandnoise/power_spectrum_signal_21cmFAST_200Mpc_z%.1f_fX%.2f_xHI%.2f_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f_%dLOS.dat' % (z_name,logfX_mock[param_i],xHI_mean_mock[param_i],telescope,spec_res,tint,S147,alphaR,Nlos))
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

  #Subtract mean noise power spectrum from the mock signal+noise power spectrum
  PS_nsub = PS_nsv_bin-PS_noise_mean

  random.seed(0)

  LOS = random.randint(0,Nlos-1,size=Nobs)
  signal_multiobs = PS_nsub[LOS][:]

  signal_multiobs_mean_mock = np.mean(signal_multiobs,axis=0)

  #Compute covariance matrix from the signal+noise ensemble
  Mcovar_mock = np.cov(np.transpose(PS_nsv_ens))



  #Choose which combination of Mcovar and mock to be used
  Mcovar = Mcovar_mock
  PS_mock_mean = signal_multiobs_mean_mock

  print('Mcovar.shape=', Mcovar.shape)

  print('Covariance matrix and mock data prepared')

  stop_clock = time.perf_counter()
  time_taken = (stop_clock-start_clock)
  print('It took %.3fs of your life' % time_taken)

  done_perc = 0.1
  print('Initiate interpolator setup')
  PS_sim_mean = np.empty((len(logfX),len(k_bins_cent)))
  #The following loop can be commented out if interpolator data already pre-computed
  '''
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
  array.astype('float32').tofile('1DPS_dimensionless/kP21_sim_200Mpc_z%.1f_%dkHz_Nobs%d_Nsamples%d.dat' % (z_name,spec_res,Nobs,Nsamples),sep='')

  '''
  #Read precomputed mean simulated power spectra for the interpolator
  datafile = str('1DPS_dimensionless/kP21_sim_200Mpc_z%.1f_%dkHz_Nobs%d_Nsamples%d.dat' % (z_name,spec_res,Nobs,Nsamples))
  data = np.fromfile(str(datafile),dtype=np.float32)
  xHI_mean = data[0*len(files):1*len(files)]
  logfX    = data[1*len(files):2*len(files)]
  PS_sim_mean = np.reshape(data[2*len(files):],(len(files),len(k_bins_cent)))
  
  #Set up N-dimensional linear interpolator for calculating P21 for any parameter values within the range given in the prior function
  inter_fun_PS21 = interpolate.LinearNDInterpolator(np.transpose([xHI_mean,logfX]),PS_sim_mean)

  print('Interpolator set up')
  stop_clock = time.perf_counter()
  time_taken = (stop_clock-start_clock)
  print('It took %.3fs of your life' % time_taken)



  #Define parameters to be estimated
  #The below steps are based on tutorial for emcee package (Foreman-Mackey et al. 2013) at https://emcee.readthedocs.io/en/stable/tutorials/line/
  par_spc=np.array([xHI_mean,logfX])
  labels = [r"$\langle x_{\rm HI}\rangle$",r"$\mathrm{log}_{10}(f_{\mathrm{X}})$"]

  #Define prior distribution as uniform within the range of our simulated data
  #See Fig. 2 in Patil et al. 2026, 546, 4, 18 for the simulated IGM models array
  logfX_cut_min = -0.6
  slope_cut = (0.81-0.99)/(max_logfX-logfX_cut_min)
  def log_prior(theta):
      xHI_mean1, logfX1 = theta
      if min_xHI <= xHI_mean1 <= max_xHI and min_logfX <= logfX1 < logfX_cut_min:
        return 0
      elif logfX_cut_min <= logfX1 <= max_logfX and min_xHI <= xHI_mean1 <= 0.99+slope_cut*(logfX1-logfX_cut_min):
        return 0
      return -np.inf

  #Define (negative) likelihood function based on covariance matrix
  def log_likelihood(theta,signal_mock_mean,covar_mat):
    xHI_mean1, logfX1 = theta 
    d_mat = signal_mock_mean-inter_fun_PS21(xHI_mean1, logfX1)
    d_mat_T = np.transpose(d_mat)
    covar_mat_inv = np.linalg.inv(covar_mat)
    covar_mat_det = np.linalg.det(covar_mat)

    #print(d_mat)
    #print(d_mat_T)
    #print(covar_mat)
    #print(covar_mat_det)
    #print(covar_mat_inv)
    #print(np.linalg.multi_dot([covar_mat,covar_mat_inv]))

    return 0.5*np.linalg.multi_dot([d_mat_T,covar_mat_inv,d_mat])+np.log(covar_mat_det)

  #Define posterior function based on Bayes therom. Note that no normalization is assumed.
  def log_posterior(theta,signal_mock_mean,covar_mat):
      LP = log_prior(theta)
      if not np.isfinite(LP):
          return -np.inf
      return LP-log_likelihood(theta,signal_mock_mean,covar_mat)



  #Initiate MCMC for the parameter estimation
  n_walk=64
  ndim=2
  Nsteps = 100000

  initial = np.array([0.3,-2.2])# + 0.1 * np.random.randn(2)
  soln = minimize(log_likelihood, initial, args=(PS_mock_mean,Mcovar),bounds=([min_xHI,max_xHI],[min_logfX,max_logfX]))
  #Move best-fit starting point slightly away from prior boundaries if needed
  if soln.x[0]==0.01:
     soln.x[0] = 0.05
  if soln.x[1]==1.:
     soln.x[1] = 0.9
  elif soln.x[1]==-4.:
     soln.x[1] = -3.5
  print(soln.x)

  para0 = soln.x+1e-4*np.random.randn(n_walk, ndim)
  #para0 = np.array([0.3,-2.2])+1e-4*np.random.randn(n_walk, ndim)
  sampler = emcee.EnsembleSampler(n_walk, ndim, log_posterior, args=(PS_mock_mean,Mcovar))
  state=sampler.run_mcmc(para0, Nsteps, progress=True)
  samples = sampler.get_chain()

  #And plot the chains for each parameter
  fsize=16
  fig, axes = plt.subplots(ndim,sharex=True, figsize=(5.,5.))

  for i in range(ndim):
      ax = axes[i]
      ax.plot(samples[:, :, i], "k", alpha=0.3)
      ax.set_xlim(0, len(samples))
      ax.set_ylabel(labels[i],fontsize=fsize)
      ax.yaxis.set_label_coords(-0.07, 0.5)

  axes[-1].set_xlabel("Step number",fontsize=fsize)

  suffix = 'nsub_200Mpc_xHI%.2f_fX%.1f_%s_%dkHz_Smin%.1fmJy_alphaR%.2f_t%dh_dk%.2f_%dkbins_%dsteps' % (xHI_mean_mock[param_i],logfX_mock[param_i],telescope,spec_res,S147,alphaR,tint,d_log_k_bins,len(k_bins_cent),Nsteps)

  plt.savefig('plots/MCMCchains_%s.png' % suffix)
  #plt.show()
  plt.close()

  #See how many steps does it take to burn-in for each parameter
  tau = sampler.get_autocorr_time()
  print('Autocorrelation time for xHI: %d' % tau[0])
  print('Autocorrelation time for fX:  %d' % tau[1])

  para_mean=np.zeros(ndim)

  #Flatten the MCMC
  flat_samples = sampler.get_chain(discard=500, thin=50, flat=True)
  param_label = ['<xHI>','log(fX)']
  
  #Compute the best estimated value and corresponding uncertainty for each parameter
  for j in range(ndim):
          mcmc = np.percentile(flat_samples[:, j], [16, 50, 84])
          q = np.diff(mcmc)
          para_mean[j]=mcmc[1]
          print('%s = %.5f + %.5f - %.5f' % (param_label[j],mcmc[1], q[0], q[1]))
          
  array = np.array([para_mean])
  array = np.concatenate((array,flat_samples))
  #Save posterior samples
  np.save('MCMC_samples/noise_sub/flatsamp_%s.npy' % suffix,array)

  #Present the result in corner plot using corner package (Foreman-Mackey et al. 2023) at https://corner.readthedocs.io/en/latest/
  #fig,axes = plt.subplots(ndim,ndim,sharex=True,figsize=(10.,10.))
  sett=dict(fontsize=14)
  fig=corner.corner(flat_samples,range=[[0.,1.],[min_logfX,max_logfX]],color='royalblue',smooth=True,labels=labels,label_kwargs=sett
                                    ,show_titles=True,title_fmt='.3f',title_kwargs=sett,truths=para_mean,truth_color='fuchsia')

  corner.overplot_points(fig=fig,xs=[[np.nan,np.nan],[np.nan,np.nan],[xHI_mean_mock[param_i],logfX_mock[param_i]],[np.nan,np.nan]],marker='x',markersize=10,markeredgewidth=3,color='darkorange')

  stop_clock = time.perf_counter()
  time_taken = (stop_clock-start_clock)
  print('It took %.3fs of your life' % time_taken)

  plt.savefig('plots/inferred_param_%s.png' % suffix)
  #plt.show()
  plt.close()

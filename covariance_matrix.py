"""
Covariance and correlation matrix computation for the 21-cm forest PS and likelihood according to Hennawi et al. 2021.

Version 6.5.2024

"""

import numpy as np
import scipy.signal as scisi
from astropy.convolution import convolve, Box1DKernel
from scipy import interpolate
from scipy.optimize import minimize
from numpy import random

import numpy.linalg #import inv, det, multi_dot


#Create N_samples samples of median PS signal from N_obs randomly drawn LOS
def multi_obs(signal,N_obs,N_samples):
   
  N_los,N_kbins = signal.shape
  signal_multiobs = np.empty((N_samples,N_kbins))

  for i in range(N_samples):
    LOS = random.randint(0,N_los-1,size=N_obs)
    signal_multiobs[i][:] = np.mean(signal[LOS][:],axis=0)

  return signal_multiobs



#Calculate the covariance matrix
def covar_matrix(signal_mock_ens,signal_sim_ens):
  
  Ndim = signal_sim_ens.shape[1]
  covariance = np.empty([Ndim,Ndim])

  for i in range(Ndim):
    for j in range(Ndim):
        covariance[i,j] = np.median((signal_mock_ens[:][:,i]-signal_sim_ens[:][:,i])*(signal_mock_ens[:][:,j]-signal_sim_ens[:][:,j]))

  return covariance

#Calculate the correllation matrix
def corr_matrix(covariance):
  
  Ndim = len(covariance)
  correlation = np.empty([Ndim,Ndim])

  for i in range(Ndim):

    for j in range(Ndim):
        
        correlation[i,j] = covariance[i][j]/np.sqrt(covariance[i][i]*covariance[j][j])

  return correlation



#Define (negative) likelihood function based on covariance matrix
def log_likelihood(signal_mock_ens,signal_mock_mean,signal_sim_ens,signal_sim_mean):
   
  Ndim = len(signal_mock_mean)
  d_mat = signal_mock_mean-signal_sim_mean
  d_mat_T = np.transpose(d_mat)
  covar_mat = covar_matrix(signal_mock_ens,signal_sim_ens)
  covar_mat_inv = np.linalg.inv(covar_mat)
  covar_mat_det = np.linalg.det(covar_mat)

  #print(d_mat)
  #print(d_mat_T)
  #print(covar_mat)
  #print(covar_mat_det)
  #print(covar_mat_inv)
  #print(np.linalg.multi_dot([covar_mat,covar_mat_inv]))

  return 0.5*np.linalg.multi_dot([d_mat_T,covar_mat_inv,d_mat])+np.log(np.power(2*np.pi,Ndim)*np.absolute(covar_mat_det))

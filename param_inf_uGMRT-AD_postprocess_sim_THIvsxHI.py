"""
Parameter inference from a simulated/forecasted uGMRT observation using the
noise-subtracted 1D power spectrum (Method A2 in Patil et al. 2026).

The inference is performed in the mean neutral hydrogen fraction - mean neutral
IGM temperature parameter space. The mean signal 1D PS and individual signal
PS realizations are interpolated across the parameter space. The latter are
combined with simulated instrumental noise to calculate the parameter-dependent
covariance matrix used in the likelihood.

Version 14.6.2026
"""

#Import necessary packages
import sys
import time
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import minimize
import emcee
import corner

start_clock = time.perf_counter()


#Input parameters
z_name = float(sys.argv[1])         #Redshift of the 21cmFAST LOS data
spec_res = float(sys.argv[2])       #Spectral resolution of the telescope in kHz
telescope = str(sys.argv[3])        #Telescope name
S147 = float(sys.argv[4])           #Source flux density at 147 MHz in mJy
alphaR = float(sys.argv[5])         #Radio spectral index of the source
WLDlvl = int(sys.argv[6])           #Level of wavelet decomposition used for residual bandpass subtraction
f_flag = float(sys.argv[7])         #Threshold for flagging bad channels in units of mean noise rms
rms = float(sys.argv[8])            #Instrumental noise rms per frequency channel in mJy/beam

seed = 370                          #Random seed of the simulated signal dataset
LOS = 511                           #Noise realization used as the mock observation


#Frequency range of the simulated observation
freq_low = 209.                     #Lower frequency limit in MHz
freq_high = 222.                    #Upper frequency limit in MHz

Nlos = 1000                         #Number of simulated noise realizations
n_los = 1000                        #Number of noise realizations used
Nsamples = 1000                     #Number of signal PS samples for each IGM model
Ntargets = 1                        #Number of target sightlines

n_walk = 64                         #Number of MCMC walkers
Nsteps = 10000000                   #Number of MCMC steps

min_xHI = 0.01                      #Minimum mean neutral hydrogen fraction
max_xHI = 0.99                      #Maximum mean neutral hydrogen fraction
min_logTHI = np.log10(15)           #Minimum log10 neutral IGM temperature in K
max_logTHI = np.log10(1e4)          #Maximum log10 neutral IGM temperature in K


#Prepare logarithmic k bins
d_log_k_bins = 0.25
log_k_bins = np.arange(0.,2.751+d_log_k_bins/2.,d_log_k_bins)

k_bins = np.power(10.,log_k_bins)
k_bins_cent = np.power(10.,log_k_bins+d_log_k_bins/2.)[:-1]
Nkbins = len(k_bins_cent)

print('Number of k bins = %d' % Nkbins)
print('k range          = %.2f-%.2f MHz^-1' % (k_bins[0],k_bins[-1]))


#Load the individual signal 1D PS parameter-space dataset
print('Initiate interpolator setup')

datafile = str('datasets/interpolators/individualPS_mask_signal-xHIlogfXTHI_%.1f-%.1fMHz_z%.1f_WLDlvl%d_fflag%.2f_%.1fkHz_Nobs%d_Nkbins%d_Nsamples%d_seed%d.dat' % (freq_low,freq_high,z_name,WLDlvl,f_flag,spec_res,Ntargets,Nkbins,Nsamples,seed))
data = np.fromfile(str(datafile),dtype=np.float32)

Npoints = int(data[0])
xHI_mean = data[1+0*Npoints:1+1*Npoints]
logT_HI = np.log10(data[1+2*Npoints:1+3*Npoints])

#Interpolate the individual signal PS realizations across the xHI-logTHI parameter space
PS_sim_ens = np.reshape(data[1+3*Npoints:],(Npoints,Nsamples,-1))
inter_fun_PS21_sv = interpolate.LinearNDInterpolator(np.transpose([xHI_mean,logT_HI]),PS_sim_ens)


#Load the mean signal 1D PS parameter-space dataset
datafile = str('datasets/interpolators/meanPS_mask_signal-xHIlogfXTHI_%.1f-%.1fMHz_z%.1f_WLDlvl%d_fflag%.2f_%.1fkHz_Nobs%d_Nkbins%d_Nsamples%d_seed%d.dat' % (freq_low,freq_high,z_name,WLDlvl,f_flag,spec_res,Ntargets,Nkbins,Nsamples,seed))
data = np.fromfile(str(datafile),dtype=np.float32)

Npoints = int(data[0])
xHI_mean = data[1+0*Npoints:1+1*Npoints]
logT_HI = np.log10(data[1+2*Npoints:1+3*Npoints])

#Interpolate the mean signal PS across the xHI-logTHI parameter space
PS_sim_ens = np.reshape(data[1+3*Npoints:],(Npoints,-1))
inter_fun_meanPS21 = interpolate.LinearNDInterpolator(np.transpose([xHI_mean,logT_HI]),PS_sim_ens)

print('Interpolator set up')

stop_clock = time.perf_counter()
time_taken = stop_clock-start_clock
print('It took %.3fs of your life' % time_taken)


#Load the simulated instrumental-noise PS realizations
datafile = str('1DPS_dimensionless/1DPS_noise/power_spectrum_noise_mask_21cmFAST_209.0-222.0MHz_z%.1f_WLDlvl%d_fflag%.2f_%s_%.1fkHz_Smin%.1fmJy_alphaR%.2f_rms%.4fmJy_%dLOS.dat' % (z_name,WLDlvl,f_flag,telescope,spec_res,S147,alphaR,rms,Nlos))
data = np.fromfile(str(datafile),dtype=np.float32)

Nlos = int(data[0])
n_kbins = int(data[1])
k_noise_sim = data[2:2+n_kbins]
PS_noise = np.reshape(data[2+n_kbins+0*n_kbins*Nlos:2+n_kbins+1*n_kbins*Nlos],(Nlos,n_kbins))


#Bin the simulated noise PS
PS_noise_bin_sim = np.empty((n_los,len(k_bins_cent)))

for i in range(n_los):
    for l in range(len(k_bins_cent)):
        ind = np.where((k_noise_sim>=k_bins[l]) & (k_noise_sim<k_bins[l+1]))[0]
        PS_noise_bin_sim[i,l] = np.mean(PS_noise[i,ind])


#Use one simulated noise realization as the mock null observation
#The full simulated noise ensemble is retained for the noise mean and covariance
PS_noise_bin = PS_noise_bin_sim
PS_signal = PS_noise_bin_sim[LOS]


#Construct the signal+noise PS ensemble used to calculate the covariance matrix
#The signal ensemble changes with the IGM parameters while the simulated noise ensemble remains fixed
def inter_fun_PS21_nsv(xHI,logTHI):
    return inter_fun_PS21_sv(xHI,logTHI)+PS_noise_bin


#Subtract the mean simulated noise PS following Method A2 in Patil et al. 2026
PS_nsub = PS_signal-np.mean(PS_noise_bin,axis=0)

print('PS_signal obs =',PS_signal)
print('PS_noise  obs =',np.mean(PS_noise_bin,axis=0))
print('PS_nsub   obs =',PS_nsub)


#Define parameters to be estimated
labels = [r"$\langle x_{\rm HI}\rangle$",r"$\langle\mathrm{log}_{10}(\mathrm{T}_{\mathrm{HI}})\rangle$"]


#Define prior distribution as uniform within the range covered by the simulated data
xHI_cut_1 = 0.9295
xHI_cut_2 = 0.99
logTHI_cut_1 = 4
logTHI_cut_2 = 2.7515
slope_cut = (xHI_cut_2-xHI_cut_1)/(logTHI_cut_2-logTHI_cut_1)

def log_prior(theta):
    xHI_mean1, logTHI1 = theta

    if min_xHI <= xHI_mean1 <= max_xHI and min_logTHI <= logTHI1 < logTHI_cut_2:
        return 0
    elif logTHI_cut_2 <= logTHI1 <= max_logTHI and min_xHI <= xHI_mean1 <= xHI_cut_1-slope_cut*(logTHI1-logTHI_cut_1):
        return 0

    return -np.inf


#Define the negative log-likelihood
#Both the mean signal model and its covariance are evaluated at the current IGM parameters
def log_likelihood(theta,obs_signal):
    xHI_mean1, logTHI1 = theta

    #Difference between the noise-subtracted mock observation and current mean signal model
    d_mat = obs_signal-inter_fun_meanPS21(xHI_mean1,logTHI1)
    d_mat_T = np.transpose(d_mat)

    #Parameter-dependent covariance from the signal+simulated-noise PS ensemble
    covar_mat = np.cov(np.transpose(inter_fun_PS21_nsv(xHI_mean1,logTHI1)))
    covar_mat_inv = np.linalg.inv(covar_mat)
    covar_mat_det = np.linalg.det(covar_mat)

    return 0.5*(np.linalg.multi_dot([d_mat_T,covar_mat_inv,d_mat])+np.log(covar_mat_det))


#Define posterior probability using Bayes' theorem
def log_posterior(theta,signal_mock_mean):
    LP = log_prior(theta)

    if not np.isfinite(LP):
        return -np.inf

    return LP-log_likelihood(theta,signal_mock_mean)


#Initiate MCMC parameter estimation
ndim = 2

#Calculate the maximum-likelihood solution as a diagnostic
#The MCMC walkers below are initialized independently of this solution
initial = np.array([0.3,1.5])
soln = minimize(log_likelihood,initial,args=(PS_nsub,),bounds=([min_xHI,max_xHI],[min_logTHI,max_logTHI]))

if soln.x[0] == 0.01:
    soln.x[0] = 0.05

if soln.x[1] == np.log10(14):
    soln.x[1] = np.log10(20)
elif soln.x[1] == 4.:
    soln.x[1] = 3.95

print(soln.x)


#Initialize MCMC walkers across the parameter space
para0 = np.random.rand(n_walk,ndim)
para0[:,1] = min_logTHI+(max_logTHI-min_logTHI)*para0[:,1]


#Create a multiprocessing pool
#Multiprocessing is disabled below because it slows down the MCMC on the system used for this analysis.
#Set pool=pool in EnsembleSampler to enable parallel sampling where beneficial, e.g. on an HPC system.
ncpu = multiprocessing.cpu_count()
print('%d CPU cores available' % ncpu)

with multiprocessing.Pool(processes=2) as pool:
    sampler = emcee.EnsembleSampler(n_walk,ndim,log_posterior,args=(PS_nsub,),pool=None)
    state = sampler.run_mcmc(para0,Nsteps,progress=True)
    samples = sampler.get_chain()


#Plot the MCMC chains
fsize = 16
fig, axes = plt.subplots(ndim,sharex=True,figsize=(5.,5.))

for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:,:,i],"k",alpha=0.3)
    ax.set_xlim(0,len(samples))
    ax.set_ylabel(labels[i],fontsize=fsize)
    ax.yaxis.set_label_coords(-0.07,0.5)

axes[-1].set_xlabel("Step number",fontsize=fsize)

suffix = 'uGMRT-AD_logTHI-xHI_sim_rms%.2f_postprocess_flagmock_mask_PSOJ352−15_WLDlvl%d_fflag%.2f_%.1f-%.1fMHz_%.1fkHz_%s_dk%.2f_%dkbins_%dw_%dsteps_LOS%d' % (rms,WLDlvl,f_flag,freq_low,freq_high,spec_res,telescope,d_log_k_bins,len(k_bins_cent),n_walk,Nsteps,LOS)

plt.savefig('uGMRT_archivaldata/MCMC_samples/MCMCchains_%s.png' % suffix)
plt.close()


#Flatten the MCMC using the burn-in and thinning adopted for the analysis in the paper
flat_samples = sampler.get_chain(discard=500,thin=50,flat=True)

param_label = ['<xHI>','<logTHI>']
para_mean = np.zeros(ndim)


#Compute the median inferred values and 16th/84th percentile uncertainties
for j in range(ndim):
    mcmc = np.percentile(flat_samples[:,j],[16,50,84])
    q = np.diff(mcmc)
    para_mean[j] = mcmc[1]

    print('%s = %.5f + %.5f - %.5f' % (param_label[j],mcmc[1],q[1],q[0]))


#Save the posterior samples together with the median inferred parameters
array = np.array([para_mean])
array = np.concatenate((array,flat_samples))

np.save('MCMC_samples/uGMRT_archivaldata/flatsamp_%s.npy' % suffix,array)


#Plot the posterior distributions
sett = dict(fontsize=14)

fig = corner.corner(flat_samples,range=[[0.,1.],[min_logTHI,max_logTHI]],color='royalblue',
                    smooth=True,labels=labels,label_kwargs=sett,show_titles=True,
                    title_fmt='.3f',title_kwargs=sett,truths=para_mean,truth_color='fuchsia')


stop_clock = time.perf_counter()
time_taken = stop_clock-start_clock
print('It took %.3fs of your life' % time_taken)

plt.savefig('uGMRT_archivaldata/MCMC_samples/inferred_param_%s.png' % suffix)
plt.close()
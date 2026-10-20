"""
Preparing the 21-cm forest 1D power spectrum datasets used for interpolation
and Bayesian parameter inference.

The code reads the full set of 21cmFAST models across the parameter space,
calculates the mean temperature of the neutral IGM for each model, and reads
the corresponding forward-modelled 21-cm forest signal 1D power spectra.
The power spectra are rebinned into the k bins used for parameter inference.

Models excluded from the final parameter space are removed before saving.
Both the individual-LOS power spectra and the mean power spectrum of each
IGM model are saved for subsequent interpolation and parameter inference.

Version 14.5.2026
"""

#Import necessary packages
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt


#Input parameters
z_name = float(sys.argv[1])         #Redshift of the 21cmFAST LOS data
spec_res = float(sys.argv[2])       #Spectral resolution of the telescope in kHz
WLDlvl = int(sys.argv[3])           #Level of wavelet decomposition used for residual bandpass subtraction
f_flag = float(sys.argv[4])         #Threshold for flagging bad channels in units of mean off-source noise rms

path_LOS = 'datasets/21cmFAST_los/los/' #Path to the full 21cmFAST parameter-space dataset

#Frequency range of the flux spectrum
freq_low = 209.                     #Lower frequency limit in MHz
freq_high = 222.                    #Upper frequency limit in MHz

n_los = 20                          #Number of signal LOS used for each IGM model
Nsamples = 1000                     #Number of individual PS samples saved for each IGM model
Ntargets = 1                        #Number of observed target sightlines


#Prepare logarithmic k bins
d_log_k_bins = 0.25                 #Logarithmic k-bin width
log_k_bins = np.arange(0.,2.751+d_log_k_bins/2.,d_log_k_bins)

k_bins = np.power(10.,log_k_bins)   #Edges of logarithmic k bins
k_bins_cent = np.power(10.,log_k_bins+d_log_k_bins/2.)[:-1] #Centers of logarithmic k bins
Nkbins = len(k_bins_cent)           #Number of k bins
print('Number of k bins = %d' % Nkbins)
print('k range          = %.2f-%.2f MHz^-1' % (k_bins[0],k_bins[-1]))


#Find the simulated IGM models used to construct the interpolator dataset
print('Initiate interpolator setup')

#Find all of the datasets for the interpolation
files = glob.glob(path_LOS+'*.dat')
print('Number of files before selection = %d' % len(files))

#Ignore the IGM models with no X-ray heating
files_to_remove = glob.glob(path_LOS+'*fXnone*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])
print('Number of files after removing fXnone models = %d' % len(files))

#Ignore the IGM models generated with different random seeds
files_to_remove = glob.glob(path_LOS+'*seed*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])
print('Number of files after removing additional seeds = %d' % len(files))


#Prepare arrays containing the IGM parameters
logfX = np.empty(len(files))        #log10 of X-ray pre-heating efficiency
xHI_mean = np.empty(len(files))     #Mean neutral hydrogen fraction
T_HI = np.empty(len(files))         #Mean temperature of neutral gas in K

done_perc = 0.1                     #Progress-printing interval
PS_sim_mean = np.empty((len(logfX),len(k_bins_cent)))              #Mean signal PS for each IGM model
PS_signal_ens = np.empty((len(logfX),Nsamples,len(k_bins_cent)))   #Individual-LOS signal PS ensemble


#Read the IGM parameters and signal 1D PS data for each simulated model
for j in range(len(files)):
   
   #Read the 21cmFAST simulation
   data = np.fromfile(str(files[j]),dtype=np.float32)

   Nbins = int(data[7])             #Number of pixels/cells/bins in one line-of-sight
   Nlos = int(data[8])              #Number of lines-of-sight
   x_initial = 12                   #Index where LOS data arrays start

   logfX[j] = data[9]               #log10 of X-ray pre-heating efficiency
   xHI_mean[j] = data[11]           #Mean neutral hydrogen fraction

   #Calculate the mean temperature of highly neutral gas
   xHI = data[(x_initial+2*Nbins+1*Nlos*Nbins):(x_initial+2*Nbins+2*Nlos*Nbins)]
   TK = data[(x_initial+2*Nbins+2*Nlos*Nbins):(x_initial+2*Nbins+3*Nlos*Nbins)]
   ind_neu = np.where(xHI>=0.9)[0]  #Select gas with neutral fraction >= 0.9
   T_HI[j] = np.mean(TK[ind_neu])   #Mean temperature of the highly neutral IGM

   #Read the forward-modelled signal 1D PS ensemble
   datafile = str('1DPS_dimensionless/1DPS_signal/power_spectrum_signal_mask_21cmFAST_%.1f-%.1fMHz_z%.1f_WLDlvl%d_fflag%.2f_fX%.2f_xHI%.2f_%.1fkHz_%dLOS.dat' % (freq_low,freq_high,z_name,WLDlvl,f_flag,logfX[j],xHI_mean[j],spec_res,n_los))
   data = np.fromfile(str(datafile),dtype=np.float32)
   Nlos = int(data[0])              #Number of LOS stored in the PS file
   n_kbins = int(data[1])           #Number of unbinned k modes
   k = data[2:2+n_kbins]            #Frequency wavenumbers in MHz^-1
   PS_signal = data[2+n_kbins+0*n_kbins*Nlos:2+n_kbins+1*n_kbins*Nlos]
   PS_signal = np.reshape(PS_signal,(Nlos,n_kbins))[:n_los,:]

   #Bin the signal 1D PS data
   PS_signal_bin = np.empty((n_los,len(k_bins_cent)))

   for i in range(n_los):
      for l in range(len(k_bins_cent)):
         ind = np.where((k>=k_bins[l]) & (k<k_bins[l+1]))[0]
         PS_signal_bin[i,l] = np.mean(PS_signal[i,ind])

   #Store the individual-LOS and mean signal 1D PS
   PS_signal_ens[j,:,:] = PS_signal_bin[:,:]
   PS_sim_mean[j,:] = np.mean(PS_signal_bin,axis=0)

   #Track the progress
   done_files = (j+1)/len(files)
   if done_files>=done_perc:
         print('Done %.2f' % done_files)
         done_perc = done_perc+0.1


#Remove models for which the neutral IGM temperature is not defined
ind_i = np.where(T_HI>0.)[0]
xHI_mean = xHI_mean[ind_i]
logfX = logfX[ind_i]
T_HI = T_HI[ind_i]
PS_signal_ens = PS_signal_ens[ind_i]
PS_sim_mean = PS_sim_mean[ind_i]

print('Number of models with valid T_HI = %d' % len(xHI_mean))


#Remove selected models from the final interpolation parameter space
ind_i = np.where((logfX==-1) & (0.109<xHI_mean) & (xHI_mean<0.146))[0]
ind_i = np.append(ind_i,np.where((logfX==-1) & (0.795<xHI_mean) & (xHI_mean<0.8))[0])
ind_i = np.append(ind_i,np.where((logfX==-1) & (0.8016<xHI_mean) & (xHI_mean<0.802))[0])
ind_i = np.append(ind_i,np.where((logfX==-3) & (0.1085<xHI_mean) & (xHI_mean<0.14))[0])
ind_i = np.append(ind_i,np.where((logfX==-3) & (0.15<xHI_mean) & (xHI_mean<0.2))[0])
ind_i = np.append(ind_i,np.where((logfX==-3) & (0.794<xHI_mean) & (xHI_mean<0.795))[0])
ind_i = np.append(ind_i,np.where((logfX==-3) & (0.8<xHI_mean) & (xHI_mean<0.81))[0])
ind_i = np.append(ind_i,np.where((logfX==-2) & (0.5<xHI_mean) & (xHI_mean<0.52))[0])
ind_i = np.append(ind_i,np.where((logfX==-2) & (0.522<xHI_mean) & (xHI_mean<0.54))[0])
ind_i = np.append(ind_i,np.where((logfX==-2) & (0.56<xHI_mean) & (xHI_mean<0.57))[0])
ind_i = np.append(ind_i,np.where((logfX==-3.6) & (0.24<xHI_mean) & (xHI_mean<0.25))[0])
ind_i = np.append(ind_i,np.where((logfX==-3.6) & (0.255<xHI_mean) & (xHI_mean<0.27))[0])
ind_i = np.append(ind_i,np.where((logfX==-3.6) & (0.31<xHI_mean) & (xHI_mean<0.32))[0])
ind_i = np.append(ind_i,np.where((logfX==-3.6) & (0.5<xHI_mean) & (xHI_mean<0.52))[0])
ind_i = np.append(ind_i,np.where((logfX==-3.6) & (0.53<xHI_mean) & (xHI_mean<0.54))[0])

xHI_mean = np.delete(xHI_mean,ind_i)
logfX = np.delete(logfX,ind_i)
T_HI = np.delete(T_HI,ind_i)
PS_signal_ens = np.delete(PS_signal_ens,ind_i,axis=0)
PS_sim_mean = np.delete(PS_sim_mean,ind_i,axis=0)

print('Number of models in final parameter space = %d' % len(xHI_mean))


#Plot the final IGM parameter space
fig = plt.figure(figsize=(5.,5.))

plt.scatter(xHI_mean, np.log10(T_HI), c=logfX)
plt.xlabel(r"$\langle x_{\rm HI} \rangle$")
plt.ylabel(r"$\log_{10}(T_{\rm HI}/{\rm K})$")
plt.colorbar(label=r"$\log_{10} f_X$")

plt.tight_layout()
plt.savefig('plots/parameterspace_logTHIvsxHIvslogfX_MCMC.png')
plt.show()
plt.close()

sys.exit()
#Save the individual-LOS signal PS dataset for parameter inference
array = np.array([len(xHI_mean)])
array = np.append(array,xHI_mean)
array = np.append(array,logfX)
array = np.append(array,T_HI)
array = np.append(array,PS_signal_ens)
array.astype('float32').tofile('interpolators/individualPS_mask_signal-xHIlogfXTHI_%.1f-%.1fMHz_z%.1f_WLDlvl%d_fflag%.2f_%.1fkHz_Nobs%d_Nkbins%d_Nsamples%d.dat' % (freq_low,freq_high,z_name,WLDlvl,f_flag,spec_res,Ntargets,Nkbins,Nsamples),sep='')


#Save the mean signal PS dataset used for interpolation
array = np.array([len(xHI_mean)])
array = np.append(array,xHI_mean)
array = np.append(array,logfX)
array = np.append(array,T_HI)
array = np.append(array,PS_sim_mean)
array.astype('float32').tofile('interpolators/meanPS_mask_signal-xHIlogfXTHI_%.1f-%.1fMHz_z%.1f_WLDlvl%d_fflag%.2f_%.1fkHz_Nobs%d_Nkbins%d_Nsamples%d.dat' % (freq_low,freq_high,z_name,WLDlvl,f_flag,spec_res,Ntargets,Nkbins,Nsamples),sep='')
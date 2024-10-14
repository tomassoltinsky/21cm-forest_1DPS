"""
Creating the power spectrum plot for different fX and xHI while fixing the other parameter.

Version 9.2.2024

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
from scipy import interpolate

fsize = 24

#Input parameters
z_name = float(sys.argv[1])   #redshift
dvH = float(sys.argv[2])      #rebinning width in km/s
spec_res = float(sys.argv[3]) #spectral resolution of telescope (i.e. frequency channel width) in kHz
n_los = 1000

fX_fid = float(sys.argv[4])   #fiducial log(f_X)
xHI_fid = float(sys.argv[5])  #fiducial <x_HI>

#Find all of the datasets for the interpolation
path_LOS = 'datasets/21cmFAST_los/los/'
files = glob.glob(path_LOS+'*.dat')

#Remove files if needed
files_to_remove = glob.glob(path_LOS+'*fXnone*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])

logfX = np.empty(len(files))
xHI_mean = np.empty(len(files))

#Generate k bins
d_log_k_bins = 0.25
log_k_bins = np.arange(-1.0-d_log_k_bins/2.,3.+d_log_k_bins/2.,d_log_k_bins)
log_k_bins = log_k_bins[1:]

k_bins = np.power(10.,log_k_bins)
k_bins_cent = np.power(10.,log_k_bins+d_log_k_bins/2.)[:-1]
PS_signal_sim = np.empty((len(files),len(k_bins_cent)))


#Load and prepare fiducial data
datafile = str('1DPS_dimensionless/1DPS_signal/power_spectrum_signal_21cmFAST_200Mpc_z%.1f_fX%.2f_xHI%.2f_%dkHz_%dLOS.dat' % (z_name,fX_fid,xHI_fid,spec_res,1000))
data = np.fromfile(str(datafile),dtype=np.float32)
Nlos = int(data[0])
n_kbins = int(data[1])
k = data[2:2+n_kbins]
PS_signal = data[2+n_kbins+0*n_kbins*Nlos:2+n_kbins+1*n_kbins*Nlos]
PS_signal = np.reshape(PS_signal,(Nlos,n_kbins))[:n_los,:]

PS_signal_bin = np.empty((n_los,len(k_bins_cent)))

for i in range(n_los):
  for l in range(len(k_bins_cent)):
    ind = np.where((k>=k_bins[l]) & (k<k_bins[l+1]))[0]
    PS_signal_bin[i,l] = np.mean(PS_signal[i,ind])

PS_signal_fid = np.median(PS_signal_bin,axis=0)

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

    #Take median for each k bin
    PS_signal_sim[j,:] = np.median(PS_signal_bin,axis=0)

    #Track the progress
    done_files = (j+1)/len(files)
    if done_files>=done_perc:
      print('Done %.2f' % done_files)
      done_perc = done_perc+0.1

#Set up interpolator
inter_fun_PS21 = interpolate.LinearNDInterpolator(np.transpose([xHI_mean,logfX]),PS_signal_sim)


#Plot varying x_HI
xHI_interpol = np.arange(0.01,0.99,0.0002)
colours = plt.cm.viridis(np.linspace(0,1,len(xHI_interpol)))

fig, (ax0, cbar_ax) = plt.subplots(ncols=2,figsize=(10.,5.),gridspec_kw={'width_ratios': [9.75,0.25]})

cmap = plt.cm.viridis_r
norm = plt.Normalize(vmin=0., vmax=1.)
cb1 = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
cbar_ax.tick_params(labelsize=fsize)
cb1.set_label(r'$\langle x_{\rm HI}\rangle$',fontsize=fsize)

for i in range(len(xHI_interpol)):
   
   PS_signal_inter = inter_fun_PS21(xHI_interpol[i], fX_fid)
   ax0.plot(k_bins_cent,PS_signal_inter,'-',color=cmap(norm(xHI_interpol[i])),label=r'Signal')

ax0.set_ylim(1e-9,1e-4)
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

ax0.text(3.5,4e-9,r'$\mathrm{log}_{10}(f_{\mathrm{X}})=%.1f$' % (fX_fid),fontsize=fsize-3)

plt.tight_layout()
plt.subplots_adjust(hspace=2.0)
plt.savefig('1DPS_plots/power_spectrum_dimless_interpol_vsxHI_21cmFAST_200Mpc_z%.1f_fX%.2f_xHI%.2f_nomock.pdf' % (z_name,fX_fid,xHI_fid))
plt.show()



#Plot varying log(f_X)
logfX_interpol = np.arange(-4.,1.0001,0.001)
colours = plt.cm.inferno(np.linspace(0,1,len(logfX_interpol)))

fig, (ax0, cbar_ax) = plt.subplots(ncols=2,figsize=(10.,5.),gridspec_kw={'width_ratios': [9.75,0.25]})

cmap = plt.cm.inferno
norm = plt.Normalize(vmin=np.amin(logfX_interpol), vmax=np.amax(logfX_interpol))
cb1 = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
cbar_ax.tick_params(labelsize=fsize)
cb1.set_label(r'$\mathrm{log}(f_{\rm X})$',fontsize=fsize)

for i in range(len(logfX_interpol)):
   
   PS_signal_inter = inter_fun_PS21(xHI_fid, logfX_interpol[i])
   ax0.plot(k_bins_cent,PS_signal_inter,'-',color=cmap(norm(logfX_interpol[i])),label=r'Signal')

ax0.text(3.5,1e-14,r'$\langle x_{\mathrm{HI}}\rangle=%.2f$' % (xHI_fid),fontsize=fsize-3)

ax0.set_ylim(5e-16,3e-5)
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
plt.subplots_adjust(hspace=2.0)
plt.savefig('1DPS_plots/power_spectrum_dimless_interpol_vsfX_21cmFAST_200Mpc_z%.1f_fX%.2f_xHI%.2f_nomock.pdf' % (z_name,fX_fid,xHI_fid))
plt.show()
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from matplotlib.ticker import AutoMinorLocator
import scipy.signal as scisi
from astropy.convolution import convolve, Box1DKernel
from matplotlib import gridspec
import scipy.stats

import constants
import instrumental_features

path = '../../datasets/21cmFAST_los/'
z_name = float(sys.argv[1])
dvH = float(sys.argv[2])
spec_res = float(sys.argv[3])
telescope = str(sys.argv[4])
Nd = int(sys.argv[5])
S_min_QSO = float(sys.argv[6])
alpha_R = float(sys.argv[7])
t_int = float(sys.argv[8])
fX_name = float(sys.argv[9])
xHI_mean = float(sys.argv[10])
LOS = int(sys.argv[11])
n_los = 1000


dF = 0.002
dFmax = 0.05
bins = np.arange(1-dFmax,1+dFmax+0.001,dF)
bins_centre = (bins-dF/2*0)[:-1]

Nlos = 1000

datafile = str('%slos/los_50Mpc_256_n1000_z%.3f_fX%.1f_xHI%.2f.dat' % (path,z_name,fX_name,xHI_mean))
data  = np.fromfile(str(datafile),dtype=np.float32)
z        = data[0]	#redshift
omega_0  = data[1]	#matter density
omega_L  = data[2]	#vacuum density
omega_b  = data[3]	#baryon density
h        = data[4]	#small h (Hubble parameter)
box_size = data[5]*4	#box size in Mpc
X_H      = data[6]	#primordial hydrogen fraction
H_0      = 1e7*h/constants.Mpc	#Hubble constant in s^-1
rho_c    = 3*np.power(H_0,2.)/(8*np.pi*constants.G)	#critical density in g cm^-3
omega_k  = 1-omega_0-omega_L			#curvature
H        = H_0*np.sqrt(omega_L+omega_0*(1+z)**3+omega_k*(1+z)**2)
dz       = 1*box_size/(1.+z)/h*constants.Mpc/1000.*H*(1.+z)/constants.c
print(box_size,dz)

datafile = str('%stau_long/los_200Mpc_n%d_z%.3f_dv%d.dat' %(path,Nlos,z_name,dvH))
data  = np.fromfile(str(datafile),dtype=np.float32)
z     = data[0]	#redshift
box_size = data[1]/1000
Nbins = int(data[2])					#Number of pixels/cells/bins in one line-of-sight
Nlos = int(data[3])						#Number of lines-of-sight
x_initial = 4
vel_axis = data[(x_initial+Nbins):(x_initial+2*Nbins)]#Hubble velocity along LoS in km/s
freq  = instrumental_features.freq_obs(z,vel_axis*1.e5)
redsh = instrumental_features.z_obs(z,vel_axis*1e5)

datafile = str('%stau_long/tau_200Mpc_n%d_z%.3f_fX%.1f_xHI%.2f_dv%d.dat' %(path,Nlos,z_name,fX_name,xHI_mean,dvH))
data = np.fromfile(str(datafile),dtype=np.float32)
tau = np.reshape(data,(Nlos,Nbins))[LOS]

signal_ori = instrumental_features.transF(tau)
freq_smooth,F_signal = instrumental_features.smooth_fixedbox(freq,signal_ori,spec_res)
bandwidth = (freq_smooth[-1]-freq_smooth[0])/1e6
print('Bandwidth = %.2fMHz' % bandwidth)

F_noise = instrumental_features.add_noise(freq_smooth,telescope,spec_res,S_min_QSO,alpha_R,t_int,Nd,showsigN=True)
F_observed = F_signal+F_noise 
F_noise = 1+F_noise

hist_signal = plt.hist(F_signal,bins)[0][:]/dz
plt.close()
hist_obs = plt.hist(F_observed,bins)[0][:]/dz
plt.close()
hist_noise = plt.hist(F_noise,bins)[0][:]/dz
plt.close()

'''
PDF_signal_med = np.mean(hist_signal,axis=1)
PDF_signal_16  = np.percentile(hist_signal,16,axis=1)
PDF_signal_84  = np.percentile(hist_signal,84,axis=1)
PDF_noise_med = np.mean(hist_noise,axis=1)
PDF_noise_16  = np.percentile(hist_noise,16,axis=1)
PDF_noise_84  = np.percentile(hist_noise,84,axis=1)
PDF_obs_med = np.mean(hist_obs,axis=1)
PDF_obs_16  = np.percentile(hist_obs,16,axis=1)
PDF_obs_84  = np.percentile(hist_obs,84,axis=1)
'''

fsize = 22
fsize_leg = 16
cmap = ['black','fuchsia','darkorange']
style = ['-',':','--']

fig = plt.figure(figsize=(5,5))
gs = gridspec.GridSpec(1,1)

ax00 = plt.subplot(gs[0])

w = np.full(len(F_observed),1/dz)
ax00.hist(F_observed,bins,weights=w,histtype='step',align='mid',color='royalblue',label='Signal+Noise')
ax00.hist(F_noise,bins,weights=w,histtype='step',align='mid',color='fuchsia',label='Noise')
ax00.hist(F_signal,bins,weights=w,histtype='step',align='mid',color='darkorange',label='Signal')
ax00.legend(frameon=False,loc='upper left',fontsize=fsize-8)

KS = scipy.stats.ks_2samp(F_observed,F_noise)
print('KS p-value=%.20f' % KS[1])
ax00.text(1.015,460,r'$\mathrm{%s}$' % telescope,fontsize=fsize-8)
ax00.text(1.015,430,r'$t_{\rm int}=%d\,\rm hr$' % t_int,fontsize=fsize-8)
ax00.text(1.015,400,r'$p = %.3f$' % KS[1],fontsize=fsize-8)
#ax00.plot(bins_centre,hist_obs,'-',color='royalblue',label=r'Observed')
#ax00.plot(bins_centre,hist_noise,'-',color='fuchsia',label=r'Noise')
#ax00.plot(bins_centre,hist_signal,'-',color='darkorange',label=r'Signal')
#ax00.fill_between(bins_centre,PDF_obs_16,PDF_obs_84,alpha=0.25,color='royalblue')
#ax00.fill_between(bins_centre,PDF_noise_16,PDF_noise_84,alpha=0.25,color='fuchsia')
#x00.fill_between(bins_centre,PDF_signal_16,PDF_signal_84,alpha=0.25,color='darkorange')

ax00.set_xlim(1-dFmax,1+dFmax)
ax00.set_ylim(0.,500.)
#ax00.set_xscale('log')
#ax00.set_yscale('log')
ax00.xaxis.set_minor_locator(AutoMinorLocator())
ax00.yaxis.set_minor_locator(AutoMinorLocator())
ax00.set_xlabel(r'$F_{21}=e^{-\tau_{21}}$', fontsize=fsize)
ax00.set_ylabel(r'$\partial^2N/\partial z\partial F_{21}$', fontsize=fsize)
ax00.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True,length=5,width=1)    
ax00.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True,length=10,width=1,labelsize=fsize)
ax00.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True,length=10,width=1,labelsize=fsize)

plt.tight_layout()
#plt.subplots_adjust(right=0.975,hspace=.0,wspace=.0)
plt.savefig('PDF/PDF_fluxes_200Mpc_z%.1f_fX%.2f_xHI%.2f_%s_%dkHz_Smin%.1fmJy_alphaR%.2f_t%dh_%dLOS.png' % (z,fX_name,xHI_mean,telescope,spec_res,S_min_QSO,alpha_R,t_int,LOS))
plt.show()
plt.close()
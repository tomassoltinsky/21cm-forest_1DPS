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
import time

start_clock = time.perf_counter()

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
n_los = 1000


dF = 0.001
dFmax = 0.02
bins = np.arange(1-dFmax-dF,1+dFmax+dF+0.0001,dF)
bins[int(len(bins)/2)] = 1.00000001
bins_centre = bins[:-1]+dF/2

#print(bins)
#print(bins_centre)

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
tau = np.reshape(data,(Nlos,Nbins))

hist_signal = np.empty((n_los,len(bins)-1))
hist_obs = np.empty((n_los,len(bins)-1))
hist_noise = np.empty((n_los,len(bins)-1))
KS = np.empty(n_los)

done_perc = 0.1

for i in range(n_los):

    signal_ori = instrumental_features.transF(tau[i])
    freq_smooth,F_signal = instrumental_features.smooth_fixedbox(freq,signal_ori,spec_res)
    bandwidth = (freq_smooth[-1]-freq_smooth[0])/1e6
    #print('Bandwidth = %.2fMHz' % bandwidth)

    F_noise = instrumental_features.add_noise(freq_smooth,telescope,spec_res,S_min_QSO,alpha_R,t_int,Nd,showsigN=False)
    F_observed = F_signal+F_noise 
    F_noise = 1+F_noise

    hist_signal[i] = plt.hist(F_signal,bins)[0][:]/dz/dF
    plt.close()
    hist_obs[i] = plt.hist(F_observed,bins)[0][:]/dz/dF
    plt.close()
    hist_noise[i] = plt.hist(F_noise,bins)[0][:]/dz/dF
    plt.close()

    KS[i] = scipy.stats.ks_2samp(F_observed,F_noise)[1]

    done_LOS = (i+1)/n_los
    if done_LOS>=done_perc:
      print('Done %.2f' % done_LOS)
      done_perc = done_perc+0.1

stop_clock = time.perf_counter()
time_taken = (stop_clock-start_clock)
print('It took %.3fs to run' % time_taken)

print('KS p-value=%.6f+-%6f' % (np.mean(KS),np.std(KS)))

PDF_signal_med = np.mean(hist_signal,axis=0)
PDF_signal_16  = np.percentile(hist_signal,16,axis=0)
PDF_signal_84  = np.percentile(hist_signal,84,axis=0)
PDF_noise_med = np.mean(hist_noise,axis=0)
PDF_noise_16  = np.percentile(hist_noise,16,axis=0)
PDF_noise_84  = np.percentile(hist_noise,84,axis=0)
PDF_obs_med = np.mean(hist_obs,axis=0)
PDF_obs_16  = np.percentile(hist_obs,16,axis=0)
PDF_obs_84  = np.percentile(hist_obs,84,axis=0)
    

fsize = 20
fsize_leg = 14
cmap = ['black','fuchsia','darkorange']
style = ['-',':','--']

fig = plt.figure(figsize=(5,5))
gs = gridspec.GridSpec(1,1)

ax00 = plt.subplot(gs[0])

#w = np.full(len(F_observed),1/dz)
#ax00.hist(F_observed,bins,weights=w,histtype='step',align='mid',color='royalblue',label='Signal+Noise')
#ax00.hist(F_noise,bins,weights=w,histtype='step',align='mid',color='fuchsia',label='Noise')
#ax00.hist(F_signal,bins,weights=w,histtype='step',align='mid',color='darkorange',label='Signal')
#ax00.legend(frameon=False,loc='upper left',fontsize=fsize-8)

ax00.set_title(r'$\mathrm{%s},\ t_{\rm int}=%d\,\rm hr$' % (telescope,t_int),fontsize=fsize)
#ax00.text(1.0075,7.9e5,r'$t_{\rm int}=%d\,\rm hr$' % t_int,fontsize=fsize_leg)

if np.std(KS)<=np.mean(KS):
    ax00.text(0.9825,8.e5,r'$p = %.3f\pm%.3f$' % (np.mean(KS),np.std(KS)),fontsize=fsize_leg)
else:
    ax00.text(0.9825,8.e5,r'$p = %.3f^{+%.3f}_{-%.3f}$' % (np.mean(KS),np.std(KS),np.mean(KS)),fontsize=fsize_leg)

non_detections = len(np.where(KS>0.05)[0])
print('Non-detections = %d' % non_detections)

ax00.step(bins_centre,PDF_signal_med,where='mid',color='darkorange',label=r'Signal')
ax00.step(bins_centre,PDF_noise_med,where='mid',color='fuchsia',label=r'Noise')
ax00.step(bins_centre,PDF_obs_med,where='mid',color='royalblue',label=r'Signal+Noise')
ax00.legend(frameon=False,loc=[0.05,0.625],fontsize=fsize_leg)
ax00.fill_between(bins_centre,PDF_obs_16,PDF_obs_84,step='mid',alpha=0.25,color='royalblue')
ax00.fill_between(bins_centre,PDF_noise_16,PDF_noise_84,step='mid',alpha=0.25,color='fuchsia')
ax00.fill_between(bins_centre,PDF_signal_16,PDF_signal_84,step='mid',alpha=0.25,color='darkorange')
'''
def gaussian(x, mu, sig):
    return (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )
ax00.plot(bins_centre,gaussian(bins_centre,1,0.1192)/dz,color='black')
'''
ax00.set_xlim(1-dFmax,1+dFmax)
ax00.set_ylim(0.,9.e5)
#ax00.set_xscale('log')
#ax00.set_yscale('log')
ax00.xaxis.set_minor_locator(AutoMinorLocator())
ax00.yaxis.set_minor_locator(AutoMinorLocator())
ax00.set_xlabel(r'$F_{21}=e^{-\tau_{21}}$', fontsize=fsize)
ax00.set_ylabel(r'$\partial^2N/\partial z\partial F_{21}$', fontsize=fsize)
ax00.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
ax00.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True,length=7.5,width=1.5)    
ax00.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True,length=15,width=1.5,labelsize=fsize)
ax00.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True,length=15,width=1.5,labelsize=fsize)

plt.tight_layout()
#plt.subplots_adjust(right=0.975,hspace=.0,wspace=.0)
plt.savefig('PDF/PDF_fluxes_200Mpc_z%.1f_fX%.2f_xHI%.2f_%s_%dkHz_Smin%.1fmJy_alphaR%.2f_t%dh_allLOS.pdf' % (z,fX_name,xHI_mean,telescope,spec_res,S_min_QSO,alpha_R,t_int))
plt.show()
plt.close()
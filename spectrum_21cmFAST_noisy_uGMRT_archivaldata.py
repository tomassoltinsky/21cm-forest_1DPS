'''
Code to generate and plot a forward-modelled synthetic noisy 21-cm forest spectrum
corresponding to the uGMRT archival observations of PSO J352-15.

The 21cmFAST signal is smoothed to the observational spectral resolution and
Gaussian instrumental noise corresponding to the measured uGMRT rms is added.
The instrumental noise rms in flux-density units is assumed to be frequency
independent, rather than calculated from the frequency-dependent telescope
sensitivity as in the previous forecasts. The fractional noise relative to the
QSO continuum remains frequency dependent due to the radio spectral index.

Version 21.6.2026
'''

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal as scisi
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
import matplotlib.patches as patches
import openpyxl

import constants
import instrumental_features



#Input parameters
path = 'datasets/21cmFAST_los/'          #Path to the simulated 21cmFAST LOS data
z_name = float(sys.argv[1])              #Redshift of the LOS data
dvH = float(sys.argv[2])                 #Velocity resolution of the LOS data in m/s
spec_res = float(sys.argv[3])            #Spectral resolution in kHz
telescope = str(sys.argv[4])             #Telescope name
S_min_QSO = float(sys.argv[5])           #Intrinsic QSO flux density at 147 MHz in mJy
alpha_R = float(sys.argv[6])             #Radio spectral index of the QSO
fX_name = float(sys.argv[7])             #X-ray pre-heating efficiency fX
xHI_mean = float(sys.argv[8])            #Mean neutral hydrogen fraction
LOS = int(sys.argv[9])                    #Index of the simulated LOS
rms = float(sys.argv[10])                 #Frequency-independent instrumental noise rms in mJy/beam

Nlos = 20                                 #Number of LOS in the simulation file

#Read the 21cmFAST LOS data
datafile = str('%stau_long/los_200Mpc_n%d_z%.3f_dv%d.dat' %(path,Nlos,z_name,dvH))
data  = np.fromfile(str(datafile),dtype=np.float32)
z     = data[0]                           #Redshift
box_size = data[1]/1000                   #Simulation box size in cMpc
Nbins = int(data[2])                      #Number of pixels/cells/bins in one line-of-sight
Nlos = int(data[3])                       #Number of lines-of-sight
sigma_8 = 0.829                           #Amplitude of matter fluctuations
omega_0 = 0.308                           #Matter energy density
omega_L = 0.692                           #Vacuum energy density
omega_b = 0.0482                          #Baryon energy density
n_s = 0.961                               #Scalar spectral index
h = 0.678                                 #Reduced Hubble parameter
X_H = 0.76                                #Hydrogen mass fraction
YHe = 0.24                                #Helium mass fraction
x_initial = 4                             #Index where LOS data arrays start
vel_axis = data[(x_initial+Nbins):(x_initial+2*Nbins)] #Hubble velocity along LOS in km/s
freq  = instrumental_features.freq_obs(z,vel_axis*1.e5) #Observed frequency
redsh = instrumental_features.z_obs(z,vel_axis*1e5)     #Redshift along the LOS

#Read the 21-cm optical depth for the selected IGM model and LOS
datafile = str('%stau_long/tau_200Mpc_n%d_z%.3f_fX%.1f_xHI%.2f_dv%d.dat' %(path,Nlos,z_name,fX_name,xHI_mean,dvH))
data = np.fromfile(str(datafile),dtype=np.float32)
tau = np.reshape(data,(Nlos,Nbins))[LOS]  #21-cm optical depth along the selected LOS

#Convert optical depth to transmitted flux and smooth to the observational spectral resolution
signal_ori = instrumental_features.transF(tau) #Intrinsic transmitted 21-cm forest signal
freq_uni,signal_uni = instrumental_features.uni_freq(freq,np.array([signal_ori]))
freq_smooth,signal_smooth = instrumental_features.smooth_fixedbox(freq_uni,signal_uni[0],spec_res,shownpix=True)
bandwidth = (freq_smooth[-1]-freq_smooth[0])/1e6 #Bandwidth of the smoothed simulated LOS in MHz
print('Bandwidth = %.2fMHz' % bandwidth)


#Prepare the frequency range corresponding to the uGMRT observations
bandwidth = 25                            #Bandwidth in MHz
z_T = 5.82                                #Redshift of the target J352-15
nu_T = constants.c/constants.lambda_0/(1.+z_T)/1.e6 #Observed 21-cm frequency at the target redshift in MHz
print("nu_T = %.6f MHz" % nu_T)
freq_range = np.arange(0,bandwidth/2.*1.0000000000001,spec_res*1e-3)
freq_smooth = np.concatenate((nu_T - freq_range[::-1], nu_T + freq_range[1:]))*1e6

freq_smooth = np.arange(205,222.5001,spec_res*1e-3)*1e6 #Frequency range of the archival data in Hz
T_ind = int(freq_smooth.shape[0]*(222.5-nu_T)/(222.5-205)) #Number of channels blueward of the target

#Add continuum redward of the target and place the simulated forest blueward of the target
signal_smooth = np.concatenate((np.full(freq_smooth.shape[0]-T_ind,1.0),np.flip(signal_smooth[:T_ind])))

#Calculate the QSO continuum and fractional instrumental noise across the band
S_v = S_min_QSO*np.power(freq_smooth/147.e6,alpha_R) #QSO flux density as a function of frequency in mJy
sigma_noise = rms/S_v                                #Fractional noise relative to the QSO continuum

#Generate Gaussian instrumental noise and add it to the simulated spectrum
np.random.seed(370)                                  #Random seed for reproducibility
noise = np.random.normal(0.,sigma_noise,len(signal_smooth)) #Gaussian fractional instrumental noise
signal_noisy = signal_smooth+noise                   #Forward-modelled signal+noise spectrum

#Plot the mock noisy 21-cm forest spectrum
fsize = 24                                           #Plot font size
fig = plt.figure(figsize=(10.,5.))
gs = gridspec.GridSpec(1,1)

v_min = freq_smooth[-1]/1e6                          #Maximum plotted frequency in MHz
v_max = freq_smooth[0]/1e6                           #Minimum plotted frequency in MHz

ax4 = plt.subplot(gs[0,0])  

ax4.plot([0,1],[0,0],'--',color='darkorange',label='Signal')
ax4.plot([0,1],[0,0],color='black',label='Signal+Noise')
plt.legend(frameon=False,loc='upper right',fontsize=fsize-8,ncol=2)

ax4.plot(freq_smooth/1e6,signal_noisy-1,color='black',label='Signal+Noise')
ax4.plot(freq_smooth/1e6,signal_smooth-1,'--',color='darkorange',label='Signal')
ax4.set_xlim(v_max,v_min)
ax4.set_ylim(-0.2,0.2)
ax4.set_yticks(np.arange(-0.2,0.21,0.1))
ax4.set_xlabel(r'$\nu_{\rm obs}\ [\rm MHz]$', fontsize=fsize)
ax4.set_ylabel(r'Residual $F_{21}=e^{-\tau_{21}}$', fontsize=fsize)
ax4.set_xticks(np.arange(208,222,4))
ax4.xaxis.set_minor_locator(AutoMinorLocator())
ax4.yaxis.set_minor_locator(AutoMinorLocator())
ax4.tick_params(axis='x',which='major',direction='in',bottom=True,top=False,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax4.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax4.tick_params(axis='both',which='minor',direction='in',bottom=True,top=False,left=True,right=True
		,length=10,width=2)

#Calculate the comoving distance corresponding to the observational bandwidth
Hz = 100*1e5*h*np.sqrt(omega_0*(1+z_T)**3+omega_L)   #Hubble parameter at the target redshift
bandwidth = freq_smooth[-1]/1e6 - freq_smooth[0]/1e6 #Observational bandwidth in MHz
box_size = constants.c*(1+z_T)/Hz * bandwidth/nu_T   #Comoving length of the observational bandwidth in cMpc
print('Target redshift       = %.2f' % z_T)
print('Target frequency      = %.2f MHz' % nu_T)
print('Hubble parameter      = %.2f km/s/Mpc' % (Hz/1e5))
print('Observational bandwidth = %.2f MHz' % bandwidth)
print('Comoving box size     = %.2f cMpc' % box_size)
print('Frequency range       = %.2f-%.2f MHz' % (v_max,v_min))
HztoMPC = constants.c*(1+z_T)/Hz/nu_T                #Conversion between frequency separation and comoving distance

#Upper horizontal axis in terms of comoving distance
ax0up = ax4.secondary_xaxis('top',
                            functions=(lambda R: HztoMPC*(R-nu_T),
                                      lambda k: k/HztoMPC + nu_T))

ax0up.set_xlabel(r'$x\ [\rm cMpc]$', fontsize=fsize)
ax0up.xaxis.set_minor_locator(AutoMinorLocator())
ax0up.tick_params(axis='x',which='major',direction='in',bottom=False,top=True,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax0up.tick_params(axis='x',which='minor',direction='in',bottom=False,top=True,left=True,right=True
		,length=10,width=2,labelsize=fsize)

#Indicate the target frequency and regions blueward/redward of the QSO
ax4.axvline(x=nu_T,color='fuchsia',linestyle=':')
ax4.axvspan(nu_T,209,alpha=0.25,color='fuchsia')
ax4.axvspan(205,nu_T,alpha=0.25,color='red')

plt.tight_layout()
plt.subplots_adjust(hspace=.0)
plt.savefig('plots/Soltinsky_2026/noisy_spectrum_uGMRT_archivaldata_%.1f-%.1f_z%.1f_fX%.2f_xHI%.2f_%s_%dkHz_Smin%.1fmJy_alphaR%.2f_rms%.4fmJy_%dLOS.png' % (freq_smooth[0]/1e6,freq_smooth[-1]/1e6,z,fX_name,xHI_mean,telescope,spec_res,S_min_QSO,alpha_R,rms,LOS))
plt.show()
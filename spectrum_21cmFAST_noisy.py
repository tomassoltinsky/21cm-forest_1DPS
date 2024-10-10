'''
Generating a plot of 21-cm forest signal spectrum without and with telescope noise.

Version 10.10.2024
'''

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec

import instrumental_features


#Input parameters
path = 'datasets/21cmFAST_los/'
z_name = float(sys.argv[1])		#redshift
dvH = float(sys.argv[2])		#rebinning width in km/s
spec_res = float(sys.argv[3])	#spectral resolution of telescope (i.e. frequency channel width) in kHz
telescope = str(sys.argv[4])	#telescope name duh
Nd = int(sys.argv[5])			#number dishes used by telescope
S_min_QSO = float(sys.argv[6])	#minimum intrinsic flux of QSO at 147Hz in mJy
alpha_R = float(sys.argv[7])	#radio spectral index of QSO
t_int = float(sys.argv[8])		#integration time of observation in hours
fX_name = float(sys.argv[9])	#log10(f_X)
xHI_mean = float(sys.argv[10])	#mean neutral hydrogen fraction
LOS = int(sys.argv[11])			#LOS ID number
Nlos = 10


#Load data for x-axis
datafile = str('%stau_long/los_200Mpc_n%d_z%.3f_dv%d.dat' %(path,Nlos,z_name,dvH))
data  = np.fromfile(str(datafile),dtype=np.float32)
z     = data[0]			#redshift
box_size = data[1]/1000	#box size in cMpc
Nbins = int(data[2])	#Number of pixels/cells/bins in one line-of-sight
Nlos = int(data[3])		#Number of lines-of-sight
x_initial = 4
vel_axis = data[(x_initial+Nbins):(x_initial+2*Nbins)]	#Hubble velocity along LoS in km/s
freq  = instrumental_features.freq_obs(z,vel_axis*1.e5)	#observed frequency in Hz
redsh = instrumental_features.z_obs(z,vel_axis*1e5)		#observed redshift

#Load data for y-axis (optical depth)
datafile = str('%stau_long/tau_200Mpc_n%d_z%.3f_fX%.1f_xHI%.2f_dv%d.dat' %(path,Nlos,z_name,fX_name,xHI_mean,dvH))
data = np.fromfile(str(datafile),dtype=np.float32)
tau = np.reshape(data,(Nlos,Nbins))[LOS]

#Smooth the spectrum to model the instrumental resolution
signal_ori = instrumental_features.transF(tau)
freq_smooth,signal_smooth = instrumental_features.smooth_fixedbox(freq,signal_ori,spec_res)
bandwidth = (freq_smooth[-1]-freq_smooth[0])/1e6
print('Bandwidth = %.2fMHz' % bandwidth)

#Add noise to the spectrum
signal_noisy = signal_smooth+instrumental_features.add_noise(freq_smooth,telescope,spec_res,S_min_QSO,alpha_R,t_int,Nd,showsigN=True)


#Plot the spectrum
fsize = 24
fig = plt.figure(figsize=(10.,5.))
gs = gridspec.GridSpec(1,1)

v_min = freq_smooth[-1]/1e6
v_max = freq_smooth[0]/1e6

ax4 = plt.subplot(gs[0,0])  

ax4.plot([0,1],[0,0],'--',color='darkorange',label='Signal')
ax4.plot([0,1],[0,0],color='black',label='Signal+Noise')
plt.legend(frameon=False,loc='upper left',fontsize=fsize-3,ncols=2)

ax4.plot(freq_smooth/1e6,signal_noisy,color='black',label='Signal+Noise')
ax4.plot(freq_smooth/1e6,signal_smooth,'--',color='darkorange',label='Signal')
ax4.set_xlim(v_max,v_min)
ax4.set_ylim(0.96,1.025)
ax4.set_yticks(np.arange(0.96,1.03,0.02))
ax4.set_xlabel(r'$\nu_{\rm obs}\ [\rm MHz]$', fontsize=fsize)
ax4.set_ylabel(r'$F_{21}=e^{-\tau_{21}}$', fontsize=fsize)
ax4.xaxis.set_minor_locator(AutoMinorLocator())
ax4.yaxis.set_minor_locator(AutoMinorLocator())
ax4.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax4.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax4.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=2)

R_label = np.arange(0,int(box_size)+1,50)
R_posit = (R_label)/R_label[-1]*(-v_max+v_min)
ax0up = ax4.twiny()
ax0up.set_xlabel(r'$x\ [\rm cMpc]$', fontsize=fsize)
ax0up.set_xticks(R_posit)
ax0up.set_xticklabels(R_label)
ax0up.xaxis.set_minor_locator(AutoMinorLocator())
ax0up.tick_params(axis='x',which='major',direction='in',bottom=False,top=True,left=True,right=True
		,length=20,width=2,labelsize=fsize)
ax0up.tick_params(axis='x',which='minor',direction='in',bottom=False,top=True,left=True,right=True
		,length=10,width=2,labelsize=fsize)

plt.tight_layout()
plt.subplots_adjust(hspace=.0)
plt.show()
plt.savefig('plots/noisy_spectrum_200Mpc_z%.1f_fX%.2f_xHI%.2f_%s_%dkHz_Smin%.1fmJy_alphaR%.2f_t%dh_%dLOS.png' % (z,fX_name,xHI_mean,telescope,spec_res,S_min_QSO,alpha_R,t_int,LOS))
plt.show()
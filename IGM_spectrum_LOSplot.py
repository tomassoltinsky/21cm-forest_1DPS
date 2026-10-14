#Code for plotting the LOS of IGM fields and 21-cm forest flux spectrum.
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal as scisi
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
import matplotlib.patches as patches
import constants
import instrumental_features

fsize = 16
path = str(sys.argv[1])				#path to the data folder
z_name = float(sys.argv[2])			#redshift
dvH = float(sys.argv[3])			#Velocity resolution in km/s
fX_name = float(sys.argv[4])		#X-ray heating efficiency
xHI_mean_name = float(sys.argv[5])	#Mean neutral fraction
filenum = int(sys.argv[6])			#File number to be plotted
LOS = int(sys.argv[7])				#Line-of-sight number to be plotted

Nlos = 10
ngrid = 256
Lbox = 50

#Read IGM fields data
datafile = str('%slos_regrid/los_%dMpc_n%d_z%.3f_fX%.1f_xHI%.2f_dv%d_file%d.dat' % (path,Lbox,Nlos,z_name,fX_name,xHI_mean_name,dvH,filenum))
data = np.fromfile(str(datafile),dtype=np.float32)

#cosmological parameters
z = data[0]		  	#redshift
omega_0 = data[1]	#matter density
omega_L = data[2]	#vacuum density
omega_b = data[3]	#baryon density
h = data[4]		    #small h (Hubble parameter)
box_size = data[5]  #box size in Mpc
X_H = data[6]	    #primordial hydrogen fraction
logfX = data[9]     #X-ray efficiency of galaxies (log10(f_X))
ioneff = data[10]   #Ionizing efficiency of galaxies (zeta)
xHI_mean = data[11] #mean neutral hydrogen fraction

print('ION_eff=%.2f, <x_HI>=%.5f, log(f_X)=%.2f' % (ioneff,xHI_mean,logfX))

#line-of-sight information
Nbins = int(data[7])	#number of pixels/cells/bins in one line-of-sight
Nlos = int(data[8]) 	#number of lines-of-sight
x_initial = 12

#read axis information
pos_axis = data[(x_initial):(x_initial+Nbins)]	#position along LoS in kpc
vel_axis = data[(x_initial+Nbins):(x_initial+2*Nbins)]#Hubble velocity along LoS in km/s
freq  = instrumental_features.freq_obs(z,vel_axis*1.e5)	#observed frequency in Hz
redsh = instrumental_features.z_obs(z,vel_axis*1e5)		#observed redshift

#extract normalized density (rho/rho_mean), neutral fraction, temperature of HI, peculiar velocity of HI for each pixel in each LoS
Delta = np.reshape(data[(x_initial+2*Nbins):(x_initial+2*Nbins+Nlos*Nbins)], (Nlos, Nbins))[LOS]
xHI = np.reshape(data[(x_initial+2*Nbins+1*Nlos*Nbins):(x_initial+2*Nbins+2*Nlos*Nbins)], (Nlos, Nbins))[LOS]
TK = np.reshape(data[(x_initial+2*Nbins+2*Nlos*Nbins):(x_initial+2*Nbins+3*Nlos*Nbins)], (Nlos, Nbins))[LOS]
vpec = np.reshape(data[(x_initial+2*Nbins+3*Nlos*Nbins):(x_initial+2*Nbins+4*Nlos*Nbins)], (Nlos, Nbins))[LOS]

#Read optical depth data
datafile = str('%stau/tau_%dMpc_n%d_z%.3f_fX%.1f_xHI%.2f_dv%d_file%d.dat' %(path,Lbox,Nlos,z_name,fX_name,xHI_mean_name,dvH,filenum))
data = np.fromfile(str(datafile),dtype=np.float32)
tau = np.reshape(data,(Nlos,Nbins))[LOS]


#Plot the LOS of IGM fields and 21-cm forest flux spectrum
fig = plt.figure(figsize=(12.,20.))
gs = gridspec.GridSpec(5,1)
z_min = redsh[-1]
z_max = redsh[0]

#Overdensity
ax0 = plt.subplot(gs[0,0])

ax0.plot(redsh,Delta,color='black')
#ax0.set_ylim(0.97,1.004)
#ax0.set_xlabel(r'$z$', fontsize=fsize)
ax0.set_yscale('log')
ax0.set_xlim(z_min,z_max)
ax0.set_ylabel(r'$\Delta$', fontsize=fsize)
ax0.xaxis.set_minor_locator(AutoMinorLocator())
ax0.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=0)
ax0.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=5,width=1)


#Peculiar velocity
ax1 = plt.subplot(gs[1,0], sharex = ax0)

ax1.plot(redsh,vpec,color='black')
#ax1.set_ylim(0.97,1.004)
#ax1.set_xlabel(r'$z$', fontsize=fsize)
ax1.set_ylabel(r'$v_{\rm pec}\ [kmh^{-1}]$', fontsize=fsize)
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=0)
ax1.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax1.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=5,width=1)


#Neutral hydrogen fraction
ax2 = plt.subplot(gs[2,0], sharex = ax0)

ax2.plot(redsh,xHI,color='black')
ax2.set_ylim(-0.05,1.05)
#ax2.set_xlabel(r'$z$', fontsize=fsize)
ax2.set_ylabel(r'$x_{\rm HI}$', fontsize=fsize)
ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())
ax2.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=0)
ax2.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax2.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=5,width=1)


#Temperature
ax3 = plt.subplot(gs[3,0], sharex = ax0)

ax3.plot(redsh,TK,color='black',label=r'$T_{\rm K}$')
ax3.set_ylim(5e0,5e5)
ax3.set_yscale('log')
ax3.set_ylabel(r'$T\ [\rm K]$', fontsize=fsize)
ax3.xaxis.set_minor_locator(AutoMinorLocator())
ax3.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=0)
ax3.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax3.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=5,width=1)


#Normalized flux
ax4 = plt.subplot(gs[4,0], sharex = ax0)

ax4.plot(redsh,instrumental_features.transF(tau),color='black')
ax4.set_ylim(0.85,1.01)
ax4.set_xlabel(r'$z$', fontsize=fsize)
ax4.set_ylabel(r'$F=e^{-\tau_{21}}$', fontsize=fsize)
ax4.xaxis.set_minor_locator(AutoMinorLocator())
ax4.yaxis.set_minor_locator(AutoMinorLocator())
ax4.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax4.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax4.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=5,width=1)


#Top axis in frequency
freq_min = np.ceil(constants.c/constants.lambda_0/1.e6/(1.+z_max))
freq_max = np.floor(constants.c/constants.lambda_0/1.e6/(1.+z_min))
freq_label = np.arange(freq_min,freq_max+0.01,1.)
freq_posit = (constants.c/constants.lambda_0/1.e6/freq_label-1.-z_min)/(z_max-z_min)
ax0up = ax0.twiny()
ax0up.set_xlabel(r'$\nu_{obs}\ [\rm MHz]$', fontsize=fsize)
ax0up.set_xticks(freq_posit)
ax0up.set_xticklabels(freq_label)
ax0up.tick_params(axis='x',which='major',direction='in',bottom=False,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)

ax1up = ax1.twiny()
ax1up.set_xticks(freq_posit)
ax1up.set_xticklabels(freq_label)
ax1up.tick_params(axis='x',which='major',direction='in',bottom=False,top=True,left=True,right=True
		,length=10,width=1,labelsize=0)

ax2up = ax2.twiny()
ax2up.set_xticks(freq_posit)
ax2up.set_xticklabels(freq_label)
ax2up.tick_params(axis='x',which='major',direction='in',bottom=False,top=True,left=True,right=True
		,length=10,width=1,labelsize=0)

ax3up = ax3.twiny()
ax3up.set_xticks(freq_posit)
ax3up.set_xticklabels(freq_label)
ax3up.tick_params(axis='x',which='major',direction='in',bottom=False,top=True,left=True,right=True
		,length=10,width=1,labelsize=0)

ax4up = ax4.twiny()
ax4up.set_xticks(freq_posit)
ax4up.set_xticklabels(freq_label)
ax4up.tick_params(axis='x',which='major',direction='in',bottom=False,top=True,left=True,right=True
		,length=10,width=1,labelsize=0)

freq_min = np.ceil(constants.c/constants.lambda_0/1.e5/(1.+z_max))/10
freq_max = np.floor(constants.c/constants.lambda_0/1.e5/(1.+z_min))/10
freq_label = np.arange(freq_min,freq_max+0.01,0.1)
freq_posit = (constants.c/constants.lambda_0/1.e6/freq_label-1.-z_min)/(z_max-z_min)

ax0upmin = ax0.twiny()
ax0upmin.set_xticks(freq_posit)
ax0upmin.tick_params(axis='x',which='major',direction='in',bottom=False,top=True,left=True,right=True
		,length=5,width=1,labeltop=False)

ax1upmin = ax1.twiny()
ax1upmin.set_xticks(freq_posit)
ax1upmin.tick_params(axis='x',which='major',direction='in',bottom=False,top=True,left=True,right=True
		,length=5,width=1,labeltop=False)

ax2upmin = ax2.twiny()
ax2upmin.set_xticks(freq_posit)
ax2upmin.tick_params(axis='x',which='major',direction='in',bottom=False,top=True,left=True,right=True
		,length=5,width=1,labeltop=False)

ax3upmin = ax3.twiny()
ax3upmin.set_xticks(freq_posit)
ax3upmin.tick_params(axis='x',which='major',direction='in',bottom=False,top=True,left=True,right=True
		,length=5,width=1,labeltop=False)

ax4upmin = ax4.twiny()
ax4upmin.set_xticks(freq_posit)
ax4upmin.tick_params(axis='x',which='major',direction='in',bottom=False,top=True,left=True,right=True
		,length=5,width=1,labeltop=False)



plt.tight_layout()
plt.subplots_adjust(hspace=.0)
plt.savefig('plots/IGMfields_spectrum_z%.1f_logfX%.1f_xHI%.2f_dv%d_file%d_LOS%d.png' % (z_name,fX_name,xHI_mean_name,dvH,filenum,LOS))
plt.show()
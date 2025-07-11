'''
Plot 2D posterior maps as a function of <xHI> and logfX for multiple combinations of these parameters.

Version 10.10.2024
'''

<<<<<<< HEAD
=======
import sys
import glob, os
>>>>>>> 8059d39cec5b600d8c2ba24fa6c96901f4c01ddd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
import corner

path = 'MCMC_samples'
<<<<<<< HEAD
spec_res = 8         #kHz
S147 = 64.2          #mJy
alphaR = -0.44       #radio spectral index
Nsteps = 100000      #number of MCMC steps
autocorr_cut = 10000 #cut-off for autocorrelation

Nkbins = 6           #number of k-bins
d_log_k_bins = 0.25  #logarithmic bin width

#Select observational setup
=======
spec_res = 8
S147 = 64.2
alphaR = -0.44
Nsteps = 100000
autocorr_cut = 10000

Nkbins = 6
d_log_k_bins = 0.25

>>>>>>> 8059d39cec5b600d8c2ba24fa6c96901f4c01ddd
telescope = 'SKA1-low'
tint = 50
xHI_mean = [0.25,0.25,0.25,0.52,0.52,0.80,0.80,0.80,0.79]
logfX    = [-1.0,-2.0,-3.0,-1.2,-3.0,-1.0,-2.0,-3.0,-0.6]
'''
telescope = 'uGMRT'
tint = 500
xHI_mean = [0.25,0.25,0.25,0.52,0.52,0.80,0.80,0.80,0.79]
logfX    = [-1.0,-2.0,-3.0,-1.6,-3.0,-1.2,-2.0,-3.0,-0.6]

telescope = 'uGMRT'
tint = 50
#xHI_mean = [0.25,0.25,0.25,0.39,0.52,0.80,0.80,0.80]
#logfX    = [-1.0,-2.0,-3.2,-3.8,-3.2,-3.2,-2.0,-1.0]
'''

<<<<<<< HEAD
#Plotting
=======
>>>>>>> 8059d39cec5b600d8c2ba24fa6c96901f4c01ddd
fsize = 20
colours  = ['royalblue','fuchsia','forestgreen','darkorange','red','lightcoral','slateblue','limegreen','teal']

fig = plt.figure(figsize=(5.,5.))
gs = gridspec.GridSpec(1,1)
ax0 = plt.subplot(gs[0,0])

contkwarg = dict(alpha=[0.,0.25,0.5])

for i in range(len(logfX)):
    data = np.load('%s/flatsamp_200Mpc_xHI%.2f_fX%.1f_%s_%dkHz_Smin%.1fmJy_alphaR%.2f_t%dh_dk%.2f_%dkbins_%dsteps.npy' % (path,xHI_mean[i],logfX[i],telescope,spec_res,S147,alphaR,tint,d_log_k_bins,Nkbins,Nsteps))
    corner.hist2d(data[autocorr_cut:,0],data[autocorr_cut:,1],levels=[1-np.exp(-0.5),1-np.exp(-2.)],plot_datapoints=False,plot_density=False,fill_contours=True,color=colours[i])#,contourf_kwargs=contkwarg)
<<<<<<< HEAD
=======
    #corner.hist2d(data[:,0],data[:,1],levels=[1-np.exp(-0.5),1-np.exp(-2.)],smooth=True,plot_datapoints=True,plot_density=True,color=colours[i])
>>>>>>> 8059d39cec5b600d8c2ba24fa6c96901f4c01ddd

for i in range(len(logfX)):
    ax0.scatter(xHI_mean[i],logfX[i],marker='X',s=200,linewidths=1.,color=colours[i],edgecolors='black',alpha=1)

ax0.set_xticks(np.arange(0.,1.1,0.2))
ax0.set_yticks(np.arange(-4.,0.1,1.))
ax0.set_xlim(0.,1.)
ax0.set_ylim(-4,-0.4)
ax0.xaxis.set_minor_locator(AutoMinorLocator())
ax0.yaxis.set_minor_locator(AutoMinorLocator())
ax0.set_xlabel(r'$\langle x_{\rm HI}\rangle$', fontsize=fsize)
ax0.set_ylabel(r'$\log_{10}(f_{\mathrm{X}})$', fontsize=fsize)
ax0.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=5,width=1)

colours_lit = ['grey','brown','darkviolet','navy']
rot = 55

<<<<<<< HEAD
#Literature values for x_HI
=======
>>>>>>> 8059d39cec5b600d8c2ba24fa6c96901f4c01ddd
ax0.axvspan(0.21-0.07,0.21+0.17,alpha=0.2,color=colours_lit[0])
ax0.text(0.15,-1.5,'Ďurovčíková+24',color=colours_lit[0],rotation=rot,fontsize=12)

ax0.axvspan(0.17-0.11,0.17+0.09,alpha=0.2,color=colours_lit[1])
ax0.text(0.08,-1.5,'Gaikwad+23',color=colours_lit[1],rotation=rot,fontsize=12)

ax0.axvspan(0,0.21,alpha=0.2,color=colours_lit[2])
ax0.text(0.005,-1.5,'Greig+24',color=colours_lit[2],rotation=rot,fontsize=12)

<<<<<<< HEAD
#Compute logfX for literature values of T_S from the 21-cm power spectrum from tomographic observations
=======
>>>>>>> 8059d39cec5b600d8c2ba24fa6c96901f4c01ddd
from scipy import interpolate
import glob, os

path_LOS = '../../datasets/21cmFAST_los/los/'
z_name = 6.000

<<<<<<< HEAD
#Read data of T_S, x_HI and logfX to setup interpolator
=======
>>>>>>> 8059d39cec5b600d8c2ba24fa6c96901f4c01ddd
files = glob.glob(path_LOS+'*.dat')
files_to_remove = glob.glob(path_LOS+'*fXnone*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])
files_to_remove = glob.glob(path_LOS+'*fX1.0*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])
files_to_remove = glob.glob(path_LOS+'*fX0.*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])
<<<<<<< HEAD

logfX = np.empty(len(files))
xHI_mean = np.empty(len(files))
TK_mean = np.empty(len(files))
=======
'''
files_to_remove = glob.glob(path_LOS+'*fX0.8*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])
files_to_remove = glob.glob(path_LOS+'*fX0.6*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])
'''
logfX = np.empty(len(files))
xHI_mean = np.empty(len(files))
>>>>>>> 8059d39cec5b600d8c2ba24fa6c96901f4c01ddd
T_HI = np.empty(len(files))

for j in range(len(files)):

  data  = np.fromfile(str(files[j]),dtype=np.float32)
<<<<<<< HEAD
  Nbins = int(data[7])	#Number of pixels/cells/bins in one line-of-sight
  Nlos = int(data[8])	#Number of lines-of-sight
=======
  Nbins = int(data[7])					#Number of pixels/cells/bins in one line-of-sight
  Nlos = int(data[8])						#Number of lines-of-sight
>>>>>>> 8059d39cec5b600d8c2ba24fa6c96901f4c01ddd
  x_initial = 12
  
  logfX[j]    = data[9]
  xHI_mean[j] = data[11]

  xHI     = data[(x_initial+2*Nbins+1*Nlos*Nbins):(x_initial+2*Nbins+2*Nlos*Nbins)]
  TK      = data[(x_initial+2*Nbins+2*Nlos*Nbins):(x_initial+2*Nbins+3*Nlos*Nbins)]
  ind_neu = np.where(xHI>=0.9)[0]
  T_HI[j] = np.mean(TK[ind_neu])

<<<<<<< HEAD
#Save the array of properties of the IGM into file for later use
array = np.array([len(files)])
array = np.append(array,xHI_mean)
array = np.append(array,logfX)
array = np.append(array,TK_mean)
array = np.append(array,T_HI)
array.astype('float32').tofile('datasets/interpolators/IGMprop_200Mpc_z%.1f.dat' % (z_name),sep='')

#Read the data for the interpolator
datafile = str('datasets/interpolators/IGMprop_200Mpc_z%.1f.dat' % (z_name))
data = np.fromfile(str(datafile),dtype=np.float32)
Nmodels = int(data[0])
xHI_mean = data[1:(Nmodels+1)]
logfX = data[(Nmodels+1):(2*Nmodels+1)]
TK = data[(2*Nmodels+1):(3*Nmodels+1)]
T_HI = data[(3*Nmodels+1):(4*Nmodels+1)]

#Setup the interpolator
=======
>>>>>>> 8059d39cec5b600d8c2ba24fa6c96901f4c01ddd
inter_fun_fx = interpolate.LinearNDInterpolator(np.transpose([xHI_mean,T_HI]),logfX)

print('T_HI=%.2f-%.2f' % (np.amin(T_HI),np.amax(T_HI)))
print('x_HI=%.4f-%.4f' % (np.amin(xHI_mean),np.amax(xHI_mean)))

xHI = np.arange(0.01,0.999,0.01)

<<<<<<< HEAD
#Find limits for logfX for literature values of T_S
=======
>>>>>>> 8059d39cec5b600d8c2ba24fa6c96901f4c01ddd
T_HI = 15.6
logfX_down = inter_fun_fx(xHI,T_HI)

T_HI = 656.7
logfX_up = inter_fun_fx(xHI,T_HI)

<<<<<<< HEAD
#Plot literature range of values for logfX
ax0.fill_between(xHI,logfX_down,logfX_up,alpha=0.2,color=colours_lit[3])
ax0.text(0.75,-0.75,'HERA+23',color=colours_lit[3],fontsize=12)

#Read the data for the null-detection limits
datafile = str('interpolators/nulldetection_limits_200Mpc_z%.1f_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f_Nobs%d_Nkbins%d.dat' % (z_name,telescope,spec_res,tint,S147,alphaR,10,Nkbins))
data = np.fromfile(str(datafile),dtype=np.float32)
xHI_lim_68 = np.reshape(data,(4,-1))[0]
logfX_lim_68 = np.reshape(data,(4,-1))[1]

#Plot the 68% confidence limits
ax0.plot(xHI_lim_68,logfX_lim_68,linestyle='-',color='black',linewidth=1.5)

plt.tight_layout()
plt.savefig('plots/multiparam_infer_%dNkbins_%s_%dhr_withobs_moreparams_test.png' % (Nkbins,telescope,tint))
=======
ax0.fill_between(xHI,logfX_down,logfX_up,alpha=0.2,color=colours_lit[3])
ax0.text(0.75,-0.75,'HERA+23',color=colours_lit[3],fontsize=12)

plt.tight_layout()
plt.savefig('MCMC_samples/multiparam_infer_%dNkbins_%s_%dhr_withobs_moreparams_test.png' % (Nkbins,telescope,tint))
>>>>>>> 8059d39cec5b600d8c2ba24fa6c96901f4c01ddd
plt.show()

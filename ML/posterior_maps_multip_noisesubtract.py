'''
Plot 2D posterior maps as a function of <xHI> and logfX for multiple combinations of these parameters.

Version 29.05.2024
'''

import sys
import glob, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
import corner

path = 'MCMC_samples'
spec_res = 8
S147 = 64.2
alphaR = -0.44
Nsteps = 100000
autocorr_cut = 1000

Nkbins = 6
d_log_k_bins = 0.25
'''
telescope = 'SKA1-low'
tint = 50
#xHI_mean = [0.25,0.25,0.52,0.52,0.80,0.80,0.80]
#logfX    = [-2.0,-3.0,-1.2,-3.0,-1.,-2.0,-3.0]
xHI_mean = [0.25,0.25,0.25,0.52,0.52,0.80,0.80,0.80]
logfX    = [-1.0,-2.0,-3.0,-1.2,-3.0,-1.0,-2.0,-3.0]
'''
'''
telescope = 'uGMRT'
tint = 500
#xHI_mean = [0.25,0.25,0.52,0.52,0.80,0.80,0.80]
#logfX    = [-2.0,-3.0,-1.6,-3.0,-1.2,-2.0,-3.0]
xHI_mean = [0.25,0.25,0.25,0.52,0.80,0.80,0.80]
logfX    = [-1.0,-2.0,-3.0,-3.0,-1.2,-2.0,-3.0]

'''
'''
telescope = 'uGMRT'
tint = 50
xHI_mean = [0.39,0.52,0.80]
logfX    = [-3.8,-3.2,-3.2]
xHI_mean = [0.25,0.25,0.25,0.52,0.80,0.80,0.80]
logfX    = [-1.0,-2.0,-3.2,-3.2,-3.2,-2.0,-1.0]
'''

telescope = 'uGMRT'
tint = 500
xHI_mean = [0.11,0.80,0.52,0.11,0.80]
logfX    = [-1.0,-1.0,-2.0,-3.0,-3.0]

print('With noise')

fsize = 20
colours  = ['royalblue','fuchsia','forestgreen','darkorange','red','lightcoral','slateblue','limegreen','teal','navy']

fig = plt.figure(figsize=(5.,5.))
gs = gridspec.GridSpec(1,1)
ax0 = plt.subplot(gs[0,0])

contkwarg = dict(alpha=[0.,0.25,0.5])
logfX_infer = np.empty(len(logfX))
xHI_infer = np.empty(len(logfX))
G_all = np.empty(len(logfX))

for i in range(len(logfX)):
    data = np.load('%s/flatsamp_200Mpc_xHI%.2f_fX%.1f_%s_%dkHz_Smin%.1fmJy_alphaR%.2f_t%dh_dk%.2f_%dkbins_%dsteps.npy' % (path,xHI_mean[i],logfX[i],telescope,spec_res,S147,alphaR,tint,d_log_k_bins,Nkbins,Nsteps))
    corner.hist2d(data[autocorr_cut:,0],data[autocorr_cut:,1],levels=[1-np.exp(-0.5),1-np.exp(-2.)],plot_datapoints=False,plot_density=False,fill_contours=True,color=colours[i])#,contourf_kwargs=contkwarg)
    #corner.hist2d(data[:,0],data[:,1],levels=[1-np.exp(-0.5),1-np.exp(-2.)],smooth=True,plot_datapoints=True,plot_density=True,color=colours[i])

    logfX_infer[i] = data[0,1]
    xHI_infer[i] = data[0,0]

    G_all[i] = np.mean((data[autocorr_cut:,0]-xHI_mean[i])**2+(data[autocorr_cut:,1]-logfX[i])**2)

for i in range(len(logfX)):
    ax0.scatter(xHI_infer[i],logfX_infer[i],marker='o',s=200,linewidths=1.,color=colours[i],edgecolors='black',alpha=1)
    ax0.scatter(xHI_mean[i],logfX[i],marker='*',s=200,linewidths=1.,color=colours[i],edgecolors='black',alpha=1)

print('Mock')
print(xHI_mean)
print(logfX)
print('Inferred')
print(xHI_infer)
print(logfX_infer)
G_5 = np.sqrt(np.mean((xHI_infer-xHI_mean)**2+(logfX_infer-logfX)**2))
print('G_5   = %.6f' % G_5)
G_all = np.sqrt(np.mean(G_all))
print('G_all = %.6f' % G_all)

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
rot = 60

ax0.axvspan(0.21-0.07,0.21+0.17,alpha=0.2,color=colours_lit[0])
ax0.text(0.15,-1.80,'Ďurovčíková+24',color=colours_lit[0],rotation=rot,fontsize=12)

ax0.axvspan(0.17-0.11,0.17+0.09,alpha=0.2,color=colours_lit[1])
ax0.text(0.08,-1.80,'Gaikwad+23',color=colours_lit[1],rotation=rot,fontsize=12)

ax0.axvspan(0,0.21,alpha=0.2,color=colours_lit[2])
ax0.text(0.005,-1.80,'Greig+24',color=colours_lit[2],rotation=rot,fontsize=12)

from scipy import interpolate
import glob, os
z_name = 6.000
'''
path_LOS = '../../datasets/21cmFAST_los/los/'

files = glob.glob(path_LOS+'*.dat')
files_to_remove = glob.glob(path_LOS+'*fXnone*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])
files_to_remove = glob.glob(path_LOS+'*fX1.0*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])
files_to_remove = glob.glob(path_LOS+'*fX0.8*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])
files_to_remove = glob.glob(path_LOS+'*fX0.6*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])

logfX = np.empty(len(files))
xHI_mean = np.empty(len(files))
TK_mean = np.empty(len(files))
T_HI = np.empty(len(files))

for j in range(len(files)):

  data  = np.fromfile(str(files[j]),dtype=np.float32)
  Nbins = int(data[7])					#Number of pixels/cells/bins in one line-of-sight
  Nlos = int(data[8])						#Number of lines-of-sight
  x_initial = 12
  
  logfX[j]    = data[9]
  xHI_mean[j] = data[11]

  xHI     = data[(x_initial+2*Nbins+1*Nlos*Nbins):(x_initial+2*Nbins+2*Nlos*Nbins)]
  TK      = data[(x_initial+2*Nbins+2*Nlos*Nbins):(x_initial+2*Nbins+3*Nlos*Nbins)]
  TK_mean[j] = np.mean(TK)
  ind_neu = np.where(xHI>=0.9)[0]
  T_HI[j] = np.mean(TK[ind_neu])

array = np.array([len(files)])
array = np.append(array,xHI_mean)
array = np.append(array,logfX)
array = np.append(array,TK_mean)
array = np.append(array,T_HI)
array.astype('float32').tofile('interpolators/IGMprop_200Mpc_z%.1f.dat' % (z_name),sep='')
'''
datafile = str('interpolators/IGMprop_200Mpc_z%.1f.dat' % (z_name))
data = np.fromfile(str(datafile),dtype=np.float32)
Nmodels = int(data[0])
xHI_mean_lim = data[1:(Nmodels+1)]
logfX_lim = data[(Nmodels+1):(2*Nmodels+1)]
TK = data[(2*Nmodels+1):(3*Nmodels+1)]
T_HI = data[(3*Nmodels+1):(4*Nmodels+1)]

inter_fun_fx = interpolate.LinearNDInterpolator(np.transpose([xHI_mean_lim,T_HI]),logfX_lim)
inter_fun_THI = interpolate.LinearNDInterpolator(np.transpose([xHI_mean_lim,logfX_lim]),T_HI)

#ax0left = ax0.twinx()
#ax0left.set_yticks(np.arange(0.,1000.,200.))
#ax0left.set_ylim(0.,1000.)


#print('T_HI=%.2f-%.2f' % (np.amin(T_HI),np.amax(T_HI)))
#print('x_HI=%.4f-%.4f' % (np.amin(xHI_mean_lim),np.amax(xHI_mean_lim)))

xHI = np.arange(0.01,0.999,0.01)
#xHI = np.append(np.amin(xHI_mean)*1.01,xHI)
#xHI = np.append(xHI,np.amax(xHI_mean)*0.99)
#print(xHI)

T_HI = 15.6
logfX_down = inter_fun_fx(xHI,T_HI)

T_HI = 656.7
logfX_up = inter_fun_fx(xHI,T_HI)

#ax0.fill_between(xHI,logfX_down,logfX_up,alpha=0.2,color=colours_lit[3])
#ax0.text(0.45,-0.85,'HERA+23',color=colours_lit[3],fontsize=12)

datafile = str('interpolators/nulldetection_limits_200Mpc_z%.1f_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f_Nobs%d_Nkbins%d.dat' % (z_name,telescope,spec_res,tint,S147,alphaR,10,Nkbins))
data = np.fromfile(str(datafile),dtype=np.float32)
xHI_lim_68 = np.reshape(data,(4,-1))[0]
logfX_lim_68 = np.reshape(data,(4,-1))[1]
xHI_lim_95 = np.reshape(data,(4,-1))[2]
logfX_lim_95 = np.reshape(data,(4,-1))[3]

ind = np.arange(18,len(xHI_lim_68),10)
#print(xHI_lim_68[ind])
#print(logfX_lim_68[ind])
#print(inter_fun_THI(xHI_lim_68[ind],logfX_lim_68[ind]))

#ax0.plot(xHI_lim_68,logfX_lim_68,linestyle='-',color='black',linewidth=1.5)
#ax0.plot(xHI_lim_95,logfX_lim_95,linestyle='--',color='black',linewidth=1.5)

plt.title(r'%s, %d hr, $G=%.2f$' % (telescope,tint,G_all), fontsize=fsize)

plt.tight_layout()
plt.savefig('post_maps/nsub/multiparam_infer_%dNkbins_%s_%dhr_%dsteps.png' % (Nkbins,telescope,tint,Nsteps))
plt.show()


print('Noise subtracted')

path = 'MCMC_samples_nsub'
Nsteps = 100000
autocorr_cut = 1
fig = plt.figure(figsize=(5.,5.))
gs = gridspec.GridSpec(1,1)
ax0 = plt.subplot(gs[0,0])

contkwarg = dict(alpha=[0.,0.25,0.5])
logfX_infer = np.empty(len(logfX))
xHI_infer = np.empty(len(logfX))
G_all = np.empty(len(logfX))

for i in range(len(logfX)):

    #if i==7:
    #    Nsteps = 100000
    #else:
    #    Nsteps = 10000

    data = np.load('%s/flatsamp_nsub_200Mpc_xHI%.2f_fX%.1f_%s_%dkHz_Smin%.1fmJy_alphaR%.2f_t%dh_dk%.2f_%dkbins_%dsteps.npy' % (path,xHI_mean[i],logfX[i],telescope,spec_res,S147,alphaR,tint,d_log_k_bins,Nkbins,Nsteps))
    corner.hist2d(data[autocorr_cut:,0],data[autocorr_cut:,1],levels=[1-np.exp(-0.5),1-np.exp(-2.)],plot_datapoints=False,plot_density=False,fill_contours=True,color=colours[i])#,contourf_kwargs=contkwarg)
    #corner.hist2d(data[:,0],data[:,1],levels=[1-np.exp(-0.5),1-np.exp(-2.)],smooth=True,plot_datapoints=True,plot_density=True,color=colours[i])

    logfX_infer[i] = data[0,1]
    xHI_infer[i] = data[0,0]

    G_all[i] = np.mean((data[autocorr_cut:,0]-xHI_mean[i])**2+(data[autocorr_cut:,1]-logfX[i])**2)



for i in range(len(logfX)):
    ax0.scatter(xHI_infer[i],logfX_infer[i],marker='o',s=200,linewidths=1.,color=colours[i],edgecolors='black',alpha=1)
    ax0.scatter(xHI_mean[i],logfX[i],marker='*',s=200,linewidths=1.,color=colours[i],edgecolors='black',alpha=1)

print('Mock')
print(xHI_mean)
print(logfX)
print('Inferred')
print(xHI_infer)
print(logfX_infer)
G_5 = np.sqrt(np.mean((xHI_infer-xHI_mean)**2+(logfX_infer-logfX)**2))
print('G_5   = %.6f' % G_5)
G_all = np.sqrt(np.mean(G_all))
print('G_all = %.6f' % G_all)

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

ax0.axvspan(0.21-0.07,0.21+0.17,alpha=0.2,color=colours_lit[0])
ax0.text(0.15,-1.80,'Ďurovčíková+24',color=colours_lit[0],rotation=rot,fontsize=12)

ax0.axvspan(0.17-0.11,0.17+0.09,alpha=0.2,color=colours_lit[1])
ax0.text(0.08,-1.80,'Gaikwad+23',color=colours_lit[1],rotation=rot,fontsize=12)

ax0.axvspan(0,0.21,alpha=0.2,color=colours_lit[2])
ax0.text(0.005,-1.80,'Greig+24',color=colours_lit[2],rotation=rot,fontsize=12)

#ax0.plot(xHI_lim_68,logfX_lim_68,linestyle='-',color='black',linewidth=1.5)
#ax0.plot(xHI_lim_95,logfX_lim_95,linestyle='--',color='black',linewidth=1.5)
plt.title(r'%s, %d hr, $G=%.2f$' % (telescope,tint,G_all), fontsize=fsize)

plt.tight_layout()
plt.savefig('post_maps/nsub/multiparam_infer_%dNkbins_%s_%dhr_%dsteps_nsub.png' % (Nkbins,telescope,tint,Nsteps))
plt.show()
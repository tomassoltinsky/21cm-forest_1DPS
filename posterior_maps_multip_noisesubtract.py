'''
Plot 2D posterior maps as a function of <xHI> and logfX for multiple combinations of these parameters
for the noisy and noise-subtracted 1D PS of the 21-cm forest.

Version 04.07.2025
'''

import sys
import glob, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
import corner

z_name = 6.000
spec_res = 8
S147 = 64.2
alphaR = -0.44
Nsteps = 100000
autocorr_cut = 1000

Nkbins = 6
d_log_k_bins = 0.25

telescope = sys.argv[1]
tint = int(sys.argv[2])
xHI_mean = [0.11,0.80,0.52,0.11,0.80]
logfX    = [-1.0,-1.0,-2.0,-3.0,-3.0]

xHI_true = np.empty(len(xHI_mean))
logfX_true = np.empty(len(logfX))
for i in range(len(logfX)):

   data = np.fromfile('datasets/21cmFAST_los/los/los_50Mpc_256_n1000_z6.000_fX%.1f_xHI%.2f.dat' % (logfX[i],xHI_mean[i]),dtype=np.float32)
   logfX_true[i] = data[9]
   xHI_true[i] = data[11]

print('Telescope: %s, integration time: %d hr' % (telescope,tint))
print('With noise')

fsize = 20
colours  = ['royalblue','fuchsia','forestgreen','darkorange','red','lightcoral','slateblue','limegreen','teal','navy']

fig = plt.figure(figsize=(5.,5.))
gs = gridspec.GridSpec(1,1)
ax0 = plt.subplot(gs[0,0])

contkwarg = dict(alpha=[0.,0.25,0.5])
logfX_infer = np.empty(len(logfX))
xHI_infer = np.empty(len(logfX))
std_xHI = np.empty(len(logfX))
std_logfX = np.empty(len(logfX))

for i in range(len(logfX)):
    data = np.load('MCMC_samples/flatsamp_200Mpc_xHI%.2f_fX%.1f_%s_%dkHz_Smin%.1fmJy_alphaR%.2f_t%dh_dk%.2f_%dkbins_%dsteps.npy' % (xHI_mean[i],logfX[i],telescope,spec_res,S147,alphaR,tint,d_log_k_bins,Nkbins,Nsteps))
    corner.hist2d(data[autocorr_cut:,0],data[autocorr_cut:,1],levels=[1-np.exp(-0.5),1-np.exp(-2.)],plot_datapoints=False,plot_density=False,fill_contours=True,color=colours[i])#,contourf_kwargs=contkwarg)

    logfX_infer[i] = data[0,1]
    xHI_infer[i] = data[0,0]

    std_xHI[i] = np.std(data[autocorr_cut:,0])
    std_logfX[i] = np.std(data[autocorr_cut:,1])

for i in range(len(logfX)):
    ax0.scatter(xHI_infer[i],logfX_infer[i],marker='o',s=200,linewidths=1.,color=colours[i],edgecolors='black',alpha=1)
    ax0.scatter(xHI_true[i],logfX_true[i],marker='*',s=200,linewidths=1.,color=colours[i],edgecolors='black',alpha=1)

print('Mock')
print(logfX_true)
print(xHI_true)
print('Inferred')
print(xHI_infer)
print(logfX_infer)

G_5 = np.sqrt(np.mean((xHI_infer-xHI_true)**2+(logfX_infer-logfX_true)**2))
print('G_5       = %.6f' % G_5)
std_total = np.mean(np.sqrt(std_xHI**2+std_logfX**2))
print('std_total = %.6f' % std_total)

ax0.set_xticks(np.arange(0.,1.1,0.2))
ax0.set_yticks(np.arange(-4.,0.1,1.))
ax0.set_xlim(0.,1.)
ax0.set_ylim(-4,-0.3)
ax0.xaxis.set_minor_locator(AutoMinorLocator())
ax0.yaxis.set_minor_locator(AutoMinorLocator())
ax0.set_xlabel(r'$\langle x_{\rm HI}\rangle$', fontsize=fsize)
ax0.set_ylabel(r'$\log_{10}f_{\mathrm{X}}$', fontsize=fsize)
ax0.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=5,width=1)



colours_lit = ['grey','brown','darkviolet','navy']
rot = 60
'''
ax0.axvspan(0.21-0.07,0.21+0.17,alpha=0.2,color=colours_lit[0])
ax0.text(0.15,-1.80,'Ďurovčíková+24',color=colours_lit[0],rotation=rot,fontsize=12)

ax0.axvspan(0.17-0.11,0.17+0.09,alpha=0.2,color=colours_lit[1])
ax0.text(0.08,-1.80,'Gaikwad+23',color=colours_lit[1],rotation=rot,fontsize=12)

ax0.axvspan(0,0.21,alpha=0.2,color=colours_lit[2])
ax0.text(0.005,-1.80,'Greig+24',color=colours_lit[2],rotation=rot,fontsize=12)
'''
ax0.axvspan(0.,0.21+0.17,alpha=0.2,color=colours_lit[0])
ax0.text(0.025,-0.60,r'Limit from Ly$\alpha$ data',color=colours_lit[0],fontsize=9)

plt.title(r'%s, %d hr, $G=%.2f$' % (telescope,tint,G_5), fontsize=fsize)

plt.tight_layout()
plt.savefig('plots/multiparam_infer_%dNkbins_%s_%dhr_%dsteps.pdf' % (Nkbins,telescope,tint,Nsteps))
plt.show()



print('Noise subtracted')

Nsteps = 100000
autocorr_cut = 1
fig = plt.figure(figsize=(5.,5.))
gs = gridspec.GridSpec(1,1)
ax0 = plt.subplot(gs[0,0])

contkwarg = dict(alpha=[0.,0.25,0.5])
logfX_infer = np.empty(len(logfX))
xHI_infer = np.empty(len(logfX))
std_xHI = np.empty(len(logfX))
std_logfX = np.empty(len(logfX))

for i in range(len(logfX)):

    data = np.load('MCMC_samples/noise_sub/flatsamp_nsub_200Mpc_xHI%.2f_fX%.1f_%s_%dkHz_Smin%.1fmJy_alphaR%.2f_t%dh_dk%.2f_%dkbins_%dsteps.npy' % (xHI_mean[i],logfX[i],telescope,spec_res,S147,alphaR,tint,d_log_k_bins,Nkbins,Nsteps))
    corner.hist2d(data[autocorr_cut:,0],data[autocorr_cut:,1],levels=[1-np.exp(-0.5),1-np.exp(-2.)],plot_datapoints=False,plot_density=False,fill_contours=True,color=colours[i])#,contourf_kwargs=contkwarg)

    logfX_infer[i] = data[0,1]
    xHI_infer[i] = data[0,0]

    std_xHI[i] = np.std(data[autocorr_cut:,0])
    std_logfX[i] = np.std(data[autocorr_cut:,1])

for i in range(len(logfX)):
    ax0.scatter(xHI_infer[i],logfX_infer[i],marker='o',s=200,linewidths=1.,color=colours[i],edgecolors='black',alpha=1)
    ax0.scatter(xHI_true[i],logfX_true[i],marker='*',s=200,linewidths=1.,color=colours[i],edgecolors='black',alpha=1)

print('Mock')
print(logfX_true)
print(xHI_true)
print('Inferred')
print(xHI_infer)
print(logfX_infer)
G_5 = np.sqrt(np.mean((xHI_infer-xHI_true)**2+(logfX_infer-logfX_true)**2))
print('G_5       = %.6f' % G_5)
std_total = np.mean(np.sqrt(std_xHI**2+std_logfX**2))
print('std_total = %.6f' % std_total)

ax0.set_xticks(np.arange(0.,1.1,0.2))
ax0.set_yticks(np.arange(-4.,0.1,1.))
ax0.set_xlim(0.,1.)
ax0.set_ylim(-4,-0.3)
ax0.xaxis.set_minor_locator(AutoMinorLocator())
ax0.yaxis.set_minor_locator(AutoMinorLocator())
ax0.set_xlabel(r'$\langle x_{\rm HI}\rangle$', fontsize=fsize)
ax0.set_ylabel(r'$\log_{10}f_{\mathrm{X}}$', fontsize=fsize)
ax0.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=5,width=1)
'''
ax0.axvspan(0.21-0.07,0.21+0.17,alpha=0.2,color=colours_lit[0])
ax0.text(0.15,-1.80,'Ďurovčíková+24',color=colours_lit[0],rotation=rot,fontsize=12)

ax0.axvspan(0.17-0.11,0.17+0.09,alpha=0.2,color=colours_lit[1])
ax0.text(0.08,-1.80,'Gaikwad+23',color=colours_lit[1],rotation=rot,fontsize=12)

ax0.axvspan(0,0.21,alpha=0.2,color=colours_lit[2])
ax0.text(0.005,-1.80,'Greig+24',color=colours_lit[2],rotation=rot,fontsize=12)
'''
ax0.axvspan(0.,0.21+0.17,alpha=0.2,color=colours_lit[0])
ax0.text(0.025,-0.60,r'Limit from Ly$\alpha$ data',color=colours_lit[0],fontsize=9)

plt.title(r'%s, %d hr, $G=%.2f$' % (telescope,tint,G_5), fontsize=fsize)

plt.tight_layout()
plt.savefig('plots/multiparam_infer_%dNkbins_%s_%dhr_%dsteps_nsub.pdf' % (Nkbins,telescope,tint,Nsteps))
plt.show()
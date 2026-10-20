"""
Plot constraints from a null detection of the 21-cm forest 1D power spectrum
based on the uGMRT archival observations of PSO J352-15.

The code plots the posterior constraints in the mean neutral hydrogen fraction -
mean neutral IGM temperature parameter space, compares them with literature
constraints, extracts numerical limits from the posterior contour, and produces
a diagnostic corner plot.

Version 25.04.2026
"""

#Import necessary packages
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
import corner
from scipy import interpolate
from scipy.interpolate import interp1d


#Input parameters
WLDlvl = int(sys.argv[1])           #Level of wavelet decomposition used for residual bandpass subtraction
f_flag = float(sys.argv[2])         #Threshold for flagging bad channels in units of mean noise rms


#Observation and MCMC parameters
path = 'MCMC_samples/uGMRT_archivaldata'
telescope = 'uGMRT'
spec_res = 6.1                      #Spectral resolution in kHz

z_name = 6.000
freq_low = 209.                     #Lower frequency limit in MHz
freq_high = 222.                    #Upper frequency limit in MHz

Nsteps = 100000
Nwalkers = 64

Nkbins = 11
d_log_k_bins = 0.25


#MCMC output suffix
suffix = 'uGMRT-AD_logTHI-xHI_obs_postprocess_flagmock_mask_PSOJ352−15_WLDlvl%d_fflag%.2f_%.1f-%.1fMHz_%.1fkHz_%s_dk%.2f_%dkbins_%dw_%dsteps' % (WLDlvl,f_flag,freq_low,freq_high,spec_res,telescope,d_log_k_bins,Nkbins,Nwalkers,Nsteps)


#Load IGM properties across the simulated parameter space
datafile = str('datasets/interpolators/IGMprop_200Mpc_z%.1f.dat' % z_name)
data = np.fromfile(str(datafile),dtype=np.float32)

Nmodels = int(data[0])
xHI_mean = data[1:(Nmodels+1)]
logfX = data[(Nmodels+1):(2*Nmodels+1)]
TK = data[(2*Nmodels+1):(3*Nmodels+1)]
T_HI = data[(3*Nmodels+1):(4*Nmodels+1)]

print('Nmodels: %d' % Nmodels)
print('xHI range: %.3f-%.3f' % (np.amin(xHI_mean),np.amax(xHI_mean)))
print('T_HI range: %.2f-%.2f' % (np.amin(T_HI),np.amax(T_HI)))

#Interpolate logfX across the xHI-logTHI parameter space
logTHI_grid = np.log10(T_HI)
inter_fun_fx = interpolate.LinearNDInterpolator(np.transpose([xHI_mean,logTHI_grid]),logfX)


#Plot posterior constraints
fsize = 18

fig = plt.figure(figsize=(5.,5.))
gs = gridspec.GridSpec(1,1)
ax0 = plt.subplot(gs[0,0])

#Load flattened MCMC samples
data = np.load('%s/flatsamp_%s.npy' % (path,suffix))

xHI_sample = data[:,0]
logTHI_sample = data[:,1]

#Plot posterior contours
contour_levels = [1-np.exp(-0.),1-np.exp(-2.)]
corner.hist2d(xHI_sample,logTHI_sample,levels=contour_levels,
              plot_datapoints=False,plot_density=False,
              plot_contours=True,fill_contours=True,color='royalblue')

#Set plot limits and x-axis ticks
ax0.set_xticks(np.arange(0.,1.1,0.2))
ax0.set_xlim(0.,1.)
ax0.set_ylim(np.log10(15),np.log10(240))

#Display temperature axis in K
y_labels = np.arange(25,241,25)
y_labels_pos = np.log10(y_labels)
ax0.set_yticks(y_labels_pos)
ax0.set_yticklabels(y_labels.astype(int))

#Add minor temperature ticks
y_minor = np.arange(15,241,5)
y_minor = np.setdiff1d(y_minor,y_labels)
ax0.set_yticks(np.log10(y_minor),minor=True)

ax0.xaxis.set_minor_locator(AutoMinorLocator())

ax0.set_xlabel(r'$\langle x_{\rm HI}\rangle$',fontsize=fsize)
ax0.set_ylabel(r'$\langle T_{\mathrm{HI}}\rangle\,\rm [K]$',fontsize=fsize)

ax0.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True,
                length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True,
                length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True,
                length=5,width=1)

ax0.text(0.24,1.775,r'J352-15 (uGMRT; this work)',color='royalblue',rotation=22,fontsize=fsize-3)


#Add literature xHI constraints at z=5.5-5.75 from Gaikwad et al. 2023, Umeda et al. 2025 and Qin et al. 2025
ax0.axvspan(0.,0.06+0.12,alpha=0.5,color='grey')
ax0.text(0.05,np.log10(65),r'$x_{\rm HI}$ limit from',color='black',rotation=90,fontsize=fsize-3)
ax0.text(0.12,np.log10(80),r'Ly$\alpha$ data',color='black',rotation=90,fontsize=fsize-3)


#Add temperature constraints at z=6 from Dhandha et al. 2025
T_HI_low = 19.
T_HI_high = 1258.1

ax0.fill_between([0,1],np.log10(T_HI_low),np.log10(T_HI_high),alpha=0.2,color='red')
ax0.text(0.27,np.log10(T_HI_low+4.5),r'$T_{\rm HI}$ limit from ',color='red',rotation=0,fontsize=fsize-3)
ax0.text(0.27,np.log10(T_HI_low+1),r'21-cm tomography data',color='red',rotation=0,fontsize=fsize-3)

plt.tight_layout()
plt.savefig('plots/Soltinsky_2026/postdist_uGMRT-AD_flagmock_logTHI-xHI_obs_WLDlvl%d_fflag%.2f_%.1f-%.1fMHz_%.1fkHz_%dsteps.png' % (WLDlvl,f_flag,freq_low,freq_high,spec_res,Nsteps))
plt.show()
plt.close()


#Extract the posterior contour used to determine numerical limits
fig, ax = plt.subplots()

contour_levels = [1-np.exp(-2.)]

corner.hist2d(
    xHI_sample,
    logTHI_sample,
    levels=contour_levels,
    plot_datapoints=False,
    plot_density=False,
    plot_contours=True,
    fill_contours=False,
    color='royalblue',
    ax=ax
)

#Force draw so that matplotlib builds the contour objects
fig.canvas.draw()

#Extract contour line paths
paths = []
for coll in ax.collections:
    try:
        paths.extend(coll.get_paths())
    except Exception:
        pass

#Keep only reasonably long contour paths
paths = [p for p in paths if len(p.vertices)>20]

#If several paths exist, select the longest one
path = max(paths,key=lambda p: len(p.vertices))
verts = path.vertices

x_cont = verts[:,0]
logT_cont = verts[:,1]


#Find the lower temperature envelope of the contour as a function of xHI
xbins = np.linspace(np.nanmin(x_cont),np.nanmax(x_cont),1000)
x_mid = 0.5*(xbins[:-1]+xbins[1:])
logT_low = np.full_like(x_mid,np.nan)

for i in range(len(x_mid)):
    m = (x_cont>=xbins[i]) & (x_cont<xbins[i+1])

    if np.any(m):
        logT_low[i] = np.nanmin(logT_cont[m])

valid = np.isfinite(logT_low)


#Interpolate the temperature limit as a function of xHI
interp_logT = interp1d(
    x_mid[valid],
    logT_low[valid],
    bounds_error=False,
    fill_value=np.nan
)


#Interpolate the xHI limit as a function of temperature
interp_xHI = interp1d(
    logT_low[valid],
    x_mid[valid],
    bounds_error=False,
    fill_value=np.nan
)


#Evaluate temperature limits at selected xHI values
xHI_eval = np.array([0.05,0.1,0.2,0.5,0.8,0.99])
logT_eval = interp_logT(xHI_eval)
T_eval = 10**logT_eval

print('\nTemperature limits at selected <xHI>:')
print('%10s %15s %12s' % ('<xHI>','log10(THI/K)','THI [K]'))

for xHI_i,logT_i,T_i in zip(xHI_eval,logT_eval,T_eval):
    print('%10.4g %15.4g %12.4g' % (xHI_i,logT_i,T_i))


#Evaluate xHI limits at selected temperatures
T_eval = np.arange(16,20.1,1)
logT_eval = np.log10(T_eval)
xHI_eval = interp_xHI(logT_eval)

print('\nxHI limits at selected <THI>:')
print('%10s %15s %12s' % ('<xHI>','log10(THI/K)','THI [K]'))

for xHI_i,logT_i,T_i in zip(xHI_eval,logT_eval,T_eval):
    print('%10.4g %15.4g %12.4g' % (xHI_i,logT_i,T_i))

plt.close()


#Calculate posterior median values and uncertainties
param_label = ['<xHI>','<logTHI>']
ndim = 2
para_mean = np.zeros(ndim)

print('\nPosterior constraints:')

for j in range(ndim):
    mcmc = np.percentile(data[:,j],[16,50,84])
    q = np.diff(mcmc)
    para_mean[j] = mcmc[1]

    print('%10s = %10.4g + %10.4g - %10.4g' % (param_label[j],mcmc[1],q[1],q[0]))


#Plot full posterior distributions
labels = [r"$\langle x_{\rm HI}\rangle$",r"$\langle T_{\mathrm{HI}}\rangle\,\rm [K]$"]

THI_min = np.min(data[:,1])
THI_max = np.max(data[:,1])

sett = dict(fontsize=14)

fig = corner.corner(data,range=[[0.,1.],[THI_min,THI_max]],color='royalblue',
                    smooth=True,labels=labels,label_kwargs=sett)

#Set ticks in the xHI-THI panel
ax = fig.axes[2]

x_labels = np.arange(0,1.1,0.2)
ax.set_xticks(x_labels)

y_labels = np.array([30,100,300,1000,3000])
y_labels_pos = np.log10(y_labels)
ax.set_yticks(y_labels_pos)
ax.set_yticklabels(y_labels.astype(int))

#Display the marginalized temperature axis in K
ax = fig.axes[3]
ax.set_xticks(y_labels_pos)
ax.set_xticklabels(y_labels.astype(int))

plt.savefig('plots/Soltinsky_2026/inferred_param_flagmock_logTHI-xHI_obs_WLDlvl%d_fflag%.2f_%.1f-%.1fMHz_%.1fkHz_%dsteps.png' % (WLDlvl,f_flag,freq_low,freq_high,spec_res,Nsteps))
plt.close()
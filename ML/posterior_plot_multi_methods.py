'''
Plot the posterior distributions for a single test points from multiple methods.
'''

import sys
import glob, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
import corner
import argparse
import plot_results as pltr
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.colors as mc
import colorsys
from matplotlib.lines import Line2D

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from matplotlib.patches import Rectangle
from collections import defaultdict
from plot_results import hist2d

def lighten_color(color, amount):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def average_group_std(y_pred, y_true):
    unique_classes = np.unique(y_true)
    stds = []

    for val in unique_classes:
        group_preds = y_pred[y_true == val]
        std = np.std(group_preds, ddof=1)  # Sample standard deviation
        stds.append(std)

    avg_std = np.mean(stds)
    return avg_std


def average_combined_std(data: np.ndarray) -> float:
    """
    Parameters
    ----------
    data : ndarray, shape (N, 4)
        Column 0 : predicted x_HI
        Column 1 : predicted log10 f_X
        Column 2 : true      x_HI
        Column 3 : true      log10 f_X

    Returns
    -------
    float
        The mean of the standard deviations of the per-row “combined
        variance”, taken over all unique (x_HI_true, logf_X_true) pairs.
    """
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError("Input array must be N×4.")

    # split predictions and truths
    pred   = data[:, :2]       # shape (N, 2)
    truth  = data[:, 2:]       # shape (N, 2)

    # row-wise errors
    err    = pred - truth      # shape (N, 2)

    # combined variance for each row (population variance across the two errors)
    row_var = np.var(err, axis=1)   # shape (N,)

    # group rows by their true-value pair
    buckets = defaultdict(list)
    for rv, (x_t, logf_t) in zip(row_var, truth):
        buckets[(x_t, logf_t)].append(rv)

    # std-dev of the row-variances inside each true-value group
    group_stds = [np.std(v, ddof=0) for v in buckets.values()]

    # average of those five standard deviations
    return float(np.mean(group_stds))


# Example usage:
# y_true = np.repeat(np.array([0, 1, 2, 3, 4]), 10000)
# y_pred = np.random.normal(loc=y_true, scale=1.0, size=50000)
# print(average_group_std(y_true, y_pred))


parser = argparse.ArgumentParser(description='Predict reionization parameters from 21cm forest')
parser.add_argument('-p', '--filepath', type=str, default='unet', help='filepath for the test results')
parser.add_argument('-d', '--datapath', type=str, default='unet', help='directory path for the IGM data')
parser.add_argument('-t', '--telescope', type=str, default='uGMRT', help='telescope')
parser.add_argument('-i', '--t_int', type=float, default=500, help='integration time of obsevation in hours')
args = parser.parse_args()

#Set parameters describing data
#results_file = 'noisy_ps_f21_inference_ps_train_test_uGMRT_t50.0_20250418124600.csv'
#results_file = 'denoised_ps_f21_inference_ps_train_test_uGMRT_t50.0_20250418124722.csv'
#results_file = 'latent_f21_inference_unet_with_dense_train_test_uGMRT_t50.0_20250413160932.csv'
spec_res = 8
S147 = 64.2
alphaR = -0.44
Nsteps = 10000
autocorr_cut = 1000

telescope = args.telescope
tint = args.t_int
#xHI_mean = [0.11,0.80,0.52,0.11,0.80]
#logfX    = [-1.0,-1.0,-2.0,-3.0,-3.0]

print(f"loading result file using pattern {args.filepath}")


methods = ['basic','nsub','noisy','denoised','latent'] #  ,'MLdenoised'
method_types = ['bayes','bayes','xgb','xgb','xgb'] # ,'bayes'
method_labels = ['Method A1', 'Method A2', 'Method B1', 'Method B2', 'Method B3'] # , 'MCMC: ML denoised'


#Start plotting
fsize = 20
fsize_meas = 16
fsize_legend = 14
#colours  = ['royalblue','fuchsia','forestgreen','darkorange','limegreen','slateblue','lightcoral','red','teal','navy']
#colours  = ['royalblue','mediumseagreen','darkorange','mediumturquoise','orchid', 'burlywood','forestgreen','teal','slateblue','limegreen','lightcoral','fuchsia','red','navy']
colours  = ['#A7D9F0','#B3E0B3','#FFDAB9','#B3E0D9','#D9B3E0']
colours  = ['royalblue','mediumseagreen','darkorange','mediumturquoise','orchid'] 
#plt.style.use('seaborn-v0_8-pastel')
fig = plt.figure(figsize=(8.,8.))
gs = gridspec.GridSpec(1,1)
ax0 = plt.subplot(gs[0,0])
xHI_mean = 0.11
logfX    = -3.0


#Plot the x_HI measurements from the Lyα forest
colours_lit = ['grey','brown','darkviolet','navy']
rot = 36

"""
ax0.axvspan(0.21-0.07,0.21+0.17,alpha=0.2,color=colours_lit[0])
ax0.text(0.15,-1.8,'Ďurovčíková+24',color=colours_lit[0],rotation=rot,fontsize=fsize_meas)  #Ďurovčíková et al. 2024, ApJ, 969, 162

ax0.axvspan(0.17-0.11,0.17+0.09,alpha=0.2,color=colours_lit[1])
ax0.text(0.08,-1.8,'Gaikwad+23',color=colours_lit[1],rotation=rot,fontsize=fsize_meas)      #Gaikwad et al. 2023, MNRAS, 525, 4093

ax0.axvspan(0,0.21,alpha=0.2,color=colours_lit[2])
ax0.text(0.005,-1.8,'Greig+24',color=colours_lit[2],rotation=rot,fontsize=fsize_meas)       #Greig et al. 2024, MNRAS, 530, 3208
"""
ax0.axvspan(0,0.21+0.17,alpha=0.2,color=colours_lit[0])
ax0.text(0.02, -1.15,r'Limit from Ly$\alpha$ data',color='darkgrey',fontsize=fsize_meas)       #Greig et al. 2024, MNRAS, 530, 3208

legends = []
for i, method in enumerate(methods):
   if method_types[i] == 'xgb':
      results_file = glob.glob(f"{args.filepath}/{method}*/test_results.csv")
      print(f"loading result file using pattern {results_file}")
      all_results = np.loadtxt(results_file[0], delimiter=",", skiprows=1)
      print(f'Total rows in file: {len(all_results)}')
      # Round columns 3 and 4 to two decimal places before comparison
      rounded_xHI = np.round(all_results[:, 2], 2)
      rounded_logfX = np.round(all_results[:, 3], 2)
      all_results = all_results[(rounded_xHI == xHI_mean) & (rounded_logfX == logfX)]
      print(f'Filtered rows in file: {len(all_results)}')
      xHI_mean_post = np.reshape(all_results[:,0],(-1,Nsteps))
      logfX_post = np.reshape(all_results[:,1],(-1,Nsteps))

      print(xHI_mean_post)
      print(logfX_post)
      #Read the best fit values from the MCMC
      logfX_infer = np.mean(logfX_post)
      xHI_infer = np.mean(xHI_mean_post)

   elif method_types[i] == 'bayes':
      results_file = f"{args.filepath}/flatsamp*{method}*.npy"
      print(f"loading result file using pattern {results_file}")

      data = np.load(glob.glob(results_file)[0])
      xHI_mean_post = data[autocorr_cut:,0]
      logfX_post = data[autocorr_cut:,1]

      all_results = data[autocorr_cut:, 0:2]
      truevalues = np.zeros((all_results.shape[0], 2))
      truevalues[:,0] = xHI_mean
      truevalues[:,1] = logfX
      all_results = np.hstack((all_results, truevalues))
      print(f'{all_results[:10]}')
      #corner.hist2d(data[:,0],data[:,1],levels=[1-np.exp(-0.5),1-np.exp(-2.)],smooth=True,plot_datapoints=True,plot_density=True,color=colours[i])

      logfX_infer = data[0,1]
      xHI_infer = data[0,0]


   #Plot the posterior distributions from the MCMC using corner package (Foreman-Mackey 2016, The Journal of Open Source Software, 1, 24)
   zorder = 1
   if i == 0: zorder = 2
   hist2d(xHI_mean_post, logfX_post, levels=[1-np.exp(-2.)], smooth=True, plot_datapoints=False, 
                 plot_density=False, plot_contours=True, fill_contours=True, color=colours[i], lighten_fill=True,
                   contourf_kwargs={'zorder': zorder}, contour_kwargs={'zorder': zorder} )
   #plt.contour(xedges[:-1], yedges[:-1], H.T, ax=ax0, levels=2, colors=colours[i]) # You can change levels and colors

   #,contourf_kwargs=contkwarg)
   # 1-np.exp(-1.), 

   #Compute the goodness metric
   g_score = np.sqrt(np.mean((xHI_infer-xHI_mean)**2+(logfX_infer-logfX)**2))
   print('G-Score=%.6f' % g_score)

   r2score_xHI = r2_score(all_results[:,:1], all_results[:,2:3])
   r2score_fX = r2_score(all_results[:,1:2], all_results[:,3:4])
   r2score = 0.5*(r2score_xHI+r2score_fX)
   print('r2_score=%.6f | %.6f | %.6f' % (r2score_xHI,r2score_fX,r2score))


   g_all_score = pltr.rmse_all(all_results[:,:2], all_results[:,2:4])
   print('G-all Score=%.6f' % g_all_score)

   sigma_xHI = average_group_std(all_results[:,:1], all_results[:,2:3])
   sigma_fX = average_group_std(all_results[:,1:2], all_results[:,3:4])
   sigma = 0.5*(sigma_xHI+sigma_fX)
   print('sigma=%.6f | %.6f | %.6f' % (sigma_xHI,sigma_fX,sigma))

   sigma = average_combined_std(all_results)
   print('sigma=%.6f' % (sigma))

   #plot_nongaussian_2sigma(np.vstack((xHI_mean_post, logfX_post)).T, label=f'{method_labels[i]}, G={g_score:.2f}', ax=ax0, color=colours[i])
   #Plot the best fit and true values
   #ax0.scatter(xHI_infer, logfX_infer, marker='o', s=200, linewidths=1., color=colours[i], edgecolors='black', alpha=1, label=f'{method_labels[i]}, G={g_score:.2f}', zorder=10)
   #marker_elem = ax0.scatter(xHI_infer, logfX_infer, marker='o', s=200, linewidths=1., color=colours[i], edgecolors='black', alpha=1, label=f'{method_labels[i]}, G={g_score:.2f}', zorder=10)
   #marker_elem.set_visible(False)
   
   # Create an invisible circle marker for legend only
   circle_legend = Line2D(
      [], [],                      # empty data, so nothing plotted
      marker='o',  
      color=colours[i],                # circle marker
      #edgecolor='black',              
      linestyle='None',            # no line
      markersize=20,                # size of marker
      label=f'{method_labels[i]}, G={g_score:.2f}'
   )


   rectangle = Rectangle((0, 0), 20, 18,
                              linewidth=1,  facecolor=colours[i], label=f'{method_labels[i]}, G={g_score:.2f}')
   legends.append(rectangle)
   print('Mock xHI and fX values')
   print(xHI_mean)
   print(logfX)
   print('Inferred xHI and fX values')
   print(xHI_infer)
   print(logfX_infer)

   
ax0.scatter(xHI_mean, logfX, marker='*', s=200, linewidths=1., color='black', edgecolors='black', alpha=1, zorder=100)
#Make the plot look nice
ax0.set_xticks(np.arange(0.,1.1,0.2))
ax0.set_yticks(np.arange(-4.,0.1,1.))
ax0.set_xlim(0.,1.)
ax0.set_ylim(-4,-1)
ax0.xaxis.set_minor_locator(AutoMinorLocator())
ax0.yaxis.set_minor_locator(AutoMinorLocator())
ax0.set_xlabel(r'$\langle x_{\rm HI}\rangle$', fontsize=fsize)
ax0.set_ylabel(r'$\log_{10} f_{\mathrm{X}}$', fontsize=fsize)
ax0.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize,zorder=1000, pad=9)
ax0.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize,zorder=1000)
ax0.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=5,width=1,zorder=1000)





#This bit is about computing which <T_HI> corresponds to the <xHI> and fX values
from scipy import interpolate
import glob, os
z_name = 6.000
'''
#Can be commented out if precomputed interpolators are used
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
datafile = str('%s/interpolators/IGMprop_200Mpc_z%.1f.dat' % (args.datapath, z_name))
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


#Plot the f_X limits corresponding to the T_HI measurements
T_HI = 15.6
logfX_down = inter_fun_fx(xHI,T_HI)

T_HI = 656.7
logfX_up = inter_fun_fx(xHI,T_HI)

#ax0.fill_between(xHI,logfX_down,logfX_up,alpha=0.2,color=colours_lit[3])
#ax0.text(0.45,-0.85,'HERA+23',color=colours_lit[3],fontsize=12)           #HERA collaboration 2023, ApJ, 945, 124

'''
#This bit is about plotting the x_HI and f_X limits based on a null-detection of the signal
#Interpolator for this has to be precomputed
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

ax0.plot(xHI_lim_68,logfX_lim_68,linestyle='-',color='black',linewidth=1.5)
#ax0.plot(xHI_lim_95,logfX_lim_95,linestyle='--',color='black',linewidth=1.5)
'''

#Complete plotting and save
plt.title(r'%s %d hr' % (telescope,tint), fontsize=fsize)
plt.legend(handles=legends, labelspacing = 1, loc='lower right', fontsize=fsize_legend, frameon=False, title=r'$z=6$', title_fontsize=fsize)
plt.tight_layout()
plt.savefig('%s/multimethod_infer_unet_%s_%dhr_%dsteps.pdf' % ("./tmp_out", telescope,tint,Nsteps), format='pdf')

plt.show()

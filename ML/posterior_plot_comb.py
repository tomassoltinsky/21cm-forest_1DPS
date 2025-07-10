'''
Plot 2D posterior maps as a function of <xHI> and logfX for multiple combinations of these parameters.
This is for data based on the ML-Unet.

Version 07.05.2025
'''

import sys
import glob, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
import argparse
import plot_results as pltr
from sklearn.metrics import r2_score

from collections import defaultdict
import f21_predict_base as base

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
    err    = (pred - truth)**2      # shape (N, 2)
    print(f'err.shape={err.shape} \n{err[:2]}')
    # combined variance for each row (population variance across the two errors)
    row_var = np.sum(err, axis=1)   # shape (N,)
    print(f'row_var.shape={row_var.shape} \n{row_var[:2]}')

    # group rows by their true-value pair
    buckets = defaultdict(list)
    for rv, (x_t, logf_t) in zip(row_var, truth):
        buckets[(x_t, logf_t)].append(rv)
    print(f'buckets.keys()={buckets.keys()}')

    # std-dev of the row-variances inside each true-value group
    group_stds = [np.sqrt(np.mean(v)) for v in buckets.values()]

    # average of those five standard deviations
    return float(np.mean(group_stds))


# Example usage:
# y_true = np.repeat(np.array([0, 1, 2, 3, 4]), 10000)
# y_pred = np.random.normal(loc=y_true, scale=1.0, size=50000)
# print(average_group_std(y_true, y_pred))


parser = argparse.ArgumentParser(description='Predict reionization parameters from 21cm forest')
parser.add_argument('-t', '--telescopes', type=str, default='gmrt50h', help='telescope and int time')
parser.add_argument('--feats', type=str, default='latent', help='type of featuers')
parser.add_argument('--titlepref', type=str, default='tele', help='type of title')

args = parser.parse_args()

#Set parameters describing data
#results_file = 'noisy_ps_f21_inference_ps_train_test_uGMRT_t50.0_20250418124600.csv'
#results_file = 'denoised_ps_f21_inference_ps_train_test_uGMRT_t50.0_20250418124722.csv'
#results_file = 'latent_f21_inference_unet_with_dense_train_test_uGMRT_t50.0_20250413160932.csv'
spec_res = 8
S147 = 64.2
alphaR = -0.44
Nsteps = 10000

#Start plotting
fsize = 14
fsize_meas = 9
fsize_legend = 12
colours  = ['royalblue','fuchsia','forestgreen','darkorange','red','lightcoral','slateblue','limegreen','teal','navy']

#base.initplt()
plt.rcParams['figure.figsize'] = [5., 5.]
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
#gs = gridspec.GridSpec(1,1)

teles = args.telescopes.split(",")
feats = args.feats.split(",")
files = []
telestrs = []
featstrs = []
for tele, feat in zip (teles, feats):
    filepath = f"saved_output/inference_{tele}/{feat}*/test_results.csv"
    print(filepath)
    file = glob.glob(filepath)[0]
    files.append(file)
    if tele == 'gmrt50h': telestrs.append('uGMRT 50h')
    elif tele == 'gmrt500h': telestrs.append('uGMRT 500h')
    if tele == 'ska50h': telestrs.append('SKA1-low 50h')
       
   
    if feat == 'noisy': featstrs.append('Method B1')
    elif feat == 'denoised': featstrs.append('Method B2')
    elif feat == 'latent': featstrs.append('Method B3')
    elif feat == 'laten1': featstrs.append('Method B3 - Single LoS')


for ax0,file,telestr,featstr in zip(axes, files, telestrs, featstrs):

   #Plot the x_HI measurements from the Lyα forest
   #ax0.axvspan(0,0.21+0.17,alpha=0.2,color='grey')
   #ax0.text(0.025, -3.82,r'Limit from Ly$\alpha$ data',color='darkgrey',fontsize=fsize_meas)       #Greig et al. 2024, MNRAS, 530, 3208

   print(f"loading result file: {file}")
   all_results = np.loadtxt(file, delimiter=",", skiprows=1)

   xHI_mean = np.reshape(all_results[:,2],(-1,Nsteps))[:,0]
   logfX    = np.reshape(all_results[:,3],(-1,Nsteps))[:,0]
   #print(xHI_mean)
   #print(logfX)

   xHI_mean_post = np.reshape(all_results[:,0],(-1,Nsteps))
   logfX_post = np.reshape(all_results[:,1],(-1,Nsteps))

   print(xHI_mean_post)
   print(logfX_post)

   logfX_infer = np.empty(len(logfX))
   xHI_infer = np.empty(len(logfX))

   for i in range(len(logfX)):
      #Plot the posterior distributions from the MCMC using corner package (Foreman-Mackey 2016, The Journal of Open Source Software, 1, 24)
      pltr.hist2d(xHI_mean_post[i],logfX_post[i],ax=ax0,levels=[1-np.exp(-0.5),1-np.exp(-2.)],
                  plot_datapoints=False,plot_density=False,fill_contours=True,color=colours[i],
                  contour_kwargs={'zorder': 1, 'linewidths': 1.} )#,contourf_kwargs=contkwarg)

      #Read the best fit values from the MCMC
      logfX_infer[i] = np.mean(logfX_post[i])
      xHI_infer[i] = np.mean(xHI_mean_post[i])

      #Plot the best fit and true values
      ax0.scatter(xHI_infer[i],logfX_infer[i],marker='o',s=200,linewidths=1.,color=colours[i],edgecolors='black',alpha=1)
      ax0.scatter(xHI_mean[i],logfX[i],marker='*',s=200,linewidths=1.,color=colours[i],edgecolors='black',alpha=1)

   print('Mock xHI and fX values')
   print(xHI_mean)
   print(logfX)
   print('Inferred xHI and fX values')
   print(xHI_infer)
   print(logfX_infer)

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

   #Make the plot look nice
   ax0.set_xticks(np.arange(0.,0.9,0.2))
   ax0.set_yticks(np.arange(-4.,0.1,1.))
   ax0.set_xlim(0.,1.)
   ax0.set_ylim(-4,-0.4)
   ax0.xaxis.set_minor_locator(AutoMinorLocator())
   ax0.yaxis.set_minor_locator(AutoMinorLocator())

   ax0.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
         ,length=10,width=1,labelsize=fsize)
   ax0.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
         ,length=10,width=1,labelsize=fsize)
   ax0.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True
         ,length=5,width=1)

   #set title
   title = ''
   score_str = rf'$G={g_score:.2f}$'
   if args.titlepref == 'tele':
      title = f'{telestr}, {score_str}'  
   elif args.titlepref == 'feat':
      title = f'{featstr}, {score_str}'
   print(f'tele={tele} feat={feat} title={title}')
   ax0.set_title(title, fontsize=fsize)
   for tick in ax0.xaxis.get_majorticklabels():
      tick.set_horizontalalignment("left")

axes[0].set_ylabel(r'$\log_{10}f_{\mathrm{X}}$', fontsize=fsize)
axes[1].set_xlabel(r'$\langle x_{\rm HI}\rangle$', fontsize=fsize)
legendstr = ''
if args.titlepref == 'tele':
    legendstr = featstrs[0]
if args.titlepref == 'feat':
    legendstr = telestrs[0]
#axes[2].legend(title=r'$z=6$, '+ legendstr, title_fontsize=fsize, frameon=False, loc=(0.43, 0.58))
fig.suptitle(r'$z=6$, '+ legendstr, fontsize=fsize+2)
fig.subplots_adjust(left=0.12, right=0.84, top=0.87, bottom=0.15, wspace=0.02, hspace=0)

#plt.tight_layout()
plt.savefig('%s/posterior_plot_comb.pdf' % ("./tmp_out"), format='pdf', bbox_inches='tight')

plt.show()
plt.close()
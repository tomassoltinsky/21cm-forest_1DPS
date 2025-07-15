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
import corner
import argparse
import plot_results as pltr
from sklearn.metrics import r2_score

from collections import defaultdict

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
parser.add_argument('-p', '--filepath', type=str, default='unet', help='filepath for the test results')
parser.add_argument('-d', '--datapath', type=str, default='unet', help='directory path for the IGM data')
parser.add_argument('-t', '--telescope', type=str, default='uGMRT', help='telescope')
parser.add_argument('-i', '--t_int', type=float, default=500, help='integration time of obsevation in hours')
parser.add_argument('--title', type=str)
parser.add_argument('--format', type=str, default='pdf', help='output image format')
args = parser.parse_args()

#Set parameters describing data
#results_file = 'noisy_ps_f21_inference_ps_train_test_uGMRT_t50.0_20250418124600.csv'
#results_file = 'denoised_ps_f21_inference_ps_train_test_uGMRT_t50.0_20250418124722.csv'
#results_file = 'latent_f21_inference_unet_with_dense_train_test_uGMRT_t50.0_20250413160932.csv'
spec_res = 8
S147 = 64.2
alphaR = -0.44
Nsteps = 100000
Nkbins = 6
d_log_k_bins = 0.25

telescope = args.telescope
tint = args.t_int
#xHI_mean = [0.11,0.80,0.52,0.11,0.80]
#logfX    = [-1.0,-1.0,-2.0,-3.0,-3.0]


#Start plotting
fsize = 20
colours  = ['royalblue','fuchsia','forestgreen','darkorange','red','lightcoral','slateblue','limegreen','teal','navy']
#colours = ['#1A9892', '#44328E', '#9E0142', '#216633', '#08306B']
#colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
#sec_colours = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']
fig = plt.figure(figsize=(8.,8.))
gs = gridspec.GridSpec(1,1)
ax0 = plt.subplot(gs[0,0])
xHI_mean = [0.11,0.80,0.52,0.11,0.80]
logfX    = [-1.0,-1.0,-2.0,-3.0,-3.0]
print(f"loading result file using pattern {args.filepath}")
#print(xHI_mean)
#print(logfX)
logfX_infer = np.empty(len(logfX))
xHI_infer = np.empty(len(logfX))

autocorr_cut = 10000 #cut-off for autocorrelation
nsub = ''
if 'noise_sub' in args.filepath:
    autocorr_cut = 1
    nsub = '_nsub'


for i in range(len(logfX)):

    data = np.load('%s/flatsamp%s_200Mpc_xHI%.2f_fX%.1f_%s_%dkHz_Smin%.1fmJy_alphaR%.2f_t%dh_dk%.2f_%dkbins_%dsteps.npy' % (args.filepath,nsub,xHI_mean[i],logfX[i],telescope,spec_res,S147,alphaR,tint,d_log_k_bins,Nkbins,Nsteps))
    pltr.hist2d(data[autocorr_cut:,0],data[autocorr_cut:,1],ax=ax0,levels=[1-np.exp(-0.5),1-np.exp(-2.)],plot_datapoints=False,
                plot_density=False,fill_contours=True,color=colours[i],contour_kwargs={'zorder': 1, 'linewidths': 1.})#,contourf_kwargs=contkwarg)
    #corner.hist2d(data[:,0],data[:,1],levels=[1-np.exp(-0.5),1-np.exp(-2.)],smooth=True,plot_datapoints=True,plot_density=True,color=colours[i])

    logfX_infer[i] = data[0,1]
    xHI_infer[i] = data[0,0]

for i in range(len(logfX)):
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
#Make the plot look nice
ax0.set_xticks(np.arange(0.,1.1,0.2))
ax0.set_yticks(np.arange(-4.,0.1,1.))
ax0.set_xlim(0.,1.)
ax0.set_ylim(-4,-0.4)
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
#Plot the x_HI measurements from the Lyα forest
colours_lit = ['grey','brown','darkviolet','navy']
rot = 60
"""
ax0.axvspan(0.21-0.07,0.21+0.17,alpha=0.2,color=colours_lit[0])
ax0.text(0.15,-1.8,'Ďurovčíková+24',color=colours_lit[0],rotation=rot,fontsize=12)  #Ďurovčíková et al. 2024, ApJ, 969, 162
ax0.axvspan(0.17-0.11,0.17+0.09,alpha=0.2,color=colours_lit[1])
ax0.text(0.08,-1.8,'Gaikwad+23',color=colours_lit[1],rotation=rot,fontsize=12)      #Gaikwad et al. 2023, MNRAS, 525, 4093
ax0.axvspan(0,0.21,alpha=0.2,color=colours_lit[2])
ax0.text(0.005,-1.8,'Greig+24',color=colours_lit[2],rotation=rot,fontsize=12)       #Greig et al. 2024, MNRAS, 530, 3208
"""
#Complete plotting and save
if args.title is None:
    title = r'%s %d hr, $z=6$' % (telescope,tint)
else:
    title = args.title
plt.title(title, fontsize=fsize)
#Make the plot look nice
        
ax0.set_xticks(np.arange(0.,0.9,0.2))
ax0.set_yticks(np.arange(-4.,0.1,1.))
ax0.set_xlim(0.,1.)
ax0.set_ylim(-4,0.8)
ax0.xaxis.set_minor_locator(AutoMinorLocator())
ax0.yaxis.set_minor_locator(AutoMinorLocator())
ax0.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
        ,length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
        ,length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True
        ,length=5,width=1)

for tick in ax0.xaxis.get_majorticklabels():
    tick.set_horizontalalignment("left")
for tick in ax0.yaxis.get_majorticklabels():
    tick.set_verticalalignment("bottom")

plt.tight_layout()
plt.savefig('%s/posterior_%s.%s' % ("./tmp_out", title,args.format), format=args.format)

plt.show()

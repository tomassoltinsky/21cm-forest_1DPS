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


parser = argparse.ArgumentParser(description='Master posterior plot for all methods and telescopes')

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
fsize = 22
#fsize_meas = 9
colours  = ['royalblue','fuchsia','forestgreen','darkorange','red','lightcoral','slateblue','limegreen','teal','navy']

#base.initplt()
plt.rcParams['figure.figsize'] = [5., 5.]
plt.rcParams['axes.titlesize'] = fsize
plt.rcParams['axes.labelsize'] = fsize
plt.rcParams['xtick.labelsize'] = fsize
plt.rcParams['ytick.labelsize'] = fsize
plt.rcParams['legend.fontsize'] = fsize
fig, axes = plt.subplots(5, 3, figsize=(18, 25), sharey=True, sharex=True)
print(f'axes.shape={axes.shape}')
#gs = gridspec.GridSpec(1,1)

teles = ['gmrt50h', 'gmrt500h', 'ska50h']
feats = ['noisy', 'denoised', 'latent']
folders = ['', 'noise_sub/']
tags = ['', 'nsub_']
autocorr_cut = 1000
NstepsMC = 100000
Nkbins = 6
d_log_k_bins = 0.25

xHI_mean = [0.11,0.80,0.52,0.11,0.80]
logfX    = [-1.0,-1.0,-2.0,-3.0,-3.0]

for i in range(5):
    for j in range(3):
        ax0 = axes[i,j]
        g_score = 0.0
        if i <= 1:
            xHI_true = np.empty(len(xHI_mean))
            logfX_true = np.empty(len(logfX))
            for k in range(len(logfX)):

                data = np.fromfile('../soltinky/21cm-forest_1DPS/datasets/21cmFAST_los/los/los_50Mpc_256_n1000_z6.000_fX%.1f_xHI%.2f.dat' % (logfX[k],xHI_mean[k]),dtype=np.float32)
                logfX_true[k] = data[9]
                xHI_true[k] = data[11]

            contkwarg = dict(alpha=[0.,0.25,0.5])
            telescope, tint = ['uGMRT', 'uGMRT', 'SKA1-low'][j], [50, 500, 50][j]
            folder, tag = folders[i], tags[i]
            print('Telescope: %s, integration time: %d hr' % (telescope,tint))
            logfX_infer = np.empty(len(logfX))
            xHI_infer = np.empty(len(logfX))
            std_xHI = np.empty(len(logfX))
            std_logfX = np.empty(len(logfX))

            for k in range(len(logfX)):
                filename = '../soltinky/21cm-forest_1DPS/MCMC_samples/%sflatsamp_%s200Mpc_xHI%.2f_fX%.1f_%s_%dkHz_Smin%.1fmJy_alphaR%.2f_t%dh_dk%.2f_%dkbins_%dsteps.npy' % (folder,tag,xHI_mean[k],logfX[k],telescope,spec_res,S147,alphaR,tint,d_log_k_bins,Nkbins,NstepsMC)
                print(f'loading {filename}')
                data = np.load(filename)
                pltr.hist2d(data[autocorr_cut:,0],data[autocorr_cut:,1],ax=ax0, levels=[1-np.exp(-0.5),1-np.exp(-2.)],
                            plot_datapoints=False,plot_density=False,fill_contours=True,color=colours[k],
                            contour_kwargs={'zorder': 1, 'linewidths': 1.})#,contourf_kwargs=contkwarg)

                logfX_infer[k] = data[0,1]
                xHI_infer[k] = data[0,0]

                std_xHI[k] = np.std(data[autocorr_cut:,0])
                std_logfX[k] = np.std(data[autocorr_cut:,1])

            for k in range(len(logfX)):
                ax0.scatter(xHI_infer[k],logfX_infer[k],marker='o',s=400,linewidths=1.,color=colours[k],edgecolors='black',alpha=1)
                ax0.scatter(xHI_true[k],logfX_true[k],marker='*',s=400,linewidths=1.,color=colours[k],edgecolors='black',alpha=1)

                """
                print('Mock')
                print(logfX_true)
                print(xHI_true)
                print('Inferred')
                print(xHI_infer)
                print(logfX_infer)
                """

            g_score = np.sqrt(np.mean((xHI_infer-xHI_true)**2+(logfX_infer-logfX_true)**2))
            print('g_score       = %.6f' % g_score)

        if i > 1:
            tele = teles[j]
            feat = feats[i-2]
            filepath = f"saved_output/inference_{tele}/{feat}*/test_results.csv"
            print(filepath)
            file = glob.glob(filepath)[0]

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

            for k in range(len(logfX)):
                #Plot the posterior distributions using corner package (Foreman-Mackey 2016, The Journal of Open Source Software, 1, 24)
                pltr.hist2d(xHI_mean_post[k],logfX_post[k],ax=ax0,levels=[1-np.exp(-0.5),1-np.exp(-2.)],
                            plot_datapoints=False,plot_density=False,fill_contours=True,color=colours[k],
                            contour_kwargs={'zorder': 1, 'linewidths': 1.} )#,contourf_kwargs=contkwarg)

                #Read the best fit values (median)
                logfX_infer[k] = np.median(logfX_post[k])
                xHI_infer[k] = np.median(xHI_mean_post[k])

                #Plot the best fit and true values
                ax0.scatter(xHI_infer[k],logfX_infer[k],marker='o',s=400,linewidths=1.,color=colours[k],edgecolors='black',alpha=1)
                ax0.scatter(xHI_mean[k],logfX[k],marker='*',s=400,linewidths=1.,color=colours[k],edgecolors='black',alpha=1)

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
            
        for tick in ax0.xaxis.get_majorticklabels():
            tick.set_horizontalalignment("left")
        for tick in ax0.yaxis.get_majorticklabels():
            tick.set_verticalalignment("bottom")
        
        
        ax0.text(
            0.38, -3.75,  # x, y position
            score_str,
            fontsize=fsize,
            color='black',
            bbox=dict(
                facecolor='white',   # Background color
                alpha=0.5,           # Transparency (0=fully transparent, 1=opaque)
                edgecolor='black'    # Optional border
            )
        )
        
        
#axes[1,0].setylabel(r'$\log_{10}f_{\mathrm{X}}$', fontsize=fsize)
#axes[2,1].set_xlabel(r'$\langle x_{\rm HI}\rangle$', fontsize=fsize)
fig.supylabel(r'$\log_{10}f_{\mathrm{X}}$', fontsize=fsize)
fig.supxlabel(r'$\langle x_{\rm HI}\rangle$', fontsize=fsize)
axes[0,2].yaxis.set_label_position('right')
axes[0,2].set_ylabel('Method A1', fontsize=fsize)
axes[1,2].yaxis.set_label_position('right')
axes[1,2].set_ylabel('Method A2', fontsize=fsize)
axes[2,2].yaxis.set_label_position('right')
axes[2,2].set_ylabel('Method B1', fontsize=fsize)
axes[3,2].yaxis.set_label_position('right')
axes[3,2].set_ylabel('Method B2', fontsize=fsize)
axes[4,2].yaxis.set_label_position('right')
axes[4,2].set_ylabel('Method B3', fontsize=fsize)
            
axes[0,0].set_title(r'uGMRT$\,50\mathrm{hr}$', fontsize=fsize)
axes[0,1].set_title(r'uGMRT$\,500\mathrm{hr}$', fontsize=fsize)
axes[0,2].set_title(r'SKA1-low$\,50\mathrm{hr}$', fontsize=fsize)

#axes[2].legend(title=r'$z=6$, '+ legendstr, title_fontsize=fsize, frameon=False, loc=(0.43, 0.58))
#plt.suptitle(r'$z=6$', fontsize=fsize+2, y=1.0001)
#fig.subplots_adjust(left=0.12, right=0.84, top=0.87, bottom=0.15, wspace=0.02, hspace=0)
fig.subplots_adjust(wspace=0.02, hspace=0.02)

plt.tight_layout()
plt.savefig('%s/posterior_plot_all_comb.pdf' % ("./tmp_out"), format='pdf', bbox_inches='tight')

plt.show()
plt.close()
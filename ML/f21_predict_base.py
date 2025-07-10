'''
Predict parameters fX and xHI from the 21cm forest data.
'''

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as clr
from sklearn.metrics import r2_score

import argparse
import logging
from datetime import datetime

import os
import sys
import glob
import pickle

import F21DataLoader as dl


logger = logging.getLogger(__name__)


def plot_single_power_spectrum(ps, ks, output_dir=".", showplots=False, saveplots=True, label=""):
    initplt()
    plt.title(f'{label} - power spectrum')
    plt.loglog(ks[1:]*1e6, ps[1:], linewidth=0.5)
    plt.xlabel(r'k (MHz$^{-1}$)')
    plt.ylabel(r'P$_{21}$(k)')
    if showplots: plt.show()
    if saveplots: plt.savefig(f"{output_dir}/power_spectra.pdf", format = "pdf")
    plt.close()

def decide_alpha(num_ps):
    if num_ps >= 1000: return 0.01
    if num_ps <= 10: return 1
    logn = np.log10(num_ps)
    return 1 - (logn - 1) * 0.495

def initplt():
    plt.rcParams['figure.figsize'] = [5., 5.]
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['legend.fontsize'] = 20
    plt.figure()

colorlabels=[r'$\langle x_{HI}\rangle$', r'$log_{10}(f_X)$']
colormaps=[plt.cm.viridis, plt.cm.viridis]
def plot_power_spectra(ps, ks, params, psn=None, colorind=1, output_dir=".", showplots=False, saveplots=True, label="", scale='log', markers=False):
    #logger.info(f'shapes ps:{ps.shape} ks:{ks.shape}')
    initplt()
    alpha = decide_alpha(len(ps))
    logger.info(params[0:2])
    coloraxs = params[:,colorind]
    mincoloraxs = min(coloraxs)
    maxcoloraxs = max(coloraxs)
    print(f"min-max range: {mincoloraxs}-{maxcoloraxs}")
    if markers: marker='o' 
    else: marker = None
    fig, ax = plt.subplots(nrows=1, ncols=1) 

    sm = plt.cm.ScalarMappable(cmap=colormaps[colorind], norm=clr.Normalize(vmin=mincoloraxs, vmax=maxcoloraxs))
    cbar = plt.colorbar(sm, ax=ax, label=colorlabels[colorind])

    plt.title('Power spectra: ' + label)    
    if scale == 'log': 
        ax.set_xscale('log')
        ax.set_yscale('log')

    for i, (row_ps, row_ks, row_coloraxs) in enumerate(zip(ps, ks, coloraxs)):
        color=sm.to_rgba(row_coloraxs)
        #if i%1000==0: print(f"color mapping: {row_coloraxs} : {color}")
        ax.plot(row_ks[1:]*1e6, row_ps[1:], linewidth=0.5, color=color, alpha=alpha, marker= marker)
        #ax.set_yscale('log')
        #if i> 10: break
        # Plot noise curve

    if psn is not None: 
        ax.plot(ks[0,1:]*1e6, psn[1:], "r--", linewidth=0.5, label="Noise")

    plt.legend()
    plt.xlabel(r"k (Hz$^{-1}$)")
    plt.ylabel("$kP_{21}$")
    #plt.ylim((1e-22, 1e-2))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    if showplots: plt.show()
    if saveplots: plt.savefig(f"{output_dir}/power_spectra.pdf", format = "pdf")
    plt.close()

def plot_pca(ps, params, colorind=1, output_dir=".", showplots=False, saveplots=True, label=""):
    #logger.info(f'shapes ps:{ps.shape} ks:{ks.shape}')
    initplt()
    alpha = decide_alpha(len(ps))
    logger.info(params[0:2])
    coloraxs = params[:,colorind]
    mincoloraxs = min(coloraxs)
    maxcoloraxs = max(coloraxs)
    print(f"min-max range: {mincoloraxs}-{maxcoloraxs}")

    fig, ax = plt.subplots(nrows=1, ncols=1) 

    sm = plt.cm.ScalarMappable(cmap=colormaps[colorind], norm=clr.Normalize(vmin=mincoloraxs, vmax=maxcoloraxs))
    cbar = plt.colorbar(sm, ax=ax, label=colorlabels[colorind])

    plt.title('PCA: ' + label)    
    for i, (row_ps, row_coloraxs) in enumerate(zip(ps, coloraxs)):
        color=sm.to_rgba(row_coloraxs)
        ax.plot(row_ps, linewidth=0.5, color=color, alpha=alpha)
    
    plt.xlabel(r"Eigenmode#")
    plt.ylabel("PCA Value")
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    #plt.ylim((-0.00025, None))
    
    if showplots: plt.show()
    if saveplots: plt.savefig(f"{output_dir}/pca.pdf", format = "pdf")
    plt.close()

#statlabels=[]
statlabels=[['ps1', 'ps2', 'total_mean', 'total_std', 'mean_skew', 'std_skew', 'skew2', 'min_skew'],
            ['ps1', 'ps2', 'bis1', 'bis2', 'bis3', 'bis4', 'bis5', 'bis6', 'bis7', 'bis8', 'bis9', 'bis10', 'bis11', 'bis12', 'bis13', 'bis14', 'bis15', 'bis16', 'bis17', 'bis18', 'bis19', 'bis20', 'bis21', 'bis22', 'bis23', 'bis24', 'bis25'],
            ['ps1', 'ps2', 'ps3', 'ps4','ps5', 'ps6','ps7', 'ps8','ps9', 'ps10','ps11', 'ps12','ps13', 'ps14','ps15', 'ps16','total_mean', 'total_std', 'mean_skew', 'std_skew', 'skew2', 'min_skew'],]
markers=['*','P','x','d','v','s','^', '+', 'o', '.', '<', '>', '1', '2','3', '4', 'p', 'h', 'X', 'D', '|', '_']
legendpos=['lower right', 'upper left']
def plot_stats(stats, params, colorind=1, statind=0, output_dir=".", showplots=False, saveplots=True, label=""):
    #logger.info(f'shapes ps:{ps.shape} ks:{ks.shape}')
    initplt()
    alpha = decide_alpha(len(stats))
    logger.info(params[0:2])
    coloraxs = params[:,colorind]
    mincoloraxs = min(coloraxs)
    maxcoloraxs = max(coloraxs)
    print(f"min-max range: {mincoloraxs}-{maxcoloraxs}")

    fig, ax = plt.subplots(nrows=1, ncols=1) 

    num_stats = stats.shape[1]
    colormap = plt.cm.get_cmap('rainbow', num_stats)  # You can choose any colormap you prefer

    plt.title('Statistics: ' + label)
    for i, (row_stat, row_coloraxs) in enumerate(zip(stats, coloraxs)):
        #print(f"row_stat={row_stat}, row_coloraxs={row_coloraxs}")
        #if i%1000==0: print(f"color mapping: {row_coloraxs} : {color}")
        for j, stat in enumerate(row_stat):
            stat_color = colormap(j) 
            label = None
            if i==0 and j < 10: label=statlabels[statind][j]
            ax.scatter(row_coloraxs, stat, color=stat_color, marker=markers[j % len(markers)], alpha=alpha, label=label)
        #ax.set_yscale('log')
        #if i> 10: break
    plt.xlabel(colorlabels[colorind])
    plt.ylabel("Stat Value")
    
    #plt.legend(loc=legendpos[colorind], fontsize=12)  
    plt.legend(loc=legendpos[colorind], ncol=len(statlabels[statind])//2, fontsize=12)
    if showplots: plt.show()
    if saveplots: plt.savefig(f"{output_dir}/statistics.pdf", format = "pdf")
    plt.close()

def plot_single_los(los, freq_axis, output_dir=".", showplots=False, saveplots=True, label=""):
    plt.rcParams['figure.figsize'] = [12, 5]
    plt.title(f'{label} - LoS')
    plt.plot(freq_axis/1e6, los)
    plt.xlabel('frequency[MHz]'), plt.ylabel('flux/S147')
    if showplots: plt.show()
    if saveplots: plt.savefig(f"{output_dir}/los_{label}.pdf", format = "pdf")
    plt.close()

def plot_los(los, freq_axis, output_dir=".", showplots=False, saveplots=True, label=""):
    plt.rcParams['figure.figsize'] = [15, 9]
    plt.title(f'{label} - LoS')
    for i, f in enumerate(los):
        plt.plot(freq_axis/1e6, f)
        if i> 10: break
    plt.xlabel('frequency[MHz]'), plt.ylabel('flux/S147')
    if showplots: plt.show()
    if saveplots: plt.savefig(f"{output_dir}/los.pdf", format = "pdf")
    plt.close()

def plot_predictions(df_y, colors):
    """
    Plot actual points with their corresponding predicted points and contours.
    
    Parameters:
    df_y: DataFrame containing columns: actual_xHI, actual_logfX, pred_xHI, pred_logfX
    """
    df_y['color'] = colors
    plt.rcParams['figure.figsize'] = [15, 9]
    
    # Group by actual points to get corresponding predictions
    for (actual_xHI, actual_logfX), group in df_y.groupby(['actual_xHI', 'actual_logfX']):
        # Get predicted points for this actual point
        pred_points = group[['pred_xHI', 'pred_logfX', 'color']].values
        
        # Create contour of predictions
        if len(pred_points) > 2:  # Need at least 3 points for ConvexHull
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(pred_points)
                # Plot the convex hull boundary
                for simplex in hull.simplices:
                    plt.plot(pred_points[simplex, 0], pred_points[simplex, 1], 
                            'c--', alpha=0.5, linewidth=0.5, c=pred_points[2])
            except:
                pass  # Skip contour if ConvexHull fails
        
        # Plot predicted points
        plt.scatter(pred_points[:, 0], pred_points[:, 1], 
                   c=pred_points[:, 2], alpha=0.5, s=30)
        
        # Plot actual point
        plt.scatter(actual_xHI, actual_logfX, 
                marker='*', s=200)
    
    plt.xlim(0, 1)
    plt.ylim(-4, 1)
    plt.xlabel('xHI')
    plt.ylabel('logfX')
    plt.title('Actual Points (red stars) with Corresponding Predictions (blue dots)')
    
    # Add legend
    plt.scatter([], [], color='lightblue', alpha=0.5, label='Predictions')
    plt.scatter([], [], color='red', marker='*', s=200, label='Actual Points')
    plt.plot([], [], 'c--', alpha=0.5, label='Prediction Boundary')
    plt.legend()
    plt.show()
    plt.close()

def save_test_results(y_pred, y_test, output_dir, label=""):
    """
    Save test results to a CSV file.
    
    Parameters:
    y_pred: numpy array of predictions
    y_test: numpy array of test values
    output_dir: directory to save the file
    label: optional label to add to filename
    """
    filename = f"{output_dir}/test_results{label}.csv"
    
    # Combine predictions and test values
    header = "pred_xHI,pred_logfX,test_xHI,test_logfX"
    combined = np.hstack((y_pred, y_test))
    
    # Save to CSV
    np.savetxt(filename, combined, delimiter=',', header=header, comments='')
    logger.info(f"Saved test results to {filename}")

def load_test_results(filepath):
    """
    Load test results from a CSV file.
    
    Parameters:
    filepath: path to the CSV file
    
    Returns:
    y_pred: numpy array of predictions
    y_test: numpy array of test values
    """
    # Load the data
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    
    # Split into predictions and test values
    y_pred = data[:, :2]  # First two columns are predictions
    y_test = data[:, 2:]  # Last two columns are test values
    
    logger.info(f"Loaded test results from {filepath}")
    return y_pred, y_test

def distance(pred_point, mean_point):
    return (pred_point[0]-mean_point[0])**2 + ((pred_point[1]-mean_point[1])**2)/25.0

def summarize_test(y_pred, y_test, output_dir=".", showplots=False, saveplots=True, label=""):
    logger.info(f"y_pred: {y_pred}")
    logger.info(f"y_test: {y_test}")

    # Calculate R2 scores
    r2 = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(2)]
    logger.info("R2 Score: " + str(r2))
    mean_r2_score = 0.5*(r2[0]+r2[1])
    logger.info("Mean R2 Score: " + str(mean_r2_score))

    # Calculate RMSE scores
    rmse = np.sqrt((y_test - y_pred) ** 2)
    # Weighted as per the range of params (xHI and logfX)
    rmse_comb = rmse[:,0] + rmse[:,1]/5 
    logger.info(f"rmse_comb : {rmse_comb.shape}\n{rmse_comb[:10]}")

    # Create arrays for actual and predicted values
    actual_xHI = y_test[:,0]
    actual_logfX = y_test[:,1]
    pred_xHI = y_pred[:,0]
    pred_logfX = y_pred[:,1]

    # Calculate aggregate RMSE for unique actual points
    unique_points = np.unique(y_test, axis=0)
    agg_rmse = []
    for point in unique_points:
        mask = np.all(y_test == point, axis=1)
        agg_rmse.append(np.mean(rmse_comb[mask]))
    
    logger.info("RMSE Summary:")
    logger.info(f"Mean RMSE: {np.mean(rmse_comb)}")
    logger.info(f"Min RMSE: {np.min(rmse_comb)}")
    logger.info(f"Max RMSE: {np.max(rmse_comb)}")

    if showplots or saveplots:
        # Calculate colors for plotting
        cmap = plt.get_cmap('viridis')
        rmse_min = np.min(rmse_comb)
        rmse_max = np.max(rmse_comb)
        norm = plt.Normalize(rmse_min, rmse_max)
        colors = cmap(norm(rmse_comb))

        # Plot predictions with colored points
        #plot_predictions(np.column_stack((actual_xHI, actual_logfX, pred_xHI, pred_logfX)), colors)
        
        # Create scatter plot
        plt.rcParams['figure.figsize'] = [15, 9]
        fig, ax = plt.subplots()
        plt.scatter(pred_xHI, pred_logfX, marker="o", s=25, label='Predicted', c=colors)
        plt.plot([pred_xHI, actual_xHI], [pred_logfX, actual_logfX], 'r--', alpha=0.2)
        plt.scatter(actual_xHI, actual_logfX, marker="x", s=100, label='Actual', c=colors)
        plt.xlim(0, 1)
        plt.ylim(-4, 1)
        plt.xlabel(r'\textlangle xHI\textrangle', fontsize=18)
        plt.ylabel(r'log_10(f_X)', fontsize=18)
        plt.title('Predictions', fontsize=18)
        plt.legend()
        plt.colorbar(label=f'RMS Error ({rmse_min:.2f} to {rmse_max:.2f})')
        if showplots: plt.show()
        if saveplots: plt.savefig(f'{output_dir}/f21_prediction{label}.pdf", format = "pdf')
        plt.close()
    return mean_r2_score
    
    """
def summarize_test(y_pred, y_test, output_dir=".", showplots=False):
    logger.info(f"y_pred: {y_pred}")
    logger.info(f"y_test: {y_test}")

    # Calculate R2 scores
    r2 = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(2)]
    logger.info("R2 Score: " + str(r2))

    # Calculate rmse scores
    rmse = np.sqrt((y_test - y_pred) ** 2)
    rmse_comb = rmse[:,0]+rmse[:,1]/5 # Weighted as per the range of params
    logger.info(f"rmse_comb : {rmse_comb.shape}\n{rmse_comb[:10]}")
    df_y = pd.DataFrame()
    df_y = df_y.assign(actual_xHI=y_test[:,0])
    df_y = df_y.assign(actual_logfX=y_test[:,1])
    df_y = df_y.assign(pred_xHI=y_pred[:,0])
    df_y = df_y.assign(pred_logfX=y_pred[:,1])
    df_y = df_y.assign(rmse_xHI=rmse[:,0])
    df_y = df_y.assign(rmse_logfx=rmse[:,1])
    df_y = df_y.assign(rmse=rmse_comb)
    logger.info(f"Describing test data with rmse: {df_y.describe()}")

    df_y_agg = df_y.groupby(["actual_xHI", "actual_logfX"])['rmse'].mean()
    df_y_agg.rename('agg_rmse', inplace=True)
    df_y = df_y.merge(df_y_agg, on=['actual_xHI', 'actual_logfX'], validate='many_to_one')
    logger.info(f"Describing data with rmse: \n{df_y.describe()}\n{df_y.head()}")


    rmse_summary = df_y.groupby(["actual_xHI", "actual_logfX"]).agg({'agg_rmse':'mean','rmse_xHI':'mean','rmse_logfx': 'mean'})
    logger.info(f"rmse Summary: \n{rmse_summary}")

    cmap = plt.get_cmap('viridis')
    rmse = df_y['agg_rmse']
    rmse_min = rmse.min()
    rmse_max = rmse.max()
    norm = plt.Normalize(rmse.min(), rmse.max())
    colors = cmap(norm(rmse))    

    plot_predictions(df_y[["actual_xHI", "actual_logfX", "pred_xHI", "pred_logfX"]], colors)
    plt.rcParams['figure.figsize'] = [15, 9]
    fig, ax = plt.subplots()
    plt.scatter(df_y['pred_xHI'], df_y['pred_logfX'], marker="o", s=25, label='Predicted', c=colors)
    plt.plot([df_y['pred_xHI'], df_y['actual_xHI']], [df_y['pred_logfX'], df_y['actual_logfX']], 'r--', alpha=0.2)
    plt.scatter(df_y['actual_xHI'], df_y['actual_logfX'], marker="X", s=100, label='Actual', c=colors)
    plt.xlim(0, 1)
    plt.ylim(-4, 1)
    plt.xlabel('xHI')
    plt.ylabel('logfX')
    plt.title('Predictions')
    plt.legend()
    plt.colorbar(label=f'RMS Error ({rmse_min:.2f} to {rmse_max:.2f})')
    if showplots: plt.show()
    plt.savefig(f'{output_dir}/f21_prediction.pdf", format="pdf')
    plt.clf()
"""

def create_output_dir(args):
    param_str = f'{args.runmode}_{args.telescope}_t{args.t_int}'
    output_dir = f'output/{sys.argv[0].split(sep=os.sep)[-1].rstrip(".py")}_{param_str}_{datetime.now().strftime("%Y%m%d%H%M%S")}'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        print("created " + output_dir)
    return output_dir

def setup_logging(output_dir):
    file_handler = logging.FileHandler(filename=f"{output_dir}/{sys.argv[0].split(sep=os.sep)[-1].rstrip('.py')}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.INFO, 
                        handlers=handlers, 
                        format='%(asctime)s %(levelname)-8s %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)
    logger.info(f"Commandline: {' '.join(sys.argv)}")
    return logger

def setup_args_parser():
    parser = argparse.ArgumentParser(description='Predict reionization parameters from 21cm forest')
    parser.add_argument('-p', '--path', type=str, default='../data/21cmFAST_los/F21_noisy/', help='filepath')
    parser.add_argument('-z', '--redshift', type=float, default=6, help='redshift')
    parser.add_argument('-d', '--dvH', type=float, default=0.0, help='rebinning width in km/s')
    parser.add_argument('-r', '--spec_res', type=int, default=8, help='spectral resolution of telescope (i.e. frequency channel width) in kHz')
    parser.add_argument('-t', '--telescope', type=str, default='uGMRT', help='telescope')
    parser.add_argument('-s', '--s147', type=float, default=64.2, help='intrinsic flux of QSO at 147Hz in mJy')
    parser.add_argument('-a', '--alpha_r', type=float, default=-0.44, help='radio spectral index of QSO')
    parser.add_argument('-i', '--t_int', type=float, default=500, help='integration time of obsevation in hours')
    parser.add_argument('-f', '--log_fx', type=str, default='*', help='log10(f_X)')
    parser.add_argument('-x', '--xHI', type=str, default='*', help='mean neutral hydrogen fraction')
    parser.add_argument('-n', '--nlos', type=int, default=1000, help='num lines of sight')
    parser.add_argument('-m', '--runmode', type=str, default='train_test', help='one of train_test, test_only, grid_search, plot_only')
    parser.add_argument('-b', '--numsamplebatches', type=int, default=1, help='Number of batches of sample data to use for plotting learning curve by sample size.')
    parser.add_argument('--maxfiles', type=int, default=None, help='Max num files to read')
    parser.add_argument('--modelfile', type=str, default="output/cnn-torch-21cmforest-model.pth", help='model file')
    parser.add_argument('--limitsamplesize', type=int, default=1000, help='limit samples from one file to this number.')
    parser.add_argument('--interactive', action='store_true', help='run in interactive mode. show plots as modals.')
    parser.add_argument('--use_saved_los_data', action='store_true', help='load LoS data from pkl file.')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epoch of training.')
    parser.add_argument('--trainingbatchsize', type=int, default=32, help='Size of batch for training.')
    parser.add_argument('--input_points_to_use', type=int, default=2762, help='use the first n points of los. ie truncate the los to first 690 points')
    parser.add_argument('--scale_y', action='store_true', help='Scale the y parameters (logfX).')
    parser.add_argument('--scale_y0', action='store_true', help='Scale the y parameters (xHI).')
    parser.add_argument('--scale_y1', action='store_true', help='Scale logfx and calculate product of logfx with xHI.')
    parser.add_argument('--scale_y2', action='store_true', help='Scale logfx and calculate pythogorean sum of logfx with xHI.')
    parser.add_argument('--logscale_X', action='store_true', help='Log scale the signal strength.')
    parser.add_argument('--channels', type=int, default=1, help='Use multiple channel inputs for the CNN.')
    parser.add_argument('--psbatchsize', type=int, default=None, help='batching for PS data.')
    parser.add_argument('--label', type=str, default='', help='just a descriptive text for the purpose of the run.')
    parser.add_argument('--trials', type=int, default=50, help='Optimization trials')
    parser.add_argument('--use_noise_channel', action='store_true', help='Use noise channel as input.')
    parser.add_argument('--xhi_only', action='store_true', help='calc loss for xhi only')
    parser.add_argument('--logfx_only', action='store_true', help='calc loss for logfx only')
    parser.add_argument('--filter_test', action='store_true', help='Filter test points in important range of xHI')
    parser.add_argument('--filter_train', action='store_true', help='Filter training points in important range of xHI')

    return parser

test_points = [[-3.00,0.11],[-1.00,0.11],[-2.00,0.52], [-3.00,0.80],[-1.00,0.80]]#,[0.00,0.80]]

def get_datafile_list(type, args, extn='dat', filter=None, override_path=None):
    path = args.path
    if override_path is not None:
        path = override_path
    if type == 'noisy':
        filepattern = str('%sF21_noisy_21cmFAST_200Mpc_z%.1f_fX%s_xHI%s_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f.%s' % 
               (path, args.redshift,args.log_fx, args.xHI, args.telescope, args.spec_res, args.t_int, args.s147, args.alpha_r, extn))
    elif type == 'signalonly':
        filepattern = str('%sF21_signalonly_21cmFAST_200Mpc_z%.1f_fX%s_xHI%s_%dkHz.%s' % 
               (path, args.redshift,args.log_fx, args.xHI, args.spec_res, extn))
    elif type == 'noiseonly':
        filepattern = str('%sF21_noiseonly_21cmFAST_200Mpc_z%.1f_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f.%s' % 
               (path, args.redshift,args.telescope, args.spec_res, args.t_int, args.s147, args.alpha_r, extn))


    logger.info(f"Loading files with pattern {filepattern}")

    datafiles = glob.glob(filepattern)

    # apply filter if specified
    if filter is not None:
        train_files = []
        test_files = []
        for file in datafiles:
            is_test_file = False
            for p in test_points:
                if file.find(f"fX{p[0]:.2f}_xHI{p[1]:.2f}") >= 0:
                    test_files.append(file)
                    is_test_file = True
                    break
            if not is_test_file:
                train_files.append(file)
        if filter == 'test_only': datafiles = test_files
        elif filter == 'train_only': datafiles = train_files

    datafiles = sorted(datafiles) # sorting is important because we want to reliably reproduce the test results 

    if args.maxfiles is not None:
        datafiles = datafiles[:args.maxfiles]

    logger.info(f"Found {len(datafiles)} files matching pattern")

    return datafiles


def load_dataset(datafiles, psbatchsize, limitsamplesize, save=False, skip_ps=True, max_workers=8, shuffle_samples = False):
    # Lists to store combined data
    all_params = []
    # Create processor with desired number of worker threads
    processor = dl.F21DataLoader(max_workers=max_workers, psbatchsize=psbatchsize, limitsamplesize=limitsamplesize, skip_ps=skip_ps, shuffle_samples=shuffle_samples)

    # Process all files and get results
    results = processor.process_all_files(datafiles)
    logger.info(f"Finished data loading.")
    # Access results
    keys = results['key']
    all_los = results['los']
    los_samples = results['los_samples']
    all_params = results['params']
    freq_axis = results['freq_axis']
    logger.info(f"sample los:{all_los[0]}")
    logger.info(f"sample params:{all_params[0]}")
    
    # Combine all data
    logger.info(f"\nCombined data shape: {all_los.shape}")
    logger.info(f"Combined parameters shape: {all_params.shape}")
        
    if save:
        logger.info(f"Saving LoS data to file")
        with open('los-21cm-forest.pkl', 'w+b') as f:  # open a text file
            pickle.dump({"all_los": all_los, "all_params": all_params}, f)
            
    return (all_los, all_params, los_samples, keys, freq_axis)

def test_multiple(modeltester, datafiles, reps=10000, size=10, save=False, skip_stats=True, use_bispectrum=False, skip_ps=False, so_datafiles=None, input_points_to_use=None, ps_bins=None, perc_bins_to_use=100):
    logger.info(f"Test_multiple started. {reps} reps x {size} points will be tested for {len(datafiles)} parameter combinations")
    # Create processor with desired number of worker threads
    all_y_test = []
    all_y_pred = []
    # Process all files and get results
    for i, f in enumerate(datafiles):
        if i==0: logger.info(f"Working on param combination #{i+1}: {f.split('/')[-1]}")
        processor = dl.F21DataLoader(max_workers=8, psbatchsize=1, limitsamplesize=None, ps_bins=ps_bins, skip_ps=skip_ps, skip_stats=skip_stats, use_bispectrum=use_bispectrum, input_points_to_use=input_points_to_use, perc_bins_to_use=perc_bins_to_use)
        results = processor.process_all_files([f])        
        # Access results
        los = results['los']
        ks = results['ks']
        ps = results['ps']
        stats = results['stats']
        bispectrum = results['bispectrum']
        if i==0 and bispectrum is not None: logger.info(f"bispectrum.shape={bispectrum.shape}")

        k_bispec = results['k_bispec']
        params = results['params']
        los_so = None
        if so_datafiles is not None:
            #logger.info(f"Loading signalonly los")
            processor = dl.F21DataLoader(max_workers=8, psbatchsize=1, limitsamplesize=None, ps_bins=ps_bins, skip_ps=skip_ps, skip_stats=skip_stats, use_bispectrum=use_bispectrum, input_points_to_use=input_points_to_use, perc_bins_to_use=perc_bins_to_use)
            results = processor.process_all_files([so_datafiles[i]])        
            # Access results
            los_so = results['los']

        #if i == 0: logger.info(f"sample test los_so:{los[:1]}")
        y_pred_for_test_point = []
        y_test = None
        for j in range(reps):
            #pick 10 samples
            rdm = np.random.randint(len(los), size=size)
            los_set, ps_set, stats_set = None, None, None
            if los is not None: los_set = los[rdm]
            if ps is not None: ps_set = ps[rdm]
            if stats is not None: stats_set = stats[rdm]
            if bispectrum is not None: bispectrum_set = bispectrum[rdm]
            if i==0 and j==0 and bispectrum is not None: logger.info(f"bispectrum_set.shape={bispectrum_set.shape}")
            params_set = params[rdm]
            los_so_set = None
            if los_so is not None: los_so_set = los_so[rdm]
            _, y_test, y_pred, r2 = modeltester.test(los_set, ps_set, stats_set, bispectrum_set, params_set, los_so=los_so_set, silent=(j!=0))
            y_pred_mean = np.mean(y_pred, axis=0)
            all_y_pred.append(y_pred_mean)
            all_y_test.append(params[0])
        logger.info(f"Test_multiple: param combination:{params[0]} predicted mean:{y_pred_mean}")
        if i==0: 
            logger.info(f"Test_multiple: param combination min, max should be the same:{np.min(params, axis=0)}, {np.min(params, axis=0)}")

    y_test_np = np.array(all_y_test)
    y_pred_np = np.array(all_y_pred)
    logger.info(f"Test_multiple completed. actual shape {y_test_np.shape} predicted shape {y_pred_np.shape}")

    return y_pred_np, y_test_np


def scaleXy(X, y, args):
    if args.scale_y: 
        xHI = y[:, 0].reshape(len(y), 1)
        scaledfx = (0.8 + y[:,1]/5.0).reshape(len(y), 1)
        y = np.hstack((xHI, scaledfx))
    if args.scale_y0: y[:,0] = y[:,0]*5.0
    if args.scale_y1:
        # we wish to create a single metric representing the expected
        # strength of the signal based on xHI (0 to 1) and logfx (-4 to +1)
        # We know that higher xHI and lower logfx lead to a stronger signal, 
        # First we scale logfx to range of 0 to 1.
        # Then we take a product of xHI and (1 - logfx)
        if args.trials == 1: logger.info(f"Before scaleXy: {y}")
        xHI = y[:, 0].reshape(len(y), 1)
        scaledfx = 1 - (0.8 + y[:,1]/5.0)
        product = np.sqrt(xHI**2 + scaledfx**2).reshape(len(y), 1)
        y = np.hstack((xHI, scaledfx, product))
        if args.trials == 1: logger.info(f"ScaledXy: {y}")
    if args.scale_y2:
        # we wish to create a single metric representing the expected
        # strength of the signal based on xHI (0 to 1) and logfx (-4 to +1)
        # We know that higher xHI and lower logfx lead to a stronger signal, 
        # First we scale logfx to range of 0 to 1.
        # Then we take a product of xHI and (1 - logfx)
        if args.trials == 1: logger.info(f"Before scaleXy: {y}")
        xHI = y[:, 0].reshape(len(y), 1)
        scaledfx = 1 - (0.8 + y[:,1]/5.0).reshape(len(y), 1)
        product = np.sqrt(xHI**2 + scaledfx**2).reshape(len(y), 1)
        y = np.hstack((xHI, scaledfx, product))
    if args.logscale_X: X = np.log(X)
    return X, y

def unscaleXy(X, y, args):
    # Undo what we did in scaleXy function
    if args.scale_y: 
        xHI = y[:, 0].reshape(len(y), 1)
        fx = 5.0*(y[:,1] - 0.8).reshape(len(y), 1)
        y = np.hstack((xHI, fx))
    if args.scale_y0: y[:,0] = y[:,0]/5.0
    if args.scale_y1:
        xHI = y[:, 0].reshape(len(y), 1)
        fx = 5.0*(1 - y[:,1] - 0.8)
        y = np.hstack((xHI, fx))
    if args.scale_y2:
        xHI = y[:, 0].reshape(len(y), 1)
        fx = 5.0*(1 - y[:,1] - 0.8).reshape(len(y), 1)
        y = np.hstack((xHI, fx))
                
    if args.logscale_X: X = np.exp(X)
    return X, y

def unscale_y(y, args):
    # Undo what we did in the scaleXy function
    if args.scale_y: 
        xHI = y[:, 0].reshape(len(y), 1)
        fx = 5.0*(y[:,1] - 0.8).reshape(len(y), 1)
        y = np.hstack((xHI, fx))
    if args.scale_y0: y[:,0] = y[:,0]/5.0
    if args.scale_y1:
        # calculate fx using product and xHI 
        xHI = np.sqrt(y[:,2]**2 - y[:,1]**2)
        logfx = 5.0*(1 - y[:,1] - 0.8)
        y = np.hstack((xHI, fx))
    if args.scale_y2:
        # calculate fx using product and xHI 
        xHI = np.sqrt(0.5*y**2).reshape((len(y), 1))
        fx = 5.0*(1 - xHI - 0.8)
        y = np.hstack((xHI, fx))
    
    return y



    

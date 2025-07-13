import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.colors as clr
from sklearn.metrics import r2_score

import argparse
import logging
from datetime import datetime
import f21_predict_base as base
import PS1D
import F21Stats as f21stats

from matplotlib.colors import LinearSegmentedColormap, colorConverter
try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None
import colorsys

logger = logging.Logger("main")


def split_key(key):
    # Split the key into two float values
    xHI, logfX = map(float, key.split('_'))
    return xHI, logfX

def create_key(xHI, logfX):
    return f"{xHI:.2f}_{logfX:.2f}" 


selected_keys = ["0.80_-3.00","0.11_-3.00","0.52_-2.00","0.80_-1.00","0.11_-1.00"]
def calc_squared_error(predictions, y_test):
    # Create keys for each row
    keys = [create_key(y_test[i][0], y_test[i][1]) for i in range(len(y_test))]
    
    # Calculate mean predictions for each unique key
    unique_keys = set(keys)
    mean_predictions = {key: [] for key in unique_keys}
    
    for i, key in enumerate(keys):
        mean_predictions[key].append(predictions[i])
    
    mean_values = {key: np.mean(values, axis=0) for key, values in mean_predictions.items()}
    
    # Calculate squared error
    total_squared_error = 0
    for i, key in enumerate(selected_keys):
        xHI, logfX = split_key(key)
        squared_error = (xHI - mean_values[key][0])**2 + (logfX - mean_values[key][1])**2 
        logger.info(f"key: {key}, mean_values: {mean_values[key]}, squared_error: {squared_error}")
        total_squared_error += squared_error
    
    logger.info(f"Total Squared Error (Means): {total_squared_error}")
    mse_means = total_squared_error/len(selected_keys)
    logger.info(f"MSE (Means): {mse_means}")
    rmse_means = np.sqrt(mse_means)
    logger.info(f"RMSE (Means): {rmse_means}")
    return total_squared_error, rmse_means

def mean_squared_error(predictions, y_test):
    mse = calc_squared_error(predictions, y_test)/5.0
    return mse

def rmse_all(y_pred, y_test):
    rmse_all = np.sqrt(np.mean((y_test - y_pred) ** 2))
    logger.info(f"RMSE (all predictions): {rmse_all}")
    return rmse_all

def summarize_test_1000(y_pred, y_test, output_dir=".", showplots=False, saveplots=True, label=""):
    """
    Analyze predictions by grouping them for each unique test point.
    
    Parameters:
    y_pred: numpy array of predictions (n, 2)
    y_test: numpy array of test values (n, 2)
    """
    logger.info(f"summarize_test_1000: Summarizing results pred shape:{y_pred.shape} actual shape: {y_test.shape}")

    # Create unique identifier for each test point
    unique_test_points = np.unique(y_test[:,:2], axis=0)
    sorted_indices_lex = np.lexsort((unique_test_points[:, 0], unique_test_points[:, 1]))
    unique_test_points = unique_test_points[sorted_indices_lex]    
    print(f"Number of unique test points: {len(unique_test_points)}")
    print(f"Unique test points: {unique_test_points}")
    
    # Calculate mean predictions for each unique test point
    mean_predictions = []
    std_predictions = []
    
    for test_point in unique_test_points:
        # Find all predictions corresponding to this test point
        mask = np.all(y_test == test_point, axis=1)
        corresponding_preds = y_pred[mask]
        logger.info(f"Test point: {test_point}, Number of preds: {len(corresponding_preds)}")

        # Calculate mean and std of predictions
        mean_pred = np.mean(corresponding_preds, axis=0)
        std_pred = np.std(corresponding_preds, axis=0)
        logger.info(f"Test point: {test_point}, Number of preds: {len(corresponding_preds)}, Mean: {mean_pred}, Std: {std_pred}")
        
        mean_predictions.append(mean_pred)
        std_predictions.append(std_pred)
    
    mean_predictions = np.array(mean_predictions)
    std_predictions = np.array(std_predictions)
    
    # Calculate R2 score using mean predictions
    r2_means = [r2_score(unique_test_points[:, i], mean_predictions[:, i]) for i in range(2)]
    r2_means_combined = np.mean(r2_means)
    logger.info(f"R2 Score (using means): {r2_means}, combined score: {r2_means_combined}")
    r2_total = r2_score(y_test, y_pred)
    logger.info(f"R2 Score for All Predictions: {r2_total}")
    tse, rmse_means =  calc_squared_error(y_pred, y_test)
    rmse_total = rmse_all(y_pred, y_test)
    
    # Plotting
    if showplots or saveplots:
        plt.rcParams['figure.figsize'] = [8, 8]
        fig, ax = plt.subplots()
        
        num_points = len(unique_test_points)
        colors = plt.cm.rainbow(np.linspace(0, 1, num_points))
        
        # For each unique test point, create contours of predictions
        for i, test_point in enumerate(unique_test_points):
            # Find all predictions corresponding to this test point
            mask = np.all(y_test == test_point, axis=1)
            corresponding_preds = y_pred[mask]

            x, y = corresponding_preds[:, 0], corresponding_preds[:, 1]
            # Step 2: Normalize the histogram to get the probability density
            # Get the bin counts and edges
            counts, xedges, yedges = np.histogram2d(x, y, bins=18)
            # Normalize the counts to create a probability density function
            pdf = counts / np.sum(counts)
            # Find the levels that correspond to 68% and 95% confidence intervals
            level_68 = np.percentile(pdf, 68)
            level_95 = np.percentile(pdf, 95)
            # Step 4: Create the contour plot
            plt.contourf(xedges[:-1], yedges[:-1], pdf.T, levels=[level_68,level_95,np.max(pdf)],colors=[colors[i],colors[i]],alpha=[0.3,0.6])
            plt.contour(xedges[:-1], yedges[:-1], pdf.T, levels=[level_68,level_95,np.max(pdf)],colors=[colors[i],colors[i]],linewidths=0.5)
            plt.xlim(0,1)
            plt.ylim(-4,0)
            plt.tick_params(axis='both', direction='in', length=10)  # Inward ticks with length 10
            plt.xlabel(r"$\langle x_{HI}\rangle$")
            plt.ylabel(r"$log_{10}(f_X)$")


        # Plot mean predictions
        plt.scatter(mean_predictions[:, 0], mean_predictions[:, 1], 
                   marker="x", s=200, label='Mean Predicted', alpha=1, c=colors)

        # Plot actual points
        plt.scatter(unique_test_points[:, 0], unique_test_points[:, 1], 
                   marker="*", s=200, label='Actual', c=colors)
        
        plt.xlim(0, 1)
        plt.ylim(-4, 1)
        
        plt.xlabel(r'$\langle x_{HI}\rangle$', fontsize=18)
        plt.ylabel(r'$log_{10}(f_X)$', fontsize=18)
        plt.yticks(fontsize=18)
        plt.xticks(fontsize=18)
        plt.title(f'Mean Predictions with  ±1σ and ±2σ Contours {label}', fontsize=18)

        # Overlay RMSE, R2 means, and R2 total on the graph
        textstr = f'RMSE: {rmse_means:.4f}\nR²: {r2_means_combined:.4f}'
        props = dict(facecolor='white', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=18,
                verticalalignment='top', bbox=props)

        plt.legend(loc='upper right')
        
        if saveplots: plt.savefig(f'{output_dir}/f21_prediction_means_contours.pdf', format='pdf')
        if showplots: plt.show()
        plt.close()

        # Make a scatter plot
        plt.rcParams['figure.figsize'] = [8, 8]
        fig, ax = plt.subplots()
        colors = plt.cm.rainbow(np.linspace(0, 1, num_points))
        # Plot all 10000 predctions
        for i, test_point in enumerate(unique_test_points):
            # Find all predictions corresponding to this test point
            mask = np.all(y_test == test_point, axis=1)
            corresponding_preds = y_pred[mask]
            plt.scatter(corresponding_preds[:, 0], corresponding_preds[:, 1], 
                marker="o", s=25, alpha=0.01, c=colors[i])
            
        # Plot mean predictions
        plt.scatter(mean_predictions[:, 0], mean_predictions[:, 1], 
                marker="o", edgecolor='b', s=100, label='Mean Predicted', alpha=1, c=colors)
        # Plot actual points
        plt.scatter(unique_test_points[:, 0], unique_test_points[:, 1], 
                marker="*", edgecolor='b', s=200, label='Actual', c=colors)

        plt.xlim(0, 1)
        plt.ylim(-4, 1)

        plt.xlabel(r'$\langle x_{HI}\rangle$', fontsize=18)
        plt.ylabel(r'$log_{10}(f_X)$', fontsize=18)
        plt.yticks(fontsize=18)
        plt.xticks(fontsize=18)
        plt.title(f'Inference {label}', fontsize=18)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=18,
                verticalalignment='top', bbox=props)
        plt.legend(loc='upper right')

        if saveplots: plt.savefig(f'{output_dir}/f21_prediction_means_scatter.pdf', format='pdf')
        if showplots: plt.show()
        plt.close()
    # Log statistics
    logger.info("\nPrediction Statistics:")
    logger.info(f"Mean xHI std: {np.mean(std_predictions[:, 0]):.4f}")
    logger.info(f"Mean logfX std: {np.mean(std_predictions[:, 1]):.4f}")
    
    return r2_means_combined

markers=['o', 'x', '*']
def plot_power_spectra(ps_set, ks, title, labels, xscale='log', yscale='log', showplots=False, saveplots=True, output_dir='tmp_out'):
    #print(f"plot_power_spectra: shapes: {ps_set.shape},{ks.shape}")

    base.initplt()
    plt.title(f'{title}')
    if len(ps_set.shape) > 1:
        for i, ps in enumerate(ps_set):

            if labels is not None: label = labels[i]
            row_ks = None
            if ks is not None:
                if len(ks.shape) > 1: row_ks = ks[i]
                else: row_ks = ks
            plt.plot(row_ks*1e6, ps, label=label, marker=markers[i% len(markers)], alpha=0.5)
    else:
        row_ks = None
        if ks is not None:
                if len(ks.shape) > 1: row_ks = ks[0]
                else: row_ks = ks
        plt.plot(ks*1e6, ps, label=label, marker='o')
        #plt.scatter(ks[1:]*1e6, ps[1:], label=label)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xlabel(r'k (Hz$^{-1}$)')
    plt.ylabel(r'$kP_{21}$')
    plt.legend()
    if saveplots: plt.savefig(f"{output_dir}/reconstructed_ps_{title}.pdf", format="pdf")
    if showplots: plt.show()
    plt.close()

def plot_denoised_ps(los_test, y_test_so, y_pred_so, samples=1, showplots=False, saveplots=True, label='', signal_bandwidth=20473830.8, output_dir='tmp_out'):
    ks_noisy, ps_noisy = PS1D.get_P_set(los_test, signal_bandwidth, scaled=True)
    #logger.info(f'get_P_set: {ks_noisy.shape}, {ps_noisy.shape},')
    ks_noisy, ps_noisy = f21stats.logbin_power_spectrum_by_k(ks_noisy, ps_noisy)
    #logger.info(f'get_P_set: {ks_noisy.shape}, {ps_noisy.shape},')
    ps_noisy_mean = np.mean(ps_noisy, axis=0)
    ks_so, ps_so = PS1D.get_P_set(y_test_so, signal_bandwidth, scaled=True)
    ks_so, ps_so = f21stats.logbin_power_spectrum_by_k(ks_so, ps_so)
    ps_so_mean = np.mean(ps_so, axis=0)
    ks_pred, ps_pred = PS1D.get_P_set(y_pred_so, signal_bandwidth, scaled=True)
    ks_pred, ps_pred = f21stats.logbin_power_spectrum_by_k(ks_pred, ps_pred)
    ps_pred_mean = np.mean(ps_pred, axis=0)

    plot_power_spectra(np.vstack((ps_so_mean,ps_noisy_mean,ps_pred_mean)), ks_noisy[0,:], title=label, labels=["signal-only", "noisy-signal", "reconstructed"], output_dir=output_dir)

def plot_denoised_los(los_test, y_test_so, y_pred_so, samples=1, showplots=False, saveplots=True, label='', output_dir='tmp_out', freq_axis=None):
    
    for i, (noisy, test, pred) in enumerate(zip(los_test[:samples], y_test_so[:samples], y_pred_so[:samples])):
        if freq_axis is None: freq_axis=range(len(noisy))
        base.initplt()
        plt.title(f'{label}')
        chisq_noisy = np.sum((noisy - test)**2 / test)
        plt.plot(freq_axis, noisy, label=f'Signal+Noise: χ²={chisq_noisy:.2f}', c='black', linewidth=0.5)
        plt.plot(freq_axis, test, label='Signal', c='orange')
        chisq_denoised = np.sum((pred - test)**2 / test)
        plt.plot(freq_axis, pred+0.1, label=f'Denoised+0.1: χ²={chisq_denoised:.2f}')
        plt.xlabel(r'$\nu_{obs}$[MHz]'), 
        plt.ylabel(r'$F_{21}=e^{-\tau_{21}}$')
        #plt.legend(loc='best')#lower right')
        if saveplots: 
            plt.savefig(f"{output_dir}/reconstructed_los_{label}.pdf", format="pdf", bbox_inches='tight')
            logger.info(f"Saved denoised los plot to {output_dir}/reconstructed_los_{label}.png")
        if i> 5: break
        if showplots: plt.show()
        print(f'denoising {label}: χ²={chisq_noisy:.2f} χ²={chisq_denoised:.2f}')
        plt.close()


def calculate_chisq_tensor(predictions, targets, epsilon=1e-10):
    """
    Calculate chi-squared between predictions and targets for PyTorch tensors.
    
    Parameters:
    predictions (torch.Tensor): Predicted values
    targets (torch.Tensor): Target/true values
    epsilon (float): Small value to prevent divide by zero
    
    Returns:
    torch.Tensor: Chi-squared value
    """
    # Ensure inputs are tensors
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)
    
    # Calculate chi-squared with divide by zero handling
    chisq = torch.sum((predictions - targets)**2 / (targets + epsilon))
    return chisq

def log_cosh_loss(predictions, targets):
    """
    Calculate log cosh loss between predictions and targets.
    This is a smooth approximation of the Huber loss.
    
    Parameters:
    predictions (torch.Tensor): Predicted values
    targets (torch.Tensor): Target/true values
    
    Returns:
    torch.Tensor: Log cosh loss value
    """
    # Ensure inputs are tensors
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)
    
    # Calculate log cosh loss
    error = predictions - targets
    loss = torch.log(torch.cosh(error))
    return torch.mean(loss)

"""Sample plotting code"""
if __name__ == "__main__":
    all_results = np.loadtxt("saved_output/unet_inference/test_results.csv", delimiter=",", skiprows=1)
    print(f"loaded data shape: {all_results.shape}")
    y_pred = all_results[:,:2]
    y_test = all_results[:,2:4]
    summarize_test_1000(y_pred, y_test, output_dir="./tmp_out", showplots=True, saveplots=True, label="test")

def lighten_color(rgba_color, amount=0.15):
    """
    Lighten a color by a given amount.
    """
    rgba_color_list = list(rgba_color)
    # Convert RGB to HLS
    h, l, s = colorsys.rgb_to_hls(*rgba_color_list[:3])

    # Increase lightness (e.g., by 20%)
    # Ensure lightness stays within [0, 1]
    new_l = min(1.0, l + amount) 

    # Convert HLS back to RGB
    lightened_rgb = colorsys.hls_to_rgb(h, new_l, s)
    return lightened_rgb + (rgba_color[3],)

# Copied from corner.py, modified for some features
def hist2d(x, y, bins=20, range=None, weights=None, levels=None, smooth=None,
           ax=None, color=None, plot_datapoints=True, plot_density=True,
           plot_contours=True, no_fill_contours=False, fill_contours=False,
           contour_kwargs=None, contourf_kwargs=None, data_kwargs=None, lighten_fill=False,
           **kwargs):
    """
    Plot a 2-D histogram of samples.

    Parameters
    ----------
    x, y : array_like (nsamples,)
       The samples.

    levels : array_like
        The contour levels to draw.

    ax : matplotlib.Axes (optional)
        A axes instance on which to add the 2-D histogram.

    plot_datapoints : bool (optional)
        Draw the individual data points.

    plot_density : bool (optional)
        Draw the density colormap.

    plot_contours : bool (optional)
        Draw the contours.

    no_fill_contours : bool (optional)
        Add no filling at all to the contours (unlike setting
        ``fill_contours=False``, which still adds a white fill at the densest
        points).

    fill_contours : bool (optional)
        Fill the contours.

    contour_kwargs : dict (optional)
        Any additional keyword arguments to pass to the `contour` method.

    contourf_kwargs : dict (optional)
        Any additional keyword arguments to pass to the `contourf` method.

    data_kwargs : dict (optional)
        Any additional keyword arguments to pass to the `plot` method when
        adding the individual data points.
    """
    if ax is None:
        ax = plt.gca()
    
    if np.all(np.isnan(x)) or np.all(np.isnan(y)):
        return
    
    # Set the default range based on the data range if not provided.
    if range is None:
        if "extent" in kwargs:
            logging.warn("Deprecated keyword argument 'extent'. "
                         "Use 'range' instead.")
            range = kwargs["extent"]
        else:
            range = [[x.min(), x.max()], [y.min(), y.max()]]

    # Set up the default plotting arguments.
    if color is None:
        color = "k"

    # Choose the default "sigma" contour levels.
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

    # This is the color map for the density plot, over-plotted to indicate the
    # density of the points near the center.
    density_cmap = LinearSegmentedColormap.from_list(
        "density_cmap", [color, (1, 1, 1, 0)])

    # This color map is used to hide the points at the high density areas.
    white_cmap = LinearSegmentedColormap.from_list(
        "white_cmap", [(1, 1, 1), (1, 1, 1)], N=2)

    # This "color map" is the list of colors for the contour levels if the
    # contours are filled.
    rgba_color = colorConverter.to_rgba(color)
    if lighten_fill:
        rgba_color = lighten_color(rgba_color, 0.1)
    contour_cmap = [list(rgba_color) for l in levels] + [list(rgba_color)]
    for i, l in enumerate(levels):
        contour_cmap[i][-1] *= float(i) / (len(levels)+1)
        print(f'contour_cmap[{i}][-1]= {contour_cmap[i][-1]}')

    # We'll make the 2D histogram to directly estimate the density.
    try:
        H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=bins,
                                 range=range, weights=weights)
    except ValueError:
        raise ValueError("It looks like at least one of your sample columns "
                         "have no dynamic range. You could try using the "
                         "'range' argument.")

    if smooth is not None:
        if gaussian_filter is None:
            raise ImportError("Please install scipy for smoothing")
        H = gaussian_filter(H, smooth)

    # Compute the density levels.
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]
    V.sort()
    m = np.diff(V) == 0
    if np.any(m):
        logging.warning("Too few points to create valid contours")
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = np.diff(V) == 0
    V.sort()

    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    # Extend the array for the sake of the contours at the plot edges.
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate([
        X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
        X1,
        X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
    ])
    Y2 = np.concatenate([
        Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
        Y1,
        Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
    ])

    if plot_datapoints:
        if data_kwargs is None:
            data_kwargs = dict()
        data_kwargs["color"] = data_kwargs.get("color", color)
        data_kwargs["ms"] = data_kwargs.get("ms", 2.0)
        data_kwargs["mec"] = data_kwargs.get("mec", "none")
        data_kwargs["alpha"] = data_kwargs.get("alpha", 0.1)
        ax.plot(x, y, "o", zorder=-1, rasterized=True, **data_kwargs)

    # Plot the base fill to hide the densest data points.
    if (plot_contours or plot_density) and not no_fill_contours:
        ax.contourf(X2, Y2, H2.T, [V.min(), H.max()],
                    cmap=white_cmap, antialiased=False)

    if fill_contours:
        if contourf_kwargs is None:
            contourf_kwargs = dict()
        contourf_kwargs["colors"] = contourf_kwargs.get("colors", contour_cmap)
        contourf_kwargs["antialiased"] = contourf_kwargs.get("antialiased",
                                                             False)
        ax.contourf(X2, Y2, H2.T, np.concatenate([[0], V, [H.max()*(1+1e-4)]]),
                    **contourf_kwargs)

    # Plot the density map. This can't be plotted at the same time as the
    # contour fills.
    elif plot_density:
        ax.pcolor(X, Y, H.max() - H.T, cmap=density_cmap)

    # Plot the contour edge colors.
    if plot_contours:
        if contour_kwargs is None:
            contour_kwargs = dict()
        contour_kwargs["colors"] = contour_kwargs.get("colors", color)
        ax.contour(X2, Y2, H2.T, V, **contour_kwargs)

    ax.set_xlim(range[0])
    ax.set_ylim(range[1])

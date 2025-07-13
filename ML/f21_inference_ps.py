'''
Use Powerspectrum data to infer cosmological parameters.
We use the powerspectrum computed on noisy data simulation.
XGBoostRegressor is trained based on labelled data of 529 parameter combinations. 
The same is then tested on 5 parameter combinations specially selected as test points.
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import argparse
import glob
from datetime import datetime

import F21DataLoader as dl
import f21_predict_base as base
import F21Stats as f21stats
import plot_results as pltr
import Scaling
import PS1D
import F21Stats as f21stats

import numpy as np
import sys

import matplotlib.pyplot as plt

import optuna
from xgboost import XGBRegressor
import os

def load_training_data(override_path, samples, args):
    files = base.get_datafile_list('noisy', args, extn='csv', override_path=override_path)
    numgroups = samples//10
    X_train = np.zeros((numgroups*len(files), 16))
    y_train = np.zeros((numgroups*len(files), 2))
    
    for i, file in enumerate(files):
        curr_xHI = float(file.split('xHI')[1].split('_')[0])
        curr_logfX = float(file.split('fX')[1].split('_')[0])
        y_train[i*numgroups:(i+1)*numgroups, 0] = curr_xHI
        y_train[i*numgroups:(i+1)*numgroups, 1] = curr_logfX
        currps = np.loadtxt(file)[:samples,:16]
        currps_grouped = currps.reshape(-1, 10, currps.shape[1]).mean(axis=1)

        if i == 0:
            print(f"Original array shape: {currps.shape}")
            print(f"Shape after grouping and taking mean: {currps_grouped.shape}")
            print(f"currps sample:\n{currps[:10,2]}")
            print(f"currps sample grouped:\n{currps_grouped[0][3]}")
        X_train[i*numgroups:(i+1)*numgroups, :] = currps_grouped[:,:]
    return X_train, y_train

def load_test_data(override_path, samples, args):
    files = base.get_datafile_list('noisy', args, extn='csv', override_path=override_path)
    X_test = np.zeros((10000*len(files), 16))
    y_test = np.zeros((10000*len(files), 2))
    for i, file in enumerate(files):
        curr_xHI = float(file.split('xHI')[1].split('_')[0])
        curr_logfX = float(file.split('fX')[1].split('_')[0])
        y_test[i*10000:(i+1)*10000, 0] = curr_xHI
        y_test[i*10000:(i+1)*10000, 1] = curr_logfX
        currps = np.loadtxt(file)[samples:,:16]
        currps_boot = f21stats.bootstrap(ps=currps, reps=10000, size=10)
        X_test[i*10000:(i+1)*10000, :] = currps_boot
    return X_test, y_test

def save_model(model, modelfile):
    # Save the model architecture and weights
    print(f'Saving model to: {modelfile}')
    model_json = model.save_model(modelfile)

logger = None
output_dir = None
args = None
# main code starts here
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.determinisitc=True
    torch.backends.cudnn.benchmark=False

    parser = base.setup_args_parser()
    #parser.add_argument('--datapath', type=str, default="saved_output/train_test_psbs_dump/noisy_500/f21_ps_dum_train_test_uGMRT_t500.0_20250511105815/ps/", help='')
    #parser.add_argument('--testdatapath', type=str, default="saved_output/train_test_psbs_dump/noisy_500/f21_ps_dum_train_test_uGMRT_t500.0_20250511105815/test_ps/", help='')
    #parser.add_argument('--datapath', type=str, default="saved_output/train_test_psbs_dump/noisy_g50/f21_ps_dum_train_test_uGMRT_t50.0_20250410153928/ps/", help='model file')
    #parser.add_argument('--testdatapath', type=str, default="saved_output/train_test_psbs_dump/noisy_g50/f21_ps_dum_train_test_uGMRT_t50.0_20250410153928/test_ps/", help='model file')
    #parser.add_argument('--datapath', type=str, default="saved_output/train_test_psbs_dump/noisy_ska/f21_ps_dum_train_test_SKA1-low_t50.0_20250511105922/ps/", help='')
    #parser.add_argument('--testdatapath', type=str, default="saved_output/train_test_psbs_dump/noisy_ska/f21_ps_dum_train_test_SKA1-low_t50.0_20250511105922/test_ps/", help='')

    #parser.add_argument('--datapath', type=str, default="saved_output/train_test_psbs_dump/denoised_500/f21_unet_ps_dum_train_test_uGMRT_t500.0_20250511164401/ps/", help='')
    #parser.add_argument('--testdatapath', type=str, default="saved_output/train_test_psbs_dump/denoised_500/f21_unet_ps_dum_train_test_uGMRT_t500.0_20250511164401/test_ps/", help='')

    #parser.add_argument('--datapath', type=str, default="saved_output/train_test_psbs_dump/denoised_500/mixed_f21_unet_ps_dum_train_test_uGMRT_t500.0_20250604091744/ps/", help='')
    #parser.add_argument('--testdatapath', type=str, default="saved_output/train_test_psbs_dump/denoised_500/mixed_f21_unet_ps_dum_train_test_uGMRT_t500.0_20250604091744/test_ps/", help='')

    #parser.add_argument('--datapath', type=str, default="saved_output/train_test_psbs_dump/denoised_50/mixed_f21_unet_ps_dum_train_test_uGMRT_t50.0_20250607223018/ps/", help='')
    #parser.add_argument('--testdatapath', type=str, default="saved_output/train_test_psbs_dump/denoised_50/mixed_f21_unet_ps_dum_train_test_uGMRT_t50.0_20250607223018/test_ps/", help='')

    #parser.add_argument('--datapath', type=str, default="saved_output/train_test_psbs_dump/denoised_ska/f21_unet_ps_dum_train_test_SKA1-low_t50.0_20250511164401/ps/", help='')
    #parser.add_argument('--testdatapath', type=str, default="saved_output/train_test_psbs_dump/denoised_ska/f21_unet_ps_dum_train_test_SKA1-low_t50.0_20250511164401/test_ps/", help='')

    #parser.add_argument('--datapath', type=str, default="saved_output/train_test_psbs_dump/denoised_ska/mixed_f21_unet_ps_dum_train_test_SKA1-low_t50.0_20250608062755/ps/", help='')
    #parser.add_argument('--testdatapath', type=str, default="saved_output/train_test_psbs_dump/denoised_ska/mixed_f21_unet_ps_dum_train_test_SKA1-low_t50.0_20250608062755/test_ps/", help='')

    #../data/denoised_gmrt50h/f21_unet_ps_dum_train_test_uGMRT_t50.0_20250417191012/denoised_ps
    
    args = parser.parse_args()

    output_dir = base.create_output_dir(args=args)
    logger = base.setup_logging(output_dir)

    # Initialize the network
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    logger.info("####")
    logger.info(f"### Using \"{device}\" device ###")
    logger.info("####")

    # Load training and test data
    logger.info("Loading training data...")
    X_train, y_train = load_training_data(override_path=args.datapath, samples=args.limitsamplesize, args=args)
    logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")

    logger.info("Loading test data...")
    X_test, y_test = load_test_data(override_path=args.testdatapath, samples=args.limitsamplesize, args=args)
    logger.info(f"Test data shape: X={X_test.shape}, y={y_test.shape}")

    # Initialize and train XGBoost model
    logger.info("Training XGBoost model...")
    model = XGBRegressor(
        random_state=42
    )
    model.fit(X_train, y_train)

    logger.info(f"Fitted regressor: {model}")
    logger.info(f"Booster: {model.get_booster()}")
    feature_importance = model.feature_importances_
    save_model(model, f'{output_dir}/xgb-f21-inf-ps.json')
    np.savetxt(f"{output_dir}/feature_importance.csv", feature_importance, delimiter=',')
    logger.info(f"Feature importance: {feature_importance}")
    for imp_type in ['weight','gain', 'cover', 'total_gain', 'total_cover']:
        logger.info(f"Importance type {imp_type}: {model.get_booster().get_score(importance_type=imp_type)}")


    # Make predictions
    logger.info("Making predictions...")
    y_pred = model.predict(X_test)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    logger.info("\nModel Performance:")
    logger.info(f"R2 Score: {r2:.4f}")
    logger.info(f"MSE: {mse:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    base.save_test_results(y_pred, y_test, output_dir)

    pltr.summarize_test_1000(y_pred, y_test, output_dir=output_dir, showplots=False, saveplots=True, label=f"{args.telescope}, {args.t_int:.0f}h")
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Plot xHI predictions
    plt.subplot(1, 2, 1)
    plt.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.5)
    plt.plot([y_test[:, 0].min(), y_test[:, 0].max()], 
             [y_test[:, 0].min(), y_test[:, 0].max()], 'r--')
    plt.xlabel('True xHI')
    plt.ylabel('Predicted xHI')
    plt.title('xHI Predictions')
    
    # Plot logfX predictions
    plt.subplot(1, 2, 2)
    plt.scatter(y_test[:, 1], y_pred[:, 1], alpha=0.5)
    plt.plot([y_test[:, 1].min(), y_test[:, 1].max()], 
             [y_test[:, 1].min(), y_test[:, 1].max()], 'r--')
    plt.xlabel('True logfX')
    plt.ylabel('Predicted logfX')
    plt.title('logfX Predictions')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions.pdf'), format='pdf')


    logger.info(f"\nResults saved to {output_dir}")

if __name__ == '__main__':
    main() 
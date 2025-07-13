'''
Dump PS and bispec for f21 data
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
import plot_results as pltr
import Scaling
import PS1D
import F21Stats as f21stats

import numpy as np
import sys
import os

signal_bandwidth=22089344.0
def dump_ps(datafile, dir, psbatchsize, save_ks):
    file_name = os.path.basename(datafile)  # Extract the filename from the path
    file_name_no_ext = os.path.splitext(file_name)[0]  # Remove the extension

    los_set = np.loadtxt(datafile)
    # powerspectrum computation
    ks_set,ps_set = PS1D.get_P_set(los_set,signal_bandwidth, scaled=True)
    ks_binned_set,ps_binned_set = f21stats.logbin_power_spectrum_by_k(ks_set, ps_set, save_ks)
    # bispectrum computation
    k_bispec_set, bs_set = f21stats.F21Stats.compute_1d_bispectrum(los_set)
    k_bispec_set, bs_set = f21stats.logbin_power_spectrum_by_k_flex(k_bispec_set, bs_set, 16, 100)
    
    if save_ks:
        logger.info(f"Shapes of datasets before batching: {ps_binned_set.shape} {bs_set.shape}")

    # Batching ps and bs
    if psbatchsize > 1:
        ps_binned_set = ps_binned_set.reshape(-1, psbatchsize, ps_binned_set.shape[1]).mean(axis=1)
        bs_set = bs_set.reshape(-1, psbatchsize, bs_set.shape[1]).mean(axis=1)
    if save_ks:
        logger.info(f"Shapes of datasets after batching: {ps_binned_set.shape} {bs_set.shape}")

    if save_ks:
        logger.info(f'Saving PS, bispec data. PS shape:{ps_binned_set.shape}, ks shape:{ks_binned_set.shape}, bispec shape: {bs_set.shape}, k_bispec shape: {k_bispec_set.shape}')
        np.savetxt(f'{dir}/ks_bin.csv', np.hstack((ks_binned_set[0], k_bispec_set)))
        logger.info(f" Saved 'k' bin values to f'{dir}/ks_bin.csv'")
    
    # Save the PS output to a file
    ps_file = f'{dir}/{file_name_no_ext}.csv' 
    np.savetxt(ps_file, np.hstack((ps_binned_set, bs_set)))
    logger.info(f" Saved PS, bispec to {ps_file}")

# main code starts here
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.determinisitc=True
torch.backends.cudnn.benchmark=False

parser = base.setup_args_parser()
parser.add_argument('--dataset', type=str, default='full', help='one of full, test_only, small_set')
args = parser.parse_args()

output_dir = base.create_output_dir(args=args)
logger = base.setup_logging(output_dir)

## Loading data
train_files = base.get_datafile_list(type='noisy', args=args, extn='csv', filter='train_only', override_path="../data/denoised/f21_unet_ps_dum_train_test_uGMRT_t500.0_20250317153922/denoised_los/")
if args.maxfiles is not None: datafiles = train_files[:args.maxfiles]
test_files = base.get_datafile_list(type='noisy', args=args, extn='csv', filter='test_only', override_path="../data/denoised/f21_unet_ps_dum_train_test_uGMRT_t500.0_20250317153922/denoised_los/")

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


ps_dir = f'{output_dir}/ps'
test_ps_dir = f'{output_dir}/test_ps'
os.mkdir(ps_dir)
os.mkdir(test_ps_dir)

if args.dataset == "full":
    logger.info(f'Dumping PS for training')
    for i,datafile in enumerate(train_files):
        logger.info(f"Loading file {i+1}/{len(train_files)}: {datafile}")
        dump_ps(datafile, dir=ps_dir, psbatchsize=10, save_ks=(i==0))

logger.info(f'Dumping PS for testing')
for i,datafile in enumerate(test_files):
    logger.info(f"Loading file {i+1}/{len(test_files)}: {datafile}")
    dump_ps(datafile, dir=test_ps_dir, psbatchsize=1, save_ks=(i==0))

logger.info(f"Dump completed. Output:{output_dir}")

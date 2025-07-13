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


def load_dataset(datafiles, psbatchsize, max_workers=8):
    # Lists to store combined data
    all_params = []
    # Create processor with desired number of worker threads
    processor = dl.F21DataLoader(max_workers=max_workers, psbatchsize=psbatchsize, use_bispectrum=True, skip_stats=True, ps_log_bins=True, ps_bins=16, perc_bins_to_use=100, scale_ps=True)

    # Process all files and get results
    results = processor.process_all_files(datafiles)
    logger.info(f"Finished data loading.")
    # Access results
    keys = results['key']
    ps = results['ps']
    ks = results['ks']
    bispec = results['bispectrum']
    k_bispec = results['k_bispec']
    params = results['params']
    freq_axis = results['freq_axis']
    logger.info(f"sample ps:{ps[0]}")
    logger.info(f"sample ks:{ks[0]}")
    logger.info(f"sample bispec:{bispec[0]}")
    logger.info(f"sample k_bispec:{k_bispec[0]}")
    logger.info(f"sample params:{params[0]}")
    
    # Combine all data
    logger.info(f"\nCombined ps shape: {ps.shape}")
    logger.info(f"\nCombined ks shape: {ks.shape}")
    logger.info(f"\nCombined bispec shape: {bispec.shape}")
    logger.info(f"\nCombined k_bispec shape: {k_bispec.shape}")
    logger.info(f"Combined parameters shape: {params.shape}")
            
    return (ks, ps, k_bispec, bispec, params, keys, freq_axis)

def dump_ps(datafile, dir, psbatchsize, save_ks):
    file_name = os.path.basename(datafile)  # Extract the filename from the path
    file_name_no_ext = os.path.splitext(file_name)[0]  # Remove the extension

    ks, ps, k_bispec, bispec, params, keys, freq_axis = load_dataset([datafile], max_workers=1, psbatchsize=1)
    if psbatchsize is not None and psbatchsize > 1:
        # need to aggregate. use the specified aggregation
        n_batches = len(ps) // psbatchsize
        ps_batched = np.zeros((n_batches, ps.shape[1]))
        bispec_batched = np.zeros((n_batches, bispec.shape[1]))
        for i in range(n_batches):
            if args.aggtype.lower() == 'mean':
                ps_batched[i,:] = np.nanmean(ps[i*psbatchsize:(i+1)*psbatchsize], axis=0)
                bispec_batched[i,:] = np.nanmean(bispec[i*psbatchsize:(i+1)*psbatchsize], axis=0)
            elif args.aggtype.lower() == 'median':
                ps_batched[i,:] = np.nanmedian(ps[i*psbatchsize:(i+1)*psbatchsize], axis=0)
                bispec_batched[i,:] = np.nanmedian(bispec[i*psbatchsize:(i+1)*psbatchsize], axis=0)
            else:
                raise ValueError(f"Invalid aggregation type: {args.aggtype}. Use 'mean' or 'median'")
        ps = ps_batched
        bispec = bispec_batched
    if save_ks:
        logger.info(f'Saving PS, bispec data. PS shape:{ps.shape}, ks shape:{ks.shape}, bispec shape: {bispec.shape}, k_bispec shape: {k_bispec.shape}')
        np.savetxt(f'{dir}/ks_bin.csv', np.hstack((ks, k_bispec))[0])
        logger.info(f" Saved 'k' bin values to f'{dir}/ks_bin.csv'")
    # Save the PS output to a file
    ps_file = f'{dir}/{file_name_no_ext}.csv' 
    np.savetxt(ps_file, np.hstack((ps, bispec)))
    logger.info(f" Saved PS, bispec to {ps_file}")

# main code starts here
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.determinisitc=True
torch.backends.cudnn.benchmark=False

parser = base.setup_args_parser()
parser.add_argument('--dataset', type=str, default='full', help='full/test_only/small_set')
parser.add_argument('--signaltype', type=str, default='noisy', help='noisy/signalonly')
parser.add_argument('--aggtype', type=str, default='mean', help='mean/median')
args = parser.parse_args()

output_dir = base.create_output_dir(args=args)
logger = base.setup_logging(output_dir)

## Loading data
train_files = base.get_datafile_list(type=args.signaltype, args=args, filter='train_only')
if args.maxfiles is not None: datafiles = train_files[:args.maxfiles]
test_files = base.get_datafile_list(type=args.signaltype, args=args, filter='test_only')

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
        dump_ps(datafile, dir=ps_dir, psbatchsize=args.psbatchsize, save_ks=(i==0))

logger.info(f'Dumping PS for testing')
for i,datafile in enumerate(test_files):
    logger.info(f"Loading file {i+1}/{len(test_files)}: {datafile}")
    dump_ps(datafile, dir=test_ps_dir, psbatchsize=1, save_ks=(i==0))

logger.info(f"Dump completed. Output:{output_dir}")

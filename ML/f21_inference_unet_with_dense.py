'''
Predict parameters fX and xHI from the 21cm forest data using CNN.
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
from UnetModelWithDense import UnetModel

import numpy as np
import sys

import matplotlib.pyplot as plt

import optuna
from xgboost import XGBRegressor

def test_multiple(datafiles, regression_model, latent_model, reps=10000, size=10, input_points_to_use=None):
    logger.info(f"Test_multiple started. {reps} reps x {size} points will be tested for {len(datafiles)} parameter combinations")
    # Create processor with desired number of worker threads
    all_y_test = np.zeros((len(datafiles)*reps, 2))
    all_y_pred = np.zeros((len(datafiles)*reps, 2))
    # Process all files and get results
    for i, f in enumerate(datafiles):
        if i==0: logger.info(f"Working on param combination #{i+1}: {f.split('/')[-1]}")
        los, params, _, _, _ = base.load_dataset([f], psbatchsize=1, limitsamplesize=None, save=False)
        if args.input_points_to_use is not None:
            los = los[:, :args.input_points_to_use]
        logger.info(f"Loaded los.shape={los.shape}")
        # remove the first 200 los as this region has been used for training
        los = los[200:]
        logger.info(f"Removed first 200 los. los.shape={los.shape}")
        los_tensor = torch.tensor(los, dtype=torch.float32)
        latent_features = latent_model.get_latent_features(los_tensor)

        #if i == 0: logger.info(f"sample test los_so:{los[:1]}")
        y_pred_for_test_point = []
        for j in range(reps):
            #pick 10 samples
            rdm = np.random.randint(len(los), size=size)
            latent_features_set = latent_features[rdm]

            #print(f"latent_features_set.shape={latent_features_set.shape}")
            latent_features_mean = np.mean(latent_features_set, axis=0, keepdims=True)
            #print(f"latent_features_mean.shape={latent_features_mean.shape}")
            y_pred = regression_model.predict(latent_features_mean)  # Predict using the trained regressor
        
            y_pred_for_test_point.append(y_pred)
            all_y_pred[i*reps+j,:] = y_pred
            all_y_test[i*reps+j,:] = params[0]
        if i==0: 
            logger.info(f"Test_multiple: param combination min, max should be the same:{np.min(params, axis=0)}, {np.max(params, axis=0)}")
            
    logger.info(f"Test_multiple: param combination:{params[0]} predicted mean:{np.mean(y_pred_for_test_point, axis=0)}")

    logger.info(f"Test_multiple completed. actual shape {all_y_test.shape} predicted shape {all_y_pred.shape}")
    
    pltr.calc_squared_error(all_y_pred, all_y_test)

    r2_means = pltr.summarize_test_1000(all_y_pred, all_y_test, output_dir, showplots=args.interactive, saveplots=True, label="_1000")

    r2 = np.mean(r2_means)
    base.save_test_results(all_y_pred, all_y_test, output_dir)

    return r2

def save_model(model):
    # Save the model architecture and weights
    logger.info(f'Saving model to: {output_dir}/f21_inference_xgb.pth')
    model_json = model.save_model(f"{output_dir}/f21_inference_xgb.pth")

# main code starts here
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.determinisitc=True
torch.backends.cudnn.benchmark=False

parser = base.setup_args_parser()
parser.add_argument('--test_multiple', action='store_true', help='Test 1000 sets of 10 LoS for each test point and plot it')
parser.add_argument('--test_reps', type=int, default=10000, help='Test repetitions for each parameter combination')
parser.add_argument('--test_sample_size', type=int, default=10, help='Number of samples of spectrum to be grouped')
args = parser.parse_args()
#if args.input_points_to_use not in [2048, 128]: raise ValueError(f"Invalid input_points_to_use {args.input_points_to_use}")
if args.input_points_to_use >= 2048: 
    step = 4
    kernel1 = 256
else: 
    step = 2
    kernel1 = 16

output_dir = base.create_output_dir(args=args)
logger = base.setup_logging(output_dir)

## Loading data
datafiles = base.get_datafile_list(type='noisy', args=args)
if args.maxfiles is not None: datafiles = datafiles[:args.maxfiles]

#test_points = [[-3.00,0.11],[-2.00,0.11],[-1.00,0.11],[-3.00,0.25],[-2.00,0.25],[-1.00,0.25],[-3.00,0.52],[-2.00,0.52],[-1.00,0.52], [-3.00,0.80],[-2.00,0.80],[-1.00,0.80]]#,[0.00,0.80]]
test_points = [[-3,0.11], [-3,0.80], [-1,0.11], [-1,0.80], [-2,0.52]]

train_files = []
test_files = []
for nof in datafiles:
    is_test_file = False
    for p in test_points:
        if nof.find(f"fX{p[0]:.2f}_xHI{p[1]:.2f}") >= 0:
            test_files.append(nof)
            is_test_file = True
            break
    if not is_test_file:
        train_files.append(nof)

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

## Load the trained Unet model
logger.info(f"Loading model from file {args.modelfile}")
model = UnetModel(input_size=args.input_points_to_use, input_channels=1, output_size=args.input_points_to_use, dropout=0.2, step=step)
model.load_model(args.modelfile)

logger.info(f"Loading training dataset {len(train_files)}")
X_train, y_train, _, keys, _ = base.load_dataset(train_files, psbatchsize=1, limitsamplesize=args.limitsamplesize, save=False, shuffle_samples = False)
if args.input_points_to_use is not None:
    X_train = X_train[:, :args.input_points_to_use]

# Predict on training data
logger.info(f"Mapping latent features for training dataset")
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
train_enc3_flattened = model.get_latent_features(X_train_tensor)
logger.info(f"Training set latent features shape={train_enc3_flattened.shape}")
if args.test_sample_size > 1:
    params_train, latent_train = f21stats.aggregate_f21_data(y_train, train_enc3_flattened, 10)
else:
    params_train, latent_train = y_train, train_enc3_flattened
# Save the enc3 output to a file
np.savetxt(f"{output_dir}/train_latent_features.csv", latent_train)
logger.info(f"Saved training data latent features to {output_dir}/train_latent_features.csv")

# Train XGBoostRegressor on the enc3 output
regressor = XGBRegressor(random_state=42)
logger.info(f"Fitting regressor model {regressor}")
regressor.fit(latent_train, params_train)  # Train on the flattened enc3 output
feature_importance = regressor.feature_importances_
save_model(regressor)
np.savetxt(f"{output_dir}/feature_importance.csv", feature_importance, delimiter=',')
logger.info(f"Feature importance: {feature_importance}")
for imp_type in ['weight','gain', 'cover', 'total_gain', 'total_cover']:
    logger.info(f"Importance type {imp_type}: {regressor.get_booster().get_score(importance_type=imp_type)}")

# Predict parameters for the test dataset
r2 = test_multiple(test_files, regression_model=regressor, latent_model=model, input_points_to_use=args.input_points_to_use, size=args.test_sample_size)

# Calculate R2 score
logger.info(f"R2 Score for 10k inference: {r2}")

# ML-Based Inference of Physical Parameters from 21 cm Forest Spectra

This repository contains Python code and Jupyter notebooks to perform inference of astrophysical parameters from mock 21 cm forest spectra, using machine learning pipelines. The project is tailored for studies of the Epoch of Reionization (EoR) where the 21 cm absorption features provide key insights into the thermal and ionization history of the intergalactic medium.

## Overview

We implement three machine learning pipelines, each extracting features from noisy 21 cm forest spectra (which include instrumental effects) and using XGBoost for regression to predict two key physical parameters:

1. **Pipeline 1:** Computes the 1D power spectrum of the noisy 21 cm forest and uses it directly for inference.
2. **Pipeline 2:** Denoises the noisy spectrum using a U-Net and then computes the 1D power spectrum for inference.
3. **Pipeline 3:** Uses the encoder of the trained U-Net to extract a latent feature vector from the spectrum, which is then used for regression.

Most of the `.py` scripts support `-h` or `--help` to display usage instructions.

---

## Directory Contents

### Main Python modules
| File | Description |
|------|-------------|
| `F21DataLoader.py`           | Loads mock 21 cm forest datasets for training/testing. |
| `F21Stats.py`                | Computes statistical measures on the spectra. |
| `PS1D.py`                    | Calculates the 1D power spectrum. |
| `Scaling.py`                 | Performs data scaling / normalization. |
| `UnetModelWithDense.py`      | Defines the U-Net architecture with a dense latent layer for feature extraction. |
| `f21_inference_ps.py`        | Pipeline 1: inference using Power spectrum of 21-cm forest spectrum with added noise. |
| `f21_inference_ps_unet.py`   | Pipeline 2: inference using power spectrum of 21-cm forest spectrum denoised with U-Net. |
| `f21_inference_unet_with_dense.py` | Pipeline 3: Latent features extraction using U-Net encoder and inference. This requires large memory for loading the U-Net model and latent feature extraction. |
| `f21_predict_*`              | Scripts for training the models (U-Net model). |
| `posterior_maps_*.py`, `posterior_plot*.py` | Scripts to generate posterior plots and statistical summaries. |
| `plot_results.py`            | Visualization utilities. |

### Jupyter notebooks
| File | Purpose |
|------|---------|
| `analyse_ps_stats_data.ipynb`, `analysis1.ipynb` | Exploratory analysis of data and power spectrum statistics. |
| `denoised_los_analysis.ipynb`, `denoised_ps_dump_analysis.ipynb` | Analysis of denoised spectra. |
| `train_test_data_analysis.ipynb` | Investigation of training/test splits. |
| `timeseries_analysis.ipynb`, `ps_dump_analysis.ipynb` | power spectrum dump checks. |
| `visualize_results.ipynb` | Plots of model outputs and inference results. |

### Output and results
- `saved_output/`: Contains stored test results, plots, and posterior samples used in our publication.
- `output/`, `tmp_out/`: Temporary or intermediate files.

---

## Usage

Most scripts support:
```bash
python script_name.py -h


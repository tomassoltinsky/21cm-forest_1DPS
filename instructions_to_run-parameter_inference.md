Here are the instructions to run various codes for IGM parameter inference
based on the 21-cm forest 1D power spectrum, including MCMC and null-detection limits.<br>
This assumes that the user has already generated 1D power spectra of the 21-cm forest signal and noise following the workflow in instructions_to_run-1DPS.md.

<br>

*covar_matrix_plot.py*

python covar_matrix_plot.py \<z_name\> \<dvH\> \<spec_res\> \<xHI_mean_mock\> \<logfX_mock\> \<telescope\> \<S147\> \<alphaR\> \<tint\> \<Nobs\>

\<z_name\> - redshift of the LOS data<br>
\<dvH\> - rebinning width in m/s<br>
\<spec_res\> - spectral resolution of telescope (i.e. frequency channel width) in kHz<br>
\<xHI_mean_mock\> - mean neutral hydrogen fraction value<br>
\<logfX_mock\> - log10(f_X) (X-ray preheating efficiency) value<br>
\<telescope\> - telescope name; choices are: uGMRT, LOFAR, SKA1-low, SKA2<br>
\<S147\> - intrinsic flux of QSO at 147Hz in mJy<br>
\<alphaR\> - radio spectral index of QSO<br>
\<tint\> - integration time of observation in hours<br>
\<Nobs\> - number of observed LOS for the mock observation<br>
This code generates a plots of the covariance matrices of the 1D power spectrum of the 21-cm forest signal, noise and their combination. The output is saved in plots/.<br>
For Fig. 8 in Šoltinský et al. 2025, MNRAS, 537, 364 - https://ui.adsabs.harvard.edu/abs/2025MNRAS.537..364S/abstract

<br>

*param_inf_PS_mock.py*

z_name = float(sys.argv[1])         #redshift
dvH = float(sys.argv[2])            #used rebinning for line profile convolution in m/s
spec_res = float(sys.argv[3])       #spectral resolution of the telescope in kHz
xHI_mean_mock = float(sys.argv[4])  #mock HI fraction
logfX_mock = float(sys.argv[5])     #mock logfX
path_LOS = '../../datasets/21cmFAST_los/los/'
telescope = str(sys.argv[6])        #telescope
S147 = float(sys.argv[7])           #intrinsic flux density of background source at 147MHz in mJy
alphaR = float(sys.argv[8])         #radio spectrum power-law index of background source
tint = float(sys.argv[9])           #intergration time for the observation in h
Nobs = int(sys.argv[10])            #number of LOS to be observed
Nkbins = int(sys.argv[11])          #number of k bins

python param_inf_PS_mock.py \<z_name\> \<dvH\> \<spec_res\> \<xHI_mean_mock\> \<logfX_mock\> \<telescope\> \<S147\> \<alphaR\> \<tint\> \<Nobs\> \<Nkbins\>

\<z_name\> - redshift of the LOS data<br>
\<dvH\> - rebinning width in m/s<br>
\<spec_res\> - spectral resolution of telescope (i.e. frequency channel width) in kHz<br>
\<xHI_mean_mock\> - mean neutral hydrogen fraction value<br>
\<logfX_mock\> - log10(f_X) (X-ray preheating efficiency) value<br>
\<telescope\> - telescope name; choices are: uGMRT, LOFAR, SKA1-low, SKA2<br>
\<S147\> - intrinsic flux of QSO at 147Hz in mJy<br>
\<alphaR\> - radio spectral index of QSO<br>
\<tint\> - integration time of observation in hours<br>
\<Nobs\> - number of observed LOS for the mock observation<br>
\<Nkbins\> - number of k bins to be used<br>

This code performs MCMC parameter inference to constrain the IGM parameters (x_HI, f_X). The result is plotted as a corner plot and a plot of 1D power spectra of the mock observation, MCMC draws and the best fit model. The output is saved in plots/.<br>
For Fig. 9 and 11 in Šoltinský et al. 2025, MNRAS, 537, 364 - https://ui.adsabs.harvard.edu/abs/2025MNRAS.537..364S/abstract

<br>

*param_inf_covarmock_multidk.py*

python param_inf_covarmock_multidk.py \<z_name\> \<dvH\> \<spec_res\> \<xHI_mean_mock\> \<logfX_mock\> \<telescope\> \<S147\> \<alphaR\> \<tint\> \<Nobs\> \<Nkbins\>

\<z_name\> - redshift of the LOS data<br>
\<dvH\> - rebinning width in m/s<br>
\<spec_res\> - spectral resolution of telescope (i.e. frequency channel width) in kHz<br>
\<xHI_mean_mock\> - mean neutral hydrogen fraction value<br>
\<logfX_mock\> - log10(f_X) (X-ray preheating efficiency) value<br>
\<telescope\> - telescope name; choices are: uGMRT, LOFAR, SKA1-low, SKA2<br>
\<S147\> - intrinsic flux of QSO at 147Hz in mJy<br>
\<alphaR\> - radio spectral index of QSO<br>
\<tint\> - integration time of observation in hours<br>
\<Nobs\> - number of observed LOS for the mock observation<br>
\<Nkbins\> - number of k bins to be used<br>

This code performs MCMC parameter inference to constrain the IGM parameters (x_HI, f_X). The resulting posterior distribution is saved in MCMC_samples/ as well as the plots showing MCMC chains and corner plot in plots/.

<br>

*null-detection_limits.py*

python null-detection_limits.py \<z_name\> \<dvH\> \<spec_res\> \<Nobs\> \<Nkbins\>

\<z_name\> - redshift of the LOS data<br>
\<dvH\> - rebinning width in m/s<br>
\<spec_res\> - spectral resolution of telescope (i.e. frequency channel width) in kHz<br>
\<Nobs\> - number of observed LOS for the mock observation<br>
\<Nkbins\> - number of k bins to be used<br>

This code finds the 95% and 68% limits in the xHI-logfX parameter space based on a null-detection of the 21-cm forest 1D power spectrum. A null-detection is defined if at least k-bin has a signal 1D PS below the noise 1D PS. The output is interpolator data saved in datasets/interpolators/.

<br>

*posterior_maps_multip.py*

python posterior_maps_multip.py

This code generates plots with multiple posterior distributions based on different mock observations of the 21-cm forest 1D power spectrum assuming different true values of logfX and xHI. It also plots null-detection limits. Change the observational setup within the code. Figures are saved in plots/.<br>
This code also generates interpolator data of logfX, xHI, TK and THI which are saved in datasets/interpolators/. Once generated, there is no need to generate it again and that part can be commented out as it takes some time to perform.<br>
For Fig. 10, 12 and 13 in Šoltinský et al. 2025, MNRAS, 537, 364 - https://ui.adsabs.harvard.edu/abs/2025MNRAS.537..364S/abstract

<br>

*param_inf_covarmock_multip_nsub.py*

python param_inf_covarmock_multip_nsub.py \<z_name\> \<dvH\> \<spec_res\> \<telescope\> \<S147\> \<alphaR\> \<tint\> \<Nobs\>

\<z_name\> - redshift of the LOS data<br>
\<dvH\> - rebinning width in m/s<br>
\<spec_res\> - spectral resolution of telescope (i.e. frequency channel width) in kHz<br>
\<telescope\> - telescope name; choices are: uGMRT, LOFAR, SKA1-low, SKA2<br>
\<S147\> - intrinsic flux of QSO at 147MHz in mJy<br>
\<alphaR\> - radio spectral index of QSO<br>
\<tint\> - integration time of observation in hours<br>
\<Nobs\> - number of observed LOS for the mock observation<br>

This code performs MCMC parameter inference to constrain the IGM parameters (x_HI, f_X) for multiple mock IGM models. The mean noise 1D power spectrum is subtracted from the mock signal+noise 1D power spectrum before performing the inference. The covariance matrix is calculated from the signal+noise ensemble. The mock IGM models and number of k bins are selected within the code. The resulting posterior distributions are saved in MCMC_samples/noise_sub/ together with the plots showing the MCMC chains and corner plots.

<br>

*posterior_maps_multip_noisesubtract.py*

python posterior_maps_multip_noisesubtract.py \<telescope\> \<tint\>

\<telescope\> - telescope name; choices are: uGMRT, LOFAR, SKA1-low, SKA2<br>
\<tint\> - integration time of observation in hours<br>

This code generates 2D posterior plots for multiple mock IGM models and compares parameter inference using the noisy and noise-subtracted 21-cm forest 1D power spectrum. The true and inferred values of xHI and logfX are shown for each mock model. The code also calculates metric G - the distance between the inferred and true parameter values and the mean posterior uncertainty. The mock IGM models and other observational parameters are selected within the code. Figures are saved in plots/.

<br>

*likelihood_check.py*

python likelihood_check.py \<z_name\> \<dvH\> \<spec_res\> \<telescope\> \<S147\> \<alphaR\> \<tint\> \<Nobs\> \<xHI_mean_mock\> \<logfX_mock\> \<k_ind\>

\<z_name\> - redshift of the LOS data<br>
\<dvH\> - rebinning width in m/s<br>
\<spec_res\> - spectral resolution of telescope (i.e. frequency channel width) in kHz<br>
\<telescope\> - telescope name; choices are: uGMRT, LOFAR, SKA1-low, SKA2<br>
\<S147\> - intrinsic flux of QSO at 147MHz in mJy<br>
\<alphaR\> - radio spectral index of QSO<br>
\<tint\> - integration time of observation in hours<br>
\<Nobs\> - number of observed LOS for the mock observation<br>
\<xHI_mean_mock\> - mean neutral hydrogen fraction of the mock observation<br>
\<logfX_mock\> - log10 of X-ray pre-heating efficiency of the mock observation<br>
\<k_ind\> - index of the k-bin for which the likelihood is tested<br>

This code checks the Gaussian likelihood assumed for the Bayesian parameter inference at a selected k-bin. It compares the assumed Gaussian likelihood with the distributions obtained from ensembles of the noisy and noise-subtracted mock 1D power spectra. The covariance matrix includes both sample variance and instrumental noise. The resulting figure is saved in plots/.<br>
For Fig. A1 in Patil et al. 2026, 546, 4, 18 - https://ui.adsabs.harvard.edu/abs/2026MNRAS.546ag236P/abstract
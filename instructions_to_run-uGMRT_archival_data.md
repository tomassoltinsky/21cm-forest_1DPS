Here are the instructions to run various codes for the analysis of the archival uGMRT observations of the radio-bright quasar PSO J352-15 presented in Šoltinský et al. (2026). The analysis starts from the residual target and off-source flux density spectra obtained after the initial radio-interferometric data reduction and calibration.

The codes perform the subsequent post-processing of these spectra, including residual bandpass feature subtraction using wavelet decomposition (WLD), identification and flagging of corrupted frequency channels, statistical comparison of the target and off-source spectra, and calculation of the line-of-sight 21-cm forest 1D power spectrum. The resulting power spectrum measurements are then compared with forward-modelled 21cmFAST simulations to constrain the thermal and ionization state of the IGM using Bayesian/MCMC parameter inference.

The codes also include tests of the analysis methodology, noise properties, likelihood assumptions, and the impact of the post-processing procedure. Together, these scripts reproduce the main analysis steps and several figures presented in Šoltinský et al. 2026, arXiv:2607.15341 - https://ui.adsabs.harvard.edu/abs/2026arXiv260715341S/abstract

The initial interferometric calibration and imaging steps used to produce the residual flux density spectra are not included here.

<br>

*obs_data_plot_postprocess_bandpassfilter.py*

python obs_data_plot_postprocess_bandpassfilter.py \<WLDlvl\> \<f_flag\>

\<WLDlvl\> - level of wavelet decomposition used for residual bandpass feature subtraction; 0 corresponds to the coarsest features<br>
\<f_flag\> - threshold for flagging bad frequency channels in units of the mean off-source noise rms<br>

This code post-processes the uGMRT residual flux density spectra of PSO J352-15 and the corresponding off-source spectra. It subtracts residual bandpass features using wavelet decomposition (WLD) and flags corrupted frequency channels based on their relative noise rms. It compares the statistical distributions of the target and off-source spectra and computes their 1D power spectra before and after the post-processing. The post-processed residual flux density spectra and 1D power spectra are saved in obs_data/, while the diagnostic figures are saved in plots/Soltinsky_2026/.<br>
For Fig. 3, 5, 6, 7, C1 and C2 in Šoltinský et al. 2026

<br>

*noise-level_uGMRT.py*

python noise-level_uGMRT.py

This code measures the noise level of the uGMRT observations as a function of effective integration time. The off-source spectra are combined cumulatively from one to five observing nights and processed using the same residual bandpass subtraction and channel-flagging procedure as the full observational dataset. The resulting noise rms per frequency channel is measured for each effective integration time and fitted with an empirical power-law relation. The measured scaling is compared with the theoretical radiometer scaling and is also used to forecast the sensitivity for longer uGMRT observations. The calibration parameters, integration times, and fraction of data excised during the analysis are specified within the code. The resulting figure is saved in plots/Soltinsky_2026/.<br>
For Fig. 4 and Table 1 in Šoltinský et al. 2026.

<br>

*obs_data_plot_MWA.py*

python obs_data_plot_MWA.py

This code processes the measured signal and thermal noise 1D power spectra from the MWA observations of PSO J352-15 for comparison with the uGMRT measurement. The MWA power spectra are rebinned using the same logarithmic k bins as the uGMRT data. The code also calculates and indicates the approximate sidelobe-induced mode-mixing cutoff for the MWA observations and converts between frequency and cosmological wavenumbers. The resulting figure is saved in plots/Soltinsky_2026/.<br>
For Fig. 8 in Šoltinský et al. 2026.

<br>

*spectrum_21cmFAST_noisy_uGMRT_archivaldata.py*

python spectrum_21cmFAST_noisy_uGMRT_archivaldata.py \<z_name\> \<dvH\> \<spec_res\> \<telescope\> \<S_min_QSO\> \<alpha_R\> \<fX_name\> \<xHI_mean\> \<LOS\> \<rms\>

\<z_name\> - redshift of the 21cmFAST LOS data<br>
\<dvH\> - velocity resolution of the LOS data in m/s<br>
\<spec_res\> - spectral resolution of the telescope (i.e. frequency channel width) in kHz<br>
\<telescope\> - telescope name<br>
\<S_min_QSO\> - intrinsic flux density of the QSO at 147 MHz in mJy<br>
\<alpha_R\> - radio spectral index of the QSO<br>
\<fX_name\> - X-ray pre-heating efficiency fX of the simulated IGM model<br>
\<xHI_mean\> - mean neutral hydrogen fraction of the simulated IGM model<br>
\<LOS\> - index of the simulated line-of-sight<br>
\<rms\> - rms instrumental noise per frequency channel in mJy/beam<br>

This code generates a forward-modelled synthetic noisy 21-cm forest spectrum corresponding to the uGMRT archival observations of PSO J352-15. The simulated forest is placed blueward of the target frequency, while a continuum is assumed redward of the target. Frequency-independent Gaussian instrumental noise in flux-density units, specified by the measured rms of the archival observations, is added to the spectrum. The fractional noise relative to the QSO continuum varies with frequency according to the assumed radio spectral index. The resulting mock signal and signal+noise spectra are plotted over the frequency range of the archival uGMRT observations. The resulting figure is saved in plots/Soltinsky_2026/.<br>
For Fig. 9 in Šoltinský et al. 2026.

<br>

*1DPS_data_signal_21cmFAST_multiparam_uGMRT-AD_flag.py*

python 1DPS_data_signal_21cmFAST_multiparam_uGMRT-AD_flag.py \<z_name\> \<dvH\> \<spec_res\> \<n_los\> \<WLDlvl\> \<f_flag\>

\<z_name\> - redshift of the 21cmFAST LOS data<br>
\<dvH\> - velocity resolution of the LOS data in m/s<br>
\<spec_res\> - spectral resolution of the telescope (i.e. frequency channel width) in kHz<br>
\<n_los\> - number of simulated LOS used to calculate the signal 1D power spectra<br>
\<WLDlvl\> - level of wavelet decomposition used for residual bandpass feature subtraction; 0 corresponds to the coarsest features<br>
\<f_flag\> - threshold for flagging bad frequency channels in units of the mean off-source noise rms<br>

This code generates forward-modelled 21-cm forest signal 1D power spectra with the same frequency-channel mask as used for the uGMRT archival observations. The corrupted channels are identified from the off-source spectra after residual bandpass subtraction using wavelet decomposition (WLD). The resulting observational channel mask is applied to the simulated 21cmFAST LOS, which are smoothed to the spectral resolution of the observations before calculating their masked 1D power spectra. The IGM models used for the calculation can be selected within the code. The resulting signal 1D power spectrum ensembles are saved in 1DPS_dimensionless/1DPS_signal/.

<br>

*1DPS_data_noise_uGMRT-AD_flag.py*

python 1DPS_data_noise_uGMRT-AD_flag.py \<z_name\> \<dvH\> \<telescope\> \<spec_res\> \<S147\> \<alphaR\> \<rms\> \<WLDlvl\> \<f_flag\>

\<z_name\> - redshift of the 21cmFAST LOS data<br>
\<dvH\> - velocity resolution of the LOS data in m/s<br>
\<telescope\> - telescope name<br>
\<spec_res\> - spectral resolution of the telescope (i.e. frequency channel width) in kHz<br>
\<S147\> - intrinsic flux density of the QSO at 147 MHz in mJy<br>
\<alphaR\> - radio spectral index of the QSO<br>
\<rms\> - frequency-independent instrumental noise rms per frequency channel in mJy/beam<br>
\<WLDlvl\> - level of wavelet decomposition used for residual bandpass feature subtraction; 0 corresponds to the coarsest features<br>
\<f_flag\> - threshold for flagging bad frequency channels in units of the mean off-source noise rms<br>

This code generates forward-modelled noise-only 1D power spectrum realizations corresponding to the uGMRT archival observations of PSO J352-15. The corrupted frequency channels are identified from the off-source spectra after residual bandpass subtraction using wavelet decomposition (WLD). Gaussian instrumental noise realizations are generated assuming a frequency-independent noise rms in flux-density units, while the fractional noise varies across the band according to the frequency-dependent QSO continuum. The same observational channel mask is applied to each synthetic noise realization before calculating its masked 1D power spectrum. The resulting noise 1D power spectrum ensemble is saved in 1DPS_dimensionless/1DPS_noise/.

<br>

*1DPS_plot_21cmFAST_multinoise_uGMRT-AD_obs_flag.py*

python 1DPS_plot_21cmFAST_multinoise_uGMRT-AD_obs_flag.py \<z_name\> \<dvH\> \<spec_res\> \<fX_name\> \<xHI_mean\> \<WLDlvl\> \<f_flag\>

\<z_name\> - redshift of the 21cmFAST LOS data<br>
\<dvH\> - velocity resolution of the LOS data in m/s<br>
\<spec_res\> - spectral resolution of the telescope (i.e. frequency channel width) in kHz<br>
\<fX_name\> - log10 of the X-ray pre-heating efficiency of the simulated IGM model<br>
\<xHI_mean\> - mean neutral hydrogen fraction of the simulated IGM model<br>
\<WLDlvl\> - level of wavelet decomposition used for residual bandpass feature subtraction; 0 corresponds to the coarsest features<br>
\<f_flag\> - threshold for flagging bad frequency channels in units of the mean off-source noise rms<br>

This code compares the post-processed 21-cm forest 1D power spectrum measured from the uGMRT archival observations of PSO J352-15 with forward-modelled 21cmFAST signal and synthetic noise 1D power spectra. The simulated signal power spectrum is shown together with its LOS scatter, while synthetic noise power spectra are shown for two instrumental noise levels specified within the code. The simulated signal and noise power spectra include the same spectral-channel mask as the uGMRT observations. The approximate sidelobe-induced mode-mixing cutoff is also indicated, and the upper horizontal axis shows the corresponding cosmological wavenumber. The resulting figure is saved in plots/Soltinsky_2026/.<br>
For Fig. 9 in Šoltinský et al. 2026.

<br>

*interpolator_1DPS.py*

python interpolator_1DPS.py \<z_name\> \<spec_res\> \<WLDlvl\> \<f_flag\>

\<z_name\> - redshift of the 21cmFAST LOS data<br>
\<spec_res\> - spectral resolution of the telescope (i.e. frequency channel width) in kHz<br>
\<WLDlvl\> - level of wavelet decomposition used for residual bandpass feature subtraction; 0 corresponds to the coarsest features<br>
\<f_flag\> - threshold for flagging bad frequency channels in units of the mean off-source noise rms<br>

This code prepares the 21-cm forest 1D power spectrum datasets used for interpolation and Bayesian parameter inference. For each 21cmFAST model, it reads the IGM parameters, calculates the mean temperature of highly neutral gas, and reads and bins the corresponding forward-modelled signal 1D power spectrum ensemble. Selected models are removed from the final parameter space before the individual-LOS and mean 1D power spectrum datasets are saved in datasets/interpolators/. The code also produces a figure (saved in plots/) showing the final parameter space in xHI, THI and logfX.<br>
The saved interpolator datasets included in this repository were generated using the full set of simulations covering the parameter space explored in the study. The underlying 21cmFAST simulation and 1D power spectrum datasets used to generate them are not included in the repository due to their large size, but are available from the author upon request.

<br>

*covar_matrix_plot_uGMRT-AD_obs_WLD_mask.py*

python covar_matrix_plot_uGMRT-AD_obs_WLD_mask.py \<z_name\> \<spec_res\> \<xHI_mock\> \<logTHI_mock\> \<WLDlvl\> \<f_flag\>

\<z_name\> - redshift of the 21cmFAST LOS data<br>
\<spec_res\> - spectral resolution of the telescope (i.e. frequency channel width) in kHz<br>
\<xHI_mock\> - mean neutral hydrogen fraction of the selected IGM model<br>
\<logTHI_mock\> - log10 of the mean temperature of the neutral IGM in K<br>
\<WLDlvl\> - level of wavelet decomposition used for residual bandpass feature subtraction; 0 corresponds to the coarsest features<br>
\<f_flag\> - threshold for flagging bad frequency channels in units of the mean off-source noise rms<br>

This code calculates and plots covariance matrices for the 21-cm forest 1D power spectrum corresponding to the uGMRT archival observations of PSO J352-15. The sample-variance covariance is calculated from an ensemble of forward-modelled 21cmFAST signal power spectra interpolated to the specified xHI and THI values. The noise covariance is calculated directly from the post-processed measured off-source power spectra, while the total covariance is calculated from the combined signal+noise ensemble. Separate covariance matrices are produced for the sample variance, noise, and signal+noise.<br>
For Fig. 10 in Šoltinský et al. 2026.

<br>

*param_inf_uGMRT-AD_postprocess_obs_THIvsxHI.py*

python param_inf_uGMRT-AD_postprocess_obs_THIvsxHI.py \<z_name\> \<spec_res\> \<telescope\> \<WLDlvl\> \<f_flag\>

\<z_name\> - redshift of the 21cmFAST LOS data<br>
\<spec_res\> - spectral resolution of the telescope (i.e. frequency channel width) in kHz<br>
\<telescope\> - telescope used for the observations<br>
\<WLDlvl\> - level of wavelet decomposition used for residual bandpass feature subtraction; 0 corresponds to the coarsest features<br>
\<f_flag\> - threshold for flagging bad frequency channels in units of the mean off-source noise rms<br>

This code performs Bayesian parameter inference in the mean neutral hydrogen fraction - mean neutral IGM temperature (xHI-logTHI) parameter space using the postprocessed uGMRT archival observations of PSO J352-15. The mean measured off-source noise 1D PS is subtracted from the target 1D PS following Method A2 in Patil et al. 2026. At each point sampled by the MCMC, the noise-subtracted observed 1D PS is compared with the interpolated mean 21-cm forest signal 1D PS at the corresponding xHI and logTHI values. The parameter-dependent covariance matrix is calculated from the corresponding ensemble of interpolated signal 1D PS realizations combined with the measured off-source noise realizations. The code saves the flattened posterior samples in MCMC_samples/uGMRT_archivaldata/ and produces MCMC chain and posterior corner plots saved in plots/Soltinsky_2026/.

<br>

*postdist_plot_uGMRTarchivaldata_obs_THIvsxHI.py*

python postdist_plot_uGMRTarchivaldata_obs_THIvsxHI.py \<WLDlvl\> \<f_flag\>

\<WLDlvl\> - level of wavelet decomposition used for residual bandpass feature subtraction; 0 corresponds to the coarsest features<br>
\<f_flag\> - threshold for flagging bad frequency channels in units of the mean off-source noise rms<br>

This code plots the posterior constraints on the mean neutral hydrogen fraction and mean temperature of the neutral IGM obtained from the null detection of the 21-cm forest 1D power spectrum in the uGMRT archival observations of PSO J352-15. The posterior constraints are compared with existing neutral-fraction constraints from Lyα observations and neutral-IGM temperature constraints from 21-cm tomography. The code also extracts the lower temperature boundary of the posterior contour and evaluates the corresponding limits at selected xHI and THI values. A full posterior corner plot is produced as an additional diagnostic.<br>
The results presented in the paper were obtained using an MCMC with 10 000 000 steps, and the corresponding figure is included in the repository. To reduce the size of the repository, the provided MCMC posterior distribution dataset is based on a shorter MCMC with 100 000 steps.<br>
For Fig. 11 and D1 in Šoltinský et al. 2026.
<br>

*covar_matrix_plot_uGMRT-AD_sim_WLD_mask.py*

python covar_matrix_plot_uGMRT-AD_sim_WLD_mask.py \<z_name\> \<spec_res\> \<xHI_mock\> \<logTHI_mock\> \<WLDlvl\> \<f_flag\> \<rms\>

\<z_name\> - redshift of the 21cmFAST LOS data<br>
\<spec_res\> - spectral resolution of the telescope (i.e. frequency channel width) in kHz<br>
\<xHI_mock\> - mean neutral hydrogen fraction of the selected IGM model<br>
\<logTHI_mock\> - log10 of the mean temperature of the neutral IGM in K<br>
\<WLDlvl\> - level of wavelet decomposition used for residual bandpass feature subtraction; 0 corresponds to the coarsest features<br>
\<f_flag\> - threshold for flagging bad frequency channels in units of the mean off-source noise rms<br>
\<rms\> - frequency-independent instrumental noise rms per frequency channel in mJy/beam<br>

This code calculates and plots covariance matrices for simulated 21-cm forest observations corresponding to the uGMRT archival observations of PSO J352-15. The sample-variance covariance is calculated from an ensemble of forward-modelled 21cmFAST signal power spectra interpolated to the specified xHI and THI values. The noise covariance is calculated from the corresponding simulated instrumental-noise power spectrum ensemble for the specified noise rms, while the total covariance is calculated from the combined signal+noise ensemble. Separate covariance matrices are produced for the sample variance, noise, and signal+noise.<br>
For Fig. E1 in Šoltinský et al. 2026.

<br>

*param_inf_uGMRT-AD_postprocess_sim_THIvsxHI.py*

python param_inf_uGMRT-AD_postprocess_sim_THIvsxHI.py \<z_name\> \<spec_res\> \<telescope\> \<S147\> \<alphaR\> \<WLDlvl\> \<f_flag\> \<rms\>

\<z_name\> - redshift of the 21cmFAST LOS data<br>
\<spec_res\> - spectral resolution of the telescope (i.e. frequency channel width) in kHz<br>
\<telescope\> - telescope used for the simulated observation<br>
\<S147\> - source flux density at 147 MHz in mJy<br>
\<alphaR\> - radio spectral index of the source<br>
\<WLDlvl\> - level of wavelet decomposition used for residual bandpass feature subtraction; 0 corresponds to the coarsest features<br>
\<f_flag\> - threshold for flagging bad frequency channels in units of the mean noise rms<br>
\<rms\> - frequency-independent instrumental noise rms per frequency channel in mJy/beam<br>

This code performs Bayesian parameter inference in the mean neutral hydrogen fraction - mean neutral IGM temperature (xHI-logTHI) parameter space for a simulated uGMRT observation. One realization of the simulated instrumental-noise 1D PS is used as a mock null observation, and the mean simulated noise 1D PS is subtracted following Method A2 in Patil et al. 2026. At each point sampled by the MCMC, the noise-subtracted mock observation is compared with the interpolated mean 21-cm forest signal 1D PS at the corresponding xHI and logTHI values. The parameter-dependent covariance matrix is calculated from the corresponding ensemble of interpolated signal 1D PS realizations combined with the simulated instrumental-noise realizations. The code saves the flattened posterior samples in MCMC_samples/uGMRT_archivaldata/ and produces MCMC chain and posterior corner plots saved in plots/Soltinsky_2026/.

<br>

*postdist_plot_uGMRTarchivaldata_sim_THIvsxHI.py*

python postdist_plot_uGMRTarchivaldata_sim_THIvsxHI.py \<WLDlvl\> \<f_flag\> \<rms\> \<LOS\>

\<WLDlvl\> - level of wavelet decomposition used for residual bandpass feature subtraction; 0 corresponds to the coarsest features<br>
\<f_flag\> - threshold for flagging bad frequency channels in units of the mean noise rms<br>
\<rms\> - frequency-independent instrumental noise rms per frequency channel in mJy/beam<br>
\<LOS\> - simulated instrumental-noise realization used as the mock observation<br>

This code plots the forecasted posterior constraints on the mean neutral hydrogen fraction and mean temperature of the neutral IGM obtained from a simulated null detection of the 21-cm forest 1D power spectrum for PSO J352-15 with uGMRT. The posterior constraints are compared with existing neutral-fraction constraints from Lyα observations and neutral-IGM temperature constraints from 21-cm tomography. The code also extracts the lower temperature boundary of the posterior contour and evaluates the corresponding limits at selected xHI values.<br>
The analysis in the paper uses an MCMC with 10 000 000 steps, and the corresponding figure is included in the repository. Due to file-size limitations, the MCMC posterior distribution dataset included in the repository is based on a shorter run with 100 000 steps.<br>
For Fig. 12 in Šoltinský et al. 2026.
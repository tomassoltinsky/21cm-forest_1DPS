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
For Fig. 4 in Šoltinský et al. 2026.

<br>
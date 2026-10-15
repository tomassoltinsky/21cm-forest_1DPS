Here are the instructions to run various codes to generate mock 1D power spectra of the 21-cm forest with recommended workflow.<br>
This assumes that the user has already generated the LOS data and computed the 21-cm forest optical depth using the instructions in instructions_to_run-Generating_mock_spectra.md.

*1DPS_data_signal_21cmFAST_multiparam.py*

python 1DPS_data_signal_21cmFAST_multiparam.py \<z_name\> \<dvH\> \<spec_res\> \<n_los\>

\<z_name\> - redshift of the LOS data<br>
\<dvH\> - rebinning width in km/s<br>
\<spec_res\> - spectral resolution of telescope (i.e. frequency channel width) in kHz<br>
\<n_los\> - number of lines-of-sight<br>

This code generates the 1D power spectrum of the 21-cm forest signal only using Periodogram estimate. The output is saved in 1DPS_dimensionless/1DPS_signal/. Note that the flux spectrum is smoothed with a fixed boxcar kernel of width equal to the spectral resolution of the telescope. Other 1DPS are computed in a similar way.

<br>

*1DPS_data_noise.py*

python 1DPS_data_noise.py \<z_name\> \<dvH\> \<spec_res\> \<telescope\> \<N_d\> \<S_min_QSO\> \<alpha_R\> \<t_int\>

\<z_name\> - redshift of the LOS data<br>
\<dvH\> - rebinning width in km/s<br>
\<spec_res\> - spectral resolution of telescope (i.e. frequency channel width) in kHz<br>
\<telescope\> - telescope name<br>
\<N_d\> - number of dishes used by telescope<br>
\<S_min_QSO\> - intrinsic flux of QSO at 147Hz in mJy<br>
\<alpha_R\> - radio spectral index of QSO<br>
\<t_int\> - integration time of observation in hours<br>

This code generates the 1D power spectrum of the noise in the 21-cm forest flux spectrum. The output is saved in 1DPS_dimensionless/1DPS_noise/.

<br>

*1DPS_data_signalandnoise_21cmFAST_multiparam.py*

python 1DPS_data_signalandnoise_21cmFAST_multiparam.py \<z_name\> \<dvH\> \<spec_res\> \<n_los\> \<telescope\> \<N_d\> \<S_min_QSO\> \<alpha_R\> \<t_int\>

This code generates the 1D power spectrum of the 21-cm forest signal and noise combined. The output is saved in 1DPS_dimensionless/1DPS_signalandnoise/.

<br>

*1DPS_plot_21cmFAST_multinoise.py*

z_name = float(sys.argv[1])   #redshift
dvH = float(sys.argv[2])      #rebinning width in km/s
spec_res = float(sys.argv[3]) #spectral resolution of telescope (i.e. frequency channel width) in kHz
fX_name = float(sys.argv[4])  #log10(f_X)
xHI_mean = float(sys.argv[5]) #mean x_HI

python 1DPS_plot_21cmFAST_multinoise.py \<z_name\> \<dvH\> \<spec_res\> \<fX_name\> \<xHI_mean\>

\<z_name\> - redshift of the LOS data<br>
\<dvH\> - rebinning width in km/s<br>
\<spec_res\> - spectral resolution of telescope (i.e. frequency channel width) in kHz<br>
\<fX_name\> - log10(f_X)<br>
\<xHI_mean\> - mean neutral hydrogen fraction<br>

This code generates a plot of the 1D power spectrum of the 21-cm forest signal and 1D power spectrum of the noise for multiple telescope configurations. The output is saved in plots/.<br>
For Fig. 6 in Šoltinský et al. 2025, MNRAS, 537, 364 - https://ui.adsabs.harvard.edu/abs/2025MNRAS.537..364S/abstract

<br>

*1DPS_plot_21cmFAST_multiparam_interpolate_nomock.py*

python 1DPS_plot_21cmFAST_multiparam_interpolate_nomock.py \<z_name\> \<dvH\> \<spec_res\> \<fX_fid\> \<xHI_fid\>

This code generates a plot of the 1D power spectrum of the 21-cm forest signal over a range of IGM parameter values (fixing one parameters and varying the other for each plot). The output is saved in plots/.<br>
For Fig. 7 in Šoltinský et al. 2025, MNRAS, 537, 364 - https://ui.adsabs.harvard.edu/abs/2025MNRAS.537..364S/abstract

<br>

*1DPS_plot_21cmFAST_novpec_200Mpc.py*

z_name = float(sys.argv[1])   #redshift
dvH = float(sys.argv[2])      #rebinning width in km/s
logfX = float(sys.argv[3])    #log10(f_X)
xHI_mean = float(sys.argv[4]) #<x_HI>

python 1DPS_plot_21cmFAST_novpec_200Mpc.py \<z_name\> \<dvH\> \<logfX\> \<xHI_mean\>

This code generates a plot of the 1D power spectrum of the 21-cm forest signal for a single IGM model with and without peculiar velocities. The output is saved in plots/.<br>
For Fig. B1 in Šoltinský et al. 2025, MNRAS, 537, 364 - https://ui.adsabs.harvard.edu/abs/2025MNRAS.537..364S/abstract
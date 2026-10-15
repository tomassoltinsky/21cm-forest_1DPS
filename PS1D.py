"""
1D PS dimensionless function.
Includes also periodogram/Lomb Scargle method for spectra with flagged channels.

Version 8.5.2026
"""

import numpy as np
from scipy.signal import lombscargle

def get_P(signal,max_size):

  n_pixels = len(signal)
  PowSpec = np.empty((int(n_pixels/2+1)), dtype=np.float64)

  contrast = signal-1     #signal contrast

  dft = np.fft.fft(contrast)
  dx = max_size/n_pixels
  dft = dx*dft/max_size

  k_fund = 2*np.pi/max_size

  kbins = np.arange(0,(n_pixels/2+1)*k_fund*(1.+1.e-100),k_fund)

  #power spectrum using Periodogram estimate
  PowSpec[0] = (np.abs(dft[0]))**2
  
  for p in range(int((n_pixels/2)-1)):
      PowSpec[p+1] = (np.abs(dft[p+1])**2 + np.abs(dft[n_pixels-p-1])**2)/2.0
  PowSpec[int(n_pixels/2)] = (np.abs(dft[int(n_pixels/2)]))**2

  PowSpec = PowSpec*max_size*kbins

  return kbins,PowSpec



def get_P_masked(signal, mask, max_size, nfreq=None):
    """
    Lomb-Scargle version of get_P() for masked 1D spectra.

    signal   : flux spectrum, expected mean around 1
    mask     : True = keep pixel, False = flagged pixel
    max_size : physical length of skewer / spectrum
    nfreq    : number of k modes; default matches FFT positive modes
    """

    signal = np.asarray(signal)
    mask = np.asarray(mask, dtype=bool)

    n_pixels = len(signal)
    dx = max_size / n_pixels

    # pixel positions in physical units
    x = np.arange(n_pixels) * dx

    # keep only unflagged pixels
    x_good = x[mask]
    y_good = signal[mask] - 1.0

    # remove mean of unflagged contrast
    y_good = y_good - np.mean(y_good)

    # same fundamental k as FFT
    k_fund = 2.0 * np.pi / max_size

    if nfreq is None:
        nfreq = n_pixels // 2

    # exclude k=0 for Lomb-Scargle
    k_modes = k_fund * np.arange(1, nfreq + 1)

    # scipy uses angular frequencies, so k_modes is already correct
    power_raw = lombscargle(
        x_good,
        y_good,
        k_modes,
        normalize=False
    )

    # Approximate normalization to match FFT convention:
    # scipy unnormalized LS gives ~ N_good A^2 / 4 for a sinusoid.
    # FFT convention gives ~ L A^2 / 4.
    N_good = len(y_good)
    Pk = (max_size / N_good) * power_raw

    # original code returns k * P(k)
    kPk = k_modes * Pk

    # add k=0 bin for compatibility
    kbins = np.concatenate([[0.0], k_modes])
    PowSpec = np.concatenate([[0.0], kPk])

    return kbins, PowSpec
"""
1D PS dimensionless function.

Version 19.10.2023
"""

import numpy as np
  
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
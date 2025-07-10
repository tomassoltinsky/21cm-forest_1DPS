"""
1D PS dimensionless function.

Version 19.10.2023
"""

import numpy as np
  
def get_P(signal,max_size, scaled=False):

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

  if scaled:
    PowSpec = PowSpec*max_size*kbins
  else:
    PowSpec = PowSpec
  return kbins,PowSpec

def get_P_set(signal,max_size, scaled=False):
  results = np.apply_along_axis(lambda row: get_P(row, max_size, scaled), 1, signal)
  #print(f"get_P_set: {results.shape}")
  return results[:,0], results[:,1]
  
   
def get_P_new(los,max_size, scaled=False):
    # Step 1: Normalize the transmission flux
    
    normalized_los = los # (los - los_mean) / los_std
    n_pixels = len(normalized_los)
    
    #normalized_los = los
    # Step 2: Calculate the power spectrum using FFT
    power_spectrum = np.abs(np.fft.fft(normalized_los))**2
    #print(f"1. power_spectrum={power_spectrum}")
    freq_axis = np.fft.fftfreq(n_pixels, d=(max_size/n_pixels))  # d is the spacing between frequency points
    freq_axis = freq_axis[:len(freq_axis)//2]
    #print(power_spectrum.shape)
    power_spectrum = power_spectrum[:n_pixels//2]
    
    # Step 3: Convert to dimensionless power spectrum
    #print(power_spectrum.shape)
    dimensionless_power_spectrum = power_spectrum * freq_axis
    #print(f"2. power_spectrum={dimensionless_power_spectrum}")    
    #print(dimensionless_power_spectrum.shape)
    return freq_axis,dimensionless_power_spectrum

def get_P_set_new(signal,max_size, scaled=False):
  results = np.apply_along_axis(lambda row: get_P_new(row, max_size, scaled), 1, signal)
  #print(f"get_P_set: {results.shape}")
  return results[:,0], results[:,1]
  
   

from concurrent.futures import ThreadPoolExecutor
import threading
from typing import List, Dict
import numpy as np
import PS1D
from scipy.stats import binned_statistic
from scipy import fftpack
import instrumental_features
import F21Stats
import logging
import time
import hashlib


class F21DataLoader:
    def __init__(self, max_workers: int = 4, psbatchsize: int = 1000, limitsamplesize: int = None, skip_ps: bool = False, ps_bins = None, ps_smoothing=True, skip_stats=True, use_bispectrum=False, scale_ps = False, input_points_to_use=None, perc_bins_to_use=100, use_new_ps_calc=False, shuffle_samples = False, ps_log_bins = False):
        np.random.seed(42)
        self.max_workers = max_workers
        self.collector = ThreadSafeArrayCollector()
        self.psbatchsize = psbatchsize
        self.limitsamplesize = limitsamplesize
        self.skip_ps = skip_ps
        self.ps_bins = ps_bins
        self.ps_smoothing = ps_smoothing
        self.skip_stats = skip_stats
        self.use_bispectrum = use_bispectrum
        self.scale_ps = scale_ps
        self.input_points_to_use = input_points_to_use
        self.perc_bins_to_use = perc_bins_to_use
        self.use_new_ps_calc = use_new_ps_calc
        self.shuffle_samples = shuffle_samples
        self.ps_log_bins = ps_log_bins

    def get_los(self, datafile: str) -> None:
        data = np.fromfile(str(datafile), dtype=np.float32)
    
        # Extract parameters
        i = 0
        z, xHI_mean, logfX = -999, -999, -999
        if "noiseonly" not in datafile:
            z = data[i]
            i += 1        # redshift
            xHI_mean = data[i] # mean neutral hydrogen fraction
            i += 1       
            logfX = data[i]    # log10(f_X)
            i += 1       
        Nlos = int(data[i])# Number of lines-of-sight
        i += 1
        Nbins = int(data[i])# Number of pixels/cells/bins in one line-of-sight
        x_initial = i + 1

        if len(data) != x_initial + (1+Nlos)*Nbins:
            error_msg = f"Error: Found {len(data)} fields, expected {x_initial + (1+Nlos)*Nbins:}. x_initial={x_initial}, Nlos={Nlos}, Nbins={Nbins}. File may be corrupted: {datafile}"
            raise ValueError(error_msg)
        # Find skipcount
        skipcount = 0
        for d in data[x_initial:]:
            if d > 1e7:
                skipcount += 1
                continue
            else:
                break

        if skipcount != Nbins:
            error_msg = f"Error: Found {skipcount} fields > 1e7 after x_initial, expected {Nbins}. File may be corrupted: {datafile}"
            raise ValueError(error_msg)

        # Extract frequency axis and F21 data
        freq_axis = data[(x_initial+0*Nbins):(x_initial+1*Nbins)]
        los_arr = np.reshape(data[(x_initial+1*Nbins):(x_initial+1*Nbins+Nlos*Nbins)],(Nlos,Nbins))

        if self.input_points_to_use is not None and los_arr.shape[-1] > self.input_points_to_use: 
            los_arr = los_arr[:self.input_points_to_use]
        return (z, xHI_mean, logfX, freq_axis, los_arr)


    def aggregate(self, dataseries):
        # Calculate mean and standard deviation across all samples
        mean = np.mean(dataseries, axis=0)
        std = np.std(dataseries, axis=0)
        samples_len = 5
        if len(dataseries) < 5:
            samples_len = len(dataseries)
        return (mean, std, dataseries[:samples_len])
    
    def logbin_power_spectrum(ks, ps):
        #Generate k bins
        d_log_k_bins = 0.25
        log_k_bins = np.arange(-1.0-d_log_k_bins/2.,3.+d_log_k_bins/2.,d_log_k_bins)
        log_k_bins = log_k_bins[1:]

        k_bins = np.power(10.,log_k_bins)
        k_bins_cent = np.power(10.,log_k_bins+d_log_k_bins/2.)[:-1]
        PS_signal_sim = np.empty((len(files),len(k_bins_cent)))
        

    
    @staticmethod
    def calculate_power_spectrum(data, sampling_rate=1.0):
        """
        Calculate the 1D power spectrum of a 1D series.

        Parameters:
        - data (array-like): The 1D input series.
        - sampling_rate (float): Sampling rate of the data (default is 1.0).

        Returns:
        - freqs (numpy.ndarray): Array of frequency bins.
        - power_spectrum (numpy.ndarray): Power spectrum corresponding to the frequencies.
        """
        # Number of data points
        n = len(data)
        
        # Perform the Fast Fourier Transform (FFT)
        fft_result = np.fft.fft(data)
        
        # Compute the power spectrum (magnitude squared of FFT components)
        power_spectrum = np.abs(fft_result)**2
        
        # Normalize the power spectrum
        power_spectrum = power_spectrum / n
        
        # Compute the corresponding frequencies
        freqs = np.fft.fftfreq(n, d=1/sampling_rate)
        
        # Return only the positive frequencies
        positive_freqs = freqs[:n // 2]
        positive_power_spectrum = power_spectrum[:n // 2]
        
        return positive_freqs, positive_power_spectrum
    
    def power_spectrum_1d(data, bins=10):
        """
        Calculate the 1D binned power spectrum of an array.

        Parameters:
        data: 1D array of data
        bins: Number of bins, or array of bin edges

        Returns:
        k: Array of wavenumbers (bin centers)
        power: Array of power spectrum values
        """

        # Calculate the Fourier transform of the data
        fft_data = fftpack.fft(data)

        # Calculate the power spectrum
        power = np.abs(fft_data)**2

        # Calculate the wavenumbers
        k = fftpack.fftfreq(len(data))

        # Bin the power spectrum
        power, bin_edges, _ = binned_statistic(np.abs(k), power, statistic='mean', bins=bins)
        k = 0.5 * (bin_edges[1:] + bin_edges[:-1])  # Bin centers

        return k, power

    def process_file(self, datafile: str) -> None:
        try:
            #print(f"Reading file: {datafile}")
            (z, xHI_mean, logfX, freq_axis, los_arr) = self.get_los(datafile)
            # Store the data
            #all_F21.append(F21_current)
            #print(F"Los loaded: {xHI_mean} , {logfX}, {freq_axis}, {los_arr.shape}")
            bandwidth = freq_axis[-1]-freq_axis[0]
            power_spectrum = []
            bispectrum = []
            k_bispec = None
            cumulative_los = []
            ks = None
            psbatchnum = 0
            samplenum = 0
            spec_res = 8 # kHz

            # Used for bispectrum calculation
            if self.limitsamplesize is not None and len(los_arr) > self.limitsamplesize:
                if self.shuffle_samples:
                    los_arr = los_arr[np.random.randint(len(los_arr), size=self.limitsamplesize)]
                else:
                    los_arr = los_arr[range(self.limitsamplesize)]#
            Nlos = len(los_arr)
 
            Nbins = len(freq_axis)
            """
            print('Number of pixels (original): %d' % Nbins)
            freq_uni = instrumental_features.uni_freq(freq_axis,np.array([freq_axis]))[0]
            freq_smooth = instrumental_features.smooth_fixedbox(freq_uni,freq_uni, spec_res)[0] # 8kHz Spectral resolution
            bandwidth = (freq_smooth[-1]-freq_smooth[0])/1e6
            print('Number of pixels (smoothed): %d' % len(freq_smooth))
            print('Bandwidth = %.2fMHz' % bandwidth)
            #n_kbins = int((len(freq_smooth)/2+1))
            signal_ori = instrumental_features.transF(los_arr)
            freq_uni,signal_uni = instrumental_features.uni_freq(freq_axis,signal_ori) #Interpolate signal to uniform frequency grid
            """
            params = np.array([xHI_mean, logfX])
            for j in range(Nlos):
                key = f"{xHI_mean:.2f}_{logfX:.2f}_{j}"
                psbatchnum += 1
                samplenum += 1
                los=los_arr[j]
                
                cumulative_los.append(los)
                if not self.skip_ps:
                    """
                    if self.ps_smoothing:
                        freq_smooth,signal_smooth = instrumental_features.smooth_fixedbox(freq_uni,los, spec_res) #Incorporate spectral resolution of telescope
                        freq_smooth = freq_smooth[:-1]
                        los = signal_smooth[:-1]
                    """
                    # Calculate the power spectrum
                    if self.use_new_ps_calc:
                        ks,ps = PS1D.get_P_new(los,bandwidth, scaled=self.scale_ps) #Calculate 1D power spectrum
                    else:
                        ks,ps = PS1D.get_P(los,bandwidth, scaled=self.scale_ps) #Calculate 1D power spectrum
                    power_spectrum.append(ps)

                if samplenum >= Nlos or psbatchnum >= self.psbatchsize:
                    # Collect this batch
                    #print(f"Collecting batch for params {params}")
                    cumulative_los_np = np.array(cumulative_los)
                    ps_mean, ps_std, ps_samples, stats_mean, bs_mean = None, None, None, None, None
                    if not self.skip_ps: (ps_mean, ps_std, ps_samples) = self.aggregate(np.array(power_spectrum))
                    if self.ps_log_bins:
                        ks, ps_mean = F21Stats.logbin_power_spectrum_by_k(ks, ps_mean, silent= (j!=0))
                        # logging.info(f"Shape of ps_mean: {ps_mean.shape}")
                        if ps_mean.shape[0] == 1: ps_mean = ps_mean.squeeze(axis=0)
                    if self.use_bispectrum: 
                        #start_time = time.perf_counter()
                        k_bispec, bs = F21Stats.F21Stats.compute_1d_bispectrum(cumulative_los_np)
                        if self.ps_bins is not None:
                            k_bispec, bs = F21Stats.logbin_power_spectrum_by_k_flex(k_bispec, bs, self.ps_bins, self.perc_bins_to_use)
                            
                        if bs.ndim > 1: bs_mean = np.mean(bs, axis=0)
                        else: bs_mean = bs
                        if j == 0: logging.info(f"Bispec shape {k_bispec.shape}, {bs.shape}") 
                        #end_time = time.perf_counter() 
                        #print(f"Calculated Bispectrum: {cumulative_los_np.shape}, {k_bispec.shape}, {bs_mean.shape} in {end_time - start_time:.8f} seconds")
                    
                    (los_mean, los_std, los_samples) = self.aggregate(cumulative_los_np)
                    if not self.skip_stats:
                        curr_statcalc = F21Stats.F21Stats.calculate_stats_torch(cumulative_los_np, params, kernel_sizes=[268])
                        stats_mean = np.mean(curr_statcalc, axis=0)
                        #print(stats_mean)

                    if ks is not None: ks = ks[0]
                    self.collector.add_data(ks, ps_mean, ps_std, los_mean, los_std, freq_axis, params, los_samples, ps_samples, stats_mean, bs_mean, k_bispec, key)
                    psbatchnum = 0
                    power_spectrum = []
                    bispectrum = []
                    cumulative_los = []

        except Exception as e:
            logging.error(f"Error processing {datafile}: {str(e)}", exc_info=True)
            
    def process_all_files(self, file_list: List[str]) -> Dict[str, np.ndarray]:
        # Process files in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all files for processing
            futures = [executor.submit(self.process_file, filepath) 
                      for filepath in file_list]
            
            # Wait for all tasks to complete
            for future in futures:
                future.result()  # This will raise any exceptions that occurred
                
        # Return the collected results
        return self.collector.get_arrays()

class ThreadSafeArrayCollector:
    def __init__(self):
        self._data = {
            'key': [],
            'ks': [],
            'ps': [],
            'ps_std': [],
            'los': [],
            'los_std': [],
            'freq_axis': [],
            'params': [],
            'los_samples': [],
            'ps_samples': [],
            'stats': [],
            'bispectrum': [],
            'k_bispec': [],
        }
        self._lock = threading.Lock()
        
    def add_data(self, ks, ps, ps_std, los, los_std, freq_axis, params, los_samples, ps_samples, stats, bispectrum, k_bispec, key=''):
        with self._lock:
            self._data['key'].append(key)
            self._data['ks'].append(ks)
            self._data['ps'].append(ps)
            self._data['ps_std'].append(ps_std)
            self._data['los'].append(los)
            self._data['los_std'].append(los_std)
            self._data['freq_axis'].append(freq_axis)
            self._data['params'].append(params)
            self._data['los_samples'].append(los_samples)
            self._data['ps_samples'].append(ps_samples)
            self._data['stats'].append(stats)
            self._data['bispectrum'].append(bispectrum)
            self._data['k_bispec'].append(k_bispec)
            
    def get_arrays(self):
        with self._lock:
            return {
                'key': np.array(self._data['key']),
                'ks': np.array(self._data['ks']),
                'ps': np.array(self._data['ps']),
                'ps_std': np.array(self._data['ps_std']),
                'los': np.array(self._data['los']),
                'los_std': np.array(self._data['los_std']),
                'freq_axis': np.array(self._data['freq_axis']),
                'params': np.array(self._data['params']),
                'los_samples': np.array(self._data['los_samples']),
                'ps_samples': np.array(self._data['ps_samples']),
                'stats': np.array(self._data['stats']),
                'bispectrum': np.array(self._data['bispectrum']),
                'k_bispec': np.array(self._data['k_bispec']),
            }

import scipy.signal
import numpy as np


def get_time_series_envelope(time_series, sample_rate, bin_size=40):
    num_bins = time_series.size // bin_size
    last_bin_size = time_series.size % bin_size
    out = np.zeros(time_series.size, dtype=time_series.dtype)

    abs_t_s = np.abs(time_series)
    for n in np.arange(num_bins):
        out[n*bin_size:(n+1)*bin_size] = np.max(abs_t_s[n*bin_size:(n+1)*bin_size]) * np.ones(bin_size)

    out[-last_bin_size:] = np.max(abs_t_s[-last_bin_size:]) * np.ones(last_bin_size)

    sos_filter = scipy.signal.butter(1, 40, output='sos', fs=sample_rate)

    return scipy.signal.sosfiltfilt(sos_filter, out)
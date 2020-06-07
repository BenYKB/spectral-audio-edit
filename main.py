import numpy as np
import cmath
import math
import matplotlib.pyplot as plt
from scipy.io import wavfile

from FunctionGenerator import FunctionGenerator
from MorletWavelet import MorletWavelet

def convolve(signal, kernel):
    signal_pad_length = kernel.size - 1
    signal_padded = np.pad(signal, (signal_pad_length, signal_pad_length))
    kernel_padded = np.pad(kernel, (0, signal_padded.size - kernel.size))

    f_signal = np.fft.fft(signal_padded)
    f_kernel = np.fft.fft(kernel_padded)

    f_result = f_signal * f_kernel

    return np.delete(np.fft.ifft(f_result), np.arange(0, signal_pad_length))

def resample():
    pass




# TODO: scaling bounds based on frequencies of interest
# TODO: resampler function
# TODO: resample function based on wavelet scaling
# TODO: resample based on wavelet frequency
# TODO: use minimal sampling required for reconstruction
# TODO: precise reconstruction
# TODO: approximate reconstruction


def wavelet_transform(signal, wavelet, sample_rate, voices_per_octave):
    def g_s(s):
        def g(t):
            return 1.0 / cmath.sqrt(s) * np.conj(wavelet.apply((-1 * t) / s))
        return g

    s_min = wavelet.frequency * 2 / sample_rate
    s_max = min(signal.size / (2 * wavelet.radius * sample_rate), wavelet.frequency / 20)
    s_0 = 2 ** (1 / voices_per_octave)

    def s(t):
        return s_0 ** t

    def kernel_radius(s):
        return wavelet.radius * s

    s_func = np.vectorize(s)

    scale_power_min = np.ceil(voices_per_octave * np.log2(s_min))
    scale_power_max = np.floor(voices_per_octave * np.log2(s_max))
    scale_powers = np.arange(scale_power_min, scale_power_max)
    scales = s_func(scale_powers)

    f_signal = None
    max_convolution_length = None
    t_0_offset = None
    output = None

    for i in range(scales.size - 1, -1, -1):
        s = scales[i]
        g_s_func = np.vectorize(g_s(s))
        t_decay_s = kernel_radius(s)
        kernel_times = np.arange(-t_decay_s, t_decay_s, 1.0 / sample_rate)
        kernel_s = g_s_func(kernel_times)

        if max_convolution_length is None or t_0_offset is None or f_signal is None:
            max_convolution_length = signal.size + kernel_times.size - 1
            t_0_offset = kernel_times.size // 2
            output = np.zeros((scales.size, max_convolution_length), np.complex64)
            signal_padded = np.pad(signal, (0, max_convolution_length - signal.size))
            kernel_padded = np.pad(kernel_s, (0, max_convolution_length - kernel_s.size))
            f_signal = np.fft.fft(signal_padded)
        else:
            left_padding = int(t_0_offset - kernel_s.size // 2)
            kernel_padded = np.pad(kernel_s, (left_padding, max_convolution_length - kernel_s.size - left_padding))

        f_kernel = np.fft.fft(kernel_padded)
        result_s = np.fft.ifft(f_signal * f_kernel)
        output[i] = result_s

    times = np.arange(0, max_convolution_length) - t_0_offset

    return output, scales, times


def morlet_wavelet_transform(sig, sample_rate, freq=1, sigma=1):
    tp = 2 * np.pi
    f = freq
    d = sigma
    nyquist_frequency = sample_rate / 2
    lower_bound_frequency = 20
    n_decays_at_bound = 3

    q = (tp*f + 2 ** .5 / (2 * d)) / (tp*f - 2 ** .5 / (2 * d))

    s_max = (f - n_decays_at_bound / (tp * d)) / lower_bound_frequency
    s_min = (f + n_decays_at_bound / (tp * d)) / nyquist_frequency

    max_pow = int(math.log(s_max/s_min, q))
    scales = np.logspace(0, max_pow, base=q, num=max_pow+1) * s_min

    def morlet_FT(s):
        def func(w):
            return (tp*s) ** 0.5 * d * np.exp(-tp*np.pi * d ** 2 * (f - w * s) ** 2)
        return func

    sig_FT = np.fft.fft(signal)
    N = sig_FT.size
    sig_FT_frequencies = sample_rate / N * np.arange(0, N)

    frequencies = f / scales
    output = np.empty((scales.size, N), dtype=np.complex64)

    for i in range(scales.size):
        s = scales[i]
        morlet_FT_func = np.vectorize(morlet_FT(s))
        filter = morlet_FT_func(sig_FT_frequencies)
        product = filter * sig_FT
        output[i] = np.fft.ifft(product)

    return output, frequencies



if __name__ == '__main__':
    sample_rate, signal = wavfile.read("Recordings\Arpeggio_mono.wav")

    phi = MorletWavelet(1,2*np.pi,6)

    output, freqs = morlet_wavelet_transform(signal, sample_rate, freq=2)
    print("f min:", np.min(freqs))
    print("f max:", np.max(freqs))
    amplitude_spec = np.absolute(output)
    plt.imshow(amplitude_spec, aspect='auto')
    plt.show()




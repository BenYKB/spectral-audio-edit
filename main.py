import numpy as np
import cmath
import matplotlib.pyplot as plt

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

def wavelet_transform(signal, wavelet, sample_rate, voices_per_octave):
    def g_s(s):
        def g(t):
            return 1.0 / cmath.sqrt(s) * np.conj(wavelet.apply((-1 * t) / s))
        return g

    s_min = wavelet.frequency * 2 / sample_rate
    s_max = signal.size / (2 * wavelet.radius * sample_rate)
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

    max_convolution_length = None
    t_0_offset = None
    output = None

    for i in range(scales.size - 1, -1, -1):
        s = scales[i]
        g_s_func = np.vectorize(g_s(s))
        t_decay_s = kernel_radius(s)
        kernel_times = np.arange(-t_decay_s, t_decay_s, 1.0 / sample_rate)
        kernel_s = g_s_func(kernel_times)

        wt_s = convolve(signal, kernel_s)

        if max_convolution_length is None or t_0_offset is None:
            t_0_offset = int(np.ceil(kernel_times.size / 2))
            print(t_0_offset)
            max_convolution_length = int(wt_s.size)
            print(max_convolution_length)
            output = np.zeros((scales.size, max_convolution_length), np.complex128 )

        else:
            left_padding = int(t_0_offset - np.ceil(kernel_times.size / 2))
            wt_s = np.pad(wt_s, (left_padding, max_convolution_length - left_padding - wt_s.size))

        output[i] = wt_s

    offsets = np.arrange(0, max_convolution_length) - t_0_offset

    return output, scales, offsets

if __name__ == '__main__':
    signal = np.array([0,0,0,0,1,-1,1,-1,1,-1,1,0,0,0,0,0])
    phi = MorletWavelet(5,1,3)

    spectrogram, scales, offset = wavelet_transform(signal, phi, 5, 2)
    print(spectrogram.shape)
    amplitude_spec = np.absolute(spectrogram)
    print(amplitude_spec)
    plt.imshow(amplitude_spec, cmap='hot')
    plt.show()



